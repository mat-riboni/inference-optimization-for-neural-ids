import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class DenseBranchyNet(nn.Module):
    """
    BranchyNet con 4 hidden layer e una sola early-exit dopo il primo layer.
    - Head1: early-exit
    - Head4: finale

    Novità:
    - self.T1: temperatura per calibrare le probabilità di head1 (buffer persistente).
    - early-exit deciso su p_T = softmax(l1 / T1), usando:
        * mode="prob_margin": conf = p_top1 - p_top2 (default robusto)
        * mode="entropy":     conf = 1 - Hn(p_T)
    - calibrate_tau: fitta T1 su validation + sceglie tau globale o per-classe.
    """

    def __init__(self, hidden_layers_sizes, taus, alphas,
                 cat_cardinalities, embedding_dims,
                 num_numerical, num_target_classes):
        super().__init__()

        assert len(hidden_layers_sizes) == 4, "hidden_layers_sizes deve avere 4 dimensioni"
        assert len(taus) == 1, "serve una sola tau (early-exit dopo layer1)"
        assert len(alphas) == 2, "alphas deve avere 2 pesi: [alpha_head1, alpha_head4]"

        # Embedding categorical
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, emb_dim)
            for cat_cardinality, emb_dim in zip(cat_cardinalities, embedding_dims)
        ])

        total_emb_size = sum(embedding_dims)
        input_size = total_emb_size + num_numerical

        # Trunk a 4 layer
        self.fc1 = nn.Linear(input_size,            hidden_layers_sizes[0])
        self.head1 = nn.Linear(hidden_layers_sizes[0], num_target_classes)  # early-exit head

        self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.fc3 = nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2])
        self.fc4 = nn.Linear(hidden_layers_sizes[2], hidden_layers_sizes[3])
        self.head4 = nn.Linear(hidden_layers_sizes[3], num_target_classes)  # final head

        # Soglia early-exit (buffer per essere salvata col state_dict)
        self.register_buffer("tau1", torch.tensor(float(taus[0]), dtype=torch.float32))
        # Temperatura per softmax calibrata (buffer persistente)
        self.register_buffer("T1", torch.tensor(1.0, dtype=torch.float32))

        # soglie per-classe (dict non finisce nello state_dict)
        self.tau1_by_class = None

        # Per salvataggio
        self.hidden_layers_sizes = hidden_layers_sizes
        self.cat_cardinalities = cat_cardinalities
        self.embedding_dims = embedding_dims
        self.num_numerical = num_numerical
        self.num_target_classes = num_target_classes

        
        self.alphas = alphas
        self.taus = taus  # anche in forma lista per metadata


    def forward(self,
                x_num: torch.Tensor,
                x_cat: torch.Tensor,
                branchy: bool = False,
                use_margin: bool = True,
                return_logits: bool = False):
        """
        branchy=False: restituisce (logits_head1, logits_head4) per il training.
        branchy=True:  early-exit dopo head1 se confidenza >= tau1 (o tau per-classe),
                       altrimenti prosegue fino a head4.
                       return_logits=False -> class indices [B]
                       return_logits=True  -> logits finali [B, C]
        """
        x = self._embed_input(x_num, x_cat)
        h1 = F.relu(self.fc1(x))
        l1 = self.head1(h1)

        if not branchy:
            # percorso completo senza decisioni
            h2 = F.relu(self.fc2(h1))
            h3 = F.relu(self.fc3(h2))
            h4 = F.relu(self.fc4(h3))
            l4 = self.head4(h4)
            return l1, l4

        # ---- branchy inference con una sola early-exit (dopo head1) ----
        B = x_num.size(0)
        C = self.head1.out_features
        device = l1.device

        # conf/pred/exit basati su probabilità calibrate con T1
        conf1, pred1, exit1 = self._confidence_and_exit(l1, use_margin=use_margin)

        if return_logits:
            out = torch.empty((B, C), dtype=l1.dtype, device=device)
        else:
            out = torch.empty((B,), dtype=torch.long, device=device)

        # assegna chi esce a head1
        if exit1.any():
            if return_logits:
                out[exit1] = l1[exit1]   # (volendo si potrebbero salvare i logit temperati, ma non serve)
            else:
                out[exit1] = pred1[exit1]

        # chi non esce prosegue fino a head4
        keep1 = ~exit1
        if keep1.any():
            idx = torch.where(keep1)[0]
            h2 = F.relu(self.fc2(h1.index_select(0, idx)))
            h3 = F.relu(self.fc3(h2))
            h4 = F.relu(self.fc4(h3))
            l4 = self.head4(h4)
            if return_logits:
                out[idx] = l4
            else:
                out[idx] = l4.argmax(dim=1)

        return out

    # -------------------------
    # Training helpers
    # -------------------------
    def fit(self, train_loader, valid_loader, optimizer, device,
            epochs=10, criterion=None, scheduler=None,
            eval_exit_stats=True, verbose=True, use_margin=True):
        """
        Allena con loss multi-head: alpha1 * CE(head1) + alpha4 * CE(head4).
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # se non viene passato, usiamo un ReduceLROnPlateau su F1
        if scheduler is None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)

        history = []
        for ep in range(1, epochs + 1):
            self.train()
            run = {"loss": 0.0, "loss1": 0.0, "loss4": 0.0}
            n_batches = 0

            for x_num, x_cat, y in train_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device).long()
                y = y.to(device).long()

                optimizer.zero_grad()
                out1, out4 = self(x_num, x_cat, branchy=False)

                loss1 = criterion(out1, y)
                loss4 = criterion(out4, y)
                loss = self.alphas[0] * loss1 + self.alphas[1] * loss4

                loss.backward()
                optimizer.step()

                run["loss"] += loss.item()
                run["loss1"] += loss1.item()
                run["loss4"] += loss4.item()
                n_batches += 1

            for k in run:
                run[k] /= max(n_batches, 1)

            val_metrics = self.evaluate(valid_loader, device, return_exit_stats=eval_exit_stats, use_margin=use_margin)
            if scheduler is not None:
                scheduler.step(metrics=val_metrics["f1_weighted"])

            log = {"epoch": ep, **run,
                   "val_accuracy": val_metrics["accuracy"],
                   "val_f1_weighted": val_metrics["f1_weighted"]}
            if eval_exit_stats:
                log["val_exit_counts"] = val_metrics["val_exit_counts"]
                log["val_exit_rate"] = val_metrics["val_exit_rate"]

            history.append(log)
            if verbose:
                print(f"[Ep {ep:03d}] loss={run['loss']:.4f} (l1={run['loss1']:.4f} l4={run['loss4']:.4f}) "
                      f"| val_acc={log['val_accuracy']:.4f} val_f1={log['val_f1_weighted']:.4f}")
        return history

    @torch.no_grad()
    def evaluate(self, dataloader, device, return_exit_stats=True, use_margin=True):
        self.eval()
        all_preds, all_labels = [], []
        exit_counts = {1: 0, 4: 0}

        for x_num, x_cat, y in dataloader:
            x_num = x_num.to(device); x_cat = x_cat.to(device).long(); y = y.to(device).long()

            x = self._embed_input(x_num, x_cat)
            h1 = F.relu(self.fc1(x)); l1 = self.head1(h1)

            # usa probabilità calibrate (T1) + tau (globale o per-classe)
            _, pred1, exit1 = self._confidence_and_exit(l1, use_margin=use_margin)

            preds = torch.empty(x_num.size(0), dtype=torch.long, device=device)

            if exit1.any():
                preds[exit1] = pred1[exit1]
                exit_counts[1] += int(exit1.sum().item())

            keep = ~exit1
            if keep.any():
                idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
                h2 = F.relu(self.fc2(h1[idx]))
                h3 = F.relu(self.fc3(h2))
                h4 = F.relu(self.fc4(h3))
                l4 = self.head4(h4)
                preds[idx] = l4.argmax(1)
                exit_counts[4] += int(idx.numel())

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        out = {"accuracy": acc, "f1_weighted": f1}
        if return_exit_stats:
            total = sum(exit_counts.values()) or 1
            out["val_exit_counts"] = exit_counts
            out["val_exit_rate"] = {k: v / total for k, v in exit_counts.items()}
        return out

    # -------------------------
    # Inference helpers
    # -------------------------
    @torch.inference_mode()
    def predict_using_margin(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # batch=1 fast path
        x = self._embed_input(x_num, x_cat)
        h1 = F.relu(self.fc1(x), inplace=True)
        l1 = self.head1(h1)

        # conf: probability-margin su softmax(l1/T1)
        v, idx = torch.topk(F.softmax(l1 / self.T1, dim=1), 2, dim=1)
        conf = (v[:, 0] - v[:, 1]).item()
        if conf >= float(self.tau1.item()):
            # esci subito
            return idx[:, 0].to(l1.device)

        # altrimenti prosegui
        h2 = F.relu(self.fc2(h1), inplace=True)
        h3 = F.relu(self.fc3(h2), inplace=True)
        h4 = F.relu(self.fc4(h3), inplace=True)
        l4 = self.head4(h4)
        return l4.argmax(1)

    @torch.inference_mode()
    def predict_using_entropy(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # batch=1 fast path
        x = self._embed_input(x_num, x_cat)
        h1 = F.relu(self.fc1(x), inplace=True)
        l1 = self.head1(h1)

        p1 = F.softmax(l1 / self.T1, dim=1)
        Hn = -(p1.clamp_min(1e-12).log() * p1).sum(dim=1) / math.log(self.num_target_classes)
        conf = (1.0 - Hn).item()
        if conf >= float(self.tau1.item()):
            return p1.argmax(1)

        h2 = F.relu(self.fc2(h1), inplace=True)
        h3 = F.relu(self.fc3(h2), inplace=True)
        h4 = F.relu(self.fc4(h3), inplace=True)
        l4 = self.head4(h4)
        return l4.argmax(1)


    @torch.inference_mode()
    def predict_all(self, dataloader, device="cpu", use_margin: bool = True):
        dev = torch.device(device)
        self.to(dev).eval()
        all_preds = []
        for x_num, x_cat, _ in dataloader:
            x_num = x_num.to(dev)
            x_cat = x_cat.to(dev).long()
            preds = (self.predict_using_margin if use_margin else self.predict_using_entropy)(x_num, x_cat)
            all_preds.append(preds.cpu())
        return torch.cat(all_preds, dim=0)

    # -------------------------
    # Utilities
    # -------------------------
    def set_tau(self, t1: float):
        self.tau1.fill_(float(t1))
        self.taus = [float(t1)]
        # se stai usando tau per-classe, azzera per evitare ambiguità
        self.tau1_by_class = None

    def save(self, path: str):
        payload = {
            "model_state": self.state_dict(),  # include tau1 e T1
            "meta": {
                "hidden_layers_sizes": self.hidden_layers_sizes,
                "alphas": self.alphas,
                "taus": [float(self.tau1.item())],
                "cat_cardinalities": self.cat_cardinalities,
                "embedding_dims": self.embedding_dims,
                "num_numerical": self.num_numerical,
                "num_target_classes": self.num_target_classes,
            },
            "extra": {
                "tau1_by_class": self.tau1_by_class  # può essere None o dict {int: float}
            }
        }
        torch.save(payload, path)
        print(f"Modello salvato in '{path}'.")

    @classmethod
    def load(cls, path: str, device: torch.device):
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        extra = payload.get("extra", {}) or {}
        model = cls(**meta)
        model.load_state_dict(payload["model_state"])
        model.tau1_by_class = extra.get("tau1_by_class", None)
        model.to(device).eval()
        print(f"Modello {cls.__name__} caricato da '{path}'.")
        return model

    @staticmethod
    def _logit_margin(logits):
        top2 = logits.topk(2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def set_stage(self, stage: str):
        """
        stage in {"stage0_trunk_final", "stage1_head1_only", "stage2_finetune_all"}
        - stage0: allena trunk (fc1..fc4) + head4, congela head1
        - stage1: allena solo head1, congela trunk + head4
        - stage2: allena tutto
        """
        for p in self.parameters():
            p.requires_grad = True

        head1 = list(self.head1.parameters())
        head4 = list(self.head4.parameters())
        trunk = list(self.fc1.parameters()) + list(self.fc2.parameters()) + \
                list(self.fc3.parameters()) + list(self.fc4.parameters())

        if stage == "stage0_trunk_final":
            for p in head1: p.requires_grad = False
        elif stage == "stage1_head1_only":
            for p in trunk + head4: p.requires_grad = False
        elif stage == "stage2_finetune_all":
            pass
        else:
            raise ValueError("stage non valido")

    def _embed_input(self, x_num, x_cat):
        if len(self.embeddings) > 0:
            embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_e = torch.cat(embedded, dim=1)
            x = torch.cat([x_num, x_cat_e], dim=1)
        else:
            x = x_num
        return x

    # ---------------------------------------------------
    # CONFIDENCE + EXIT (temperatura inclusa)
    # ---------------------------------------------------
    def _confidence_and_exit(self, l1: torch.Tensor, use_margin: bool):
        """
        Ritorna:
          conf:   [B] score in [0,1]
          pred1:  [B] classe top-1 su p_T
          exit_mask: [B] bool, decisione early-exit usando tau scalare o per-classe
        """
        # Probabilità calibrate con T1
        p1 = F.softmax(l1 / self.T1, dim=1)
        pred1 = p1.argmax(dim=1)

        if use_margin:
            v, _ = torch.topk(p1, 2, dim=1)
            conf = (v[:, 0] - v[:, 1]).clamp_(0, 1)  # probability-margin
        else:
            logp1 = (p1 + 1e-12).log()
            Hn = -(p1 * logp1).sum(dim=1) / math.log(self.num_target_classes)
            conf = (1.0 - Hn).clamp_(0, 1)           # 1 - entropia normalizzata

        # Applica tau globale o per-classe
        if isinstance(self.tau1_by_class, dict) and len(self.tau1_by_class) > 0:
            tau_vec = torch.ones_like(conf) * float(self.tau1.item())
            for c, tau_c in self.tau1_by_class.items():
                tau_vec[pred1 == int(c)] = float(tau_c)
            exit_mask = conf >= tau_vec
        else:
            exit_mask = conf >= self.tau1
        return conf, pred1, exit_mask

    # ---------------------------------------------------
    # CALIBRAZIONE (T e tau)
    # ---------------------------------------------------
    def calibrate_tau(
        self,
        loader,
        device,
        baseline_f1: float,
        mode: str = "margin",   # "margin" o "entropy"
        per_class: bool = False,     # True -> tau per-classe prevista da head1
        n_grid: int = 101,           # griglia uniforme su [0,1]
        f1_drop: float = 0.01,
        relative: bool = False,
        return_curve: bool = True,
        max_iter_temp: int = 50,
    ):
        """
        Calibra (T1, tau) per l'early-exit di head1 su validation:
          1) Fit temperatura T1 minimizzando NLL di head1 (LBFGS).
          2) Costruisci confidenza in [0,1] da p_T (softmax(l1/T1)):
             - mode="margin": conf = p_top1 - p_top2
             - mode="entropy":     conf = 1 - Hn(p_T)
          3) Scansiona soglie su griglia uniforme [0,1] e scegli tau che massimizza l'exit-rate
             soggetta al vincolo su F1 (relativo o assoluto). Fallback: F1 massimo.

        Ritorna dict con T1, tau1 (float o dict per-classe), f1, accuracy, exit_rate,
        constraint, e (opzionale) curve di supporto.
        """
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score

        self.eval()
        dev = torch.device(device)

        # ---- 1) Colleziona logits/label su validation (no grad) ----
        logits1_list, logits4_list, y_list = [], [], []
        with torch.no_grad():
            for x_num, x_cat, y in loader:
                x_num = x_num.to(dev)
                x_cat = x_cat.to(dev).long()
                y     = y.to(dev).long()

                x  = self._embed_input(x_num, x_cat)
                h1 = F.relu(self.fc1(x)); l1 = self.head1(h1)
                h2 = F.relu(self.fc2(h1)); h3 = F.relu(self.fc3(h2)); h4 = F.relu(self.fc4(h3))
                l4 = self.head4(h4)

                logits1_list.append(l1.detach())
                logits4_list.append(l4.detach())
                y_list.append(y.detach())

        l1_all = torch.cat(logits1_list, dim=0).to(dev)
        l4_all = torch.cat(logits4_list, dim=0).to(dev)
        y_all  = torch.cat(y_list,  dim=0).to(dev)

        # ---- 2) Fit Temperature T1 su head1 (Guo et al. 2017) ----
        class _Temp(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_T = nn.Parameter(torch.zeros(1))  # T=1 di default
            def forward(self, logits):
                T = self.log_T.exp()
                return logits / T
            def T(self):
                return self.log_T.exp().item()

        temp = _Temp().to(dev)
        nll  = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(temp.parameters(), lr=0.25, max_iter=max_iter_temp)

        def _closure():
            optimizer.zero_grad()
            loss = nll(temp(l1_all), y_all)
            loss.backward()
            return loss

        optimizer.step(_closure)
        T1 = float(temp.T())
        self.T1.fill_(T1)  # aggiorna il buffer persistente

        # ---- 3) Costruisci confidenza calibrata su p_T ----
        with torch.no_grad():
            p1 = F.softmax(l1_all / T1, dim=1)
            top2 = torch.topk(p1, k=2, dim=1).values
            p_top1, p_top2 = top2[:, 0], top2[:, 1]

            if mode == "margin":
                conf = (p_top1 - p_top2).clamp(0, 1)
            elif mode == "entropy":
                logp1 = (p1 + 1e-12).log()
                Hn = -(p1 * logp1).sum(dim=1) / math.log(self.num_target_classes)
                conf = (1.0 - Hn).clamp(0, 1)
            else:
                raise ValueError("mode deve essere 'margin' o 'entropy'")

            pred1 = p1.argmax(1)
            pred4 = l4_all.argmax(1)

        # ---- 4) Vincolo F1 ----
        if relative:
            f1_min = baseline_f1 * (1.0 - f1_drop)
            constraint_str = f"F1 >= {100*(1.0 - f1_drop):.2f}% della baseline ({f1_min:.4f})"
        else:
            f1_min = baseline_f1 - f1_drop
            constraint_str = f"F1 >= baseline - {f1_drop:.4f} ({f1_min:.4f})"

        # ---- helper per valutare soglie ----
        def eval_thresholds_global(conf, pred1, pred4, y, tau_float):
            exit_mask = (conf >= tau_float)
            final_pred = pred4.clone()
            final_pred[exit_mask] = pred1[exit_mask]
            y_np  = y.detach().cpu().numpy()
            fp_np = final_pred.detach().cpu().numpy()
            f1 = f1_score(y_np, fp_np, average="weighted")
            acc = accuracy_score(y_np, fp_np)
            exit_rate = float(exit_mask.float().mean().item())
            return f1, acc, exit_rate

        def eval_thresholds_perclass(conf, pred1, pred4, y, taus_dict):
            exit_mask = torch.zeros_like(conf, dtype=torch.bool)
            for c, tau_c in taus_dict.items():
                m = (pred1 == int(c))
                if m.any():
                    exit_mask[m] = conf[m] >= float(tau_c)
            final_pred = pred4.clone()
            final_pred[exit_mask] = pred1[exit_mask]
            y_np  = y.detach().cpu().numpy()
            fp_np = final_pred.detach().cpu().numpy()
            f1 = f1_score(y_np, fp_np, average="weighted")
            acc = accuracy_score(y_np, fp_np)
            exit_rate = float(exit_mask.float().mean().item())
            return f1, acc, exit_rate

        # ---- 5) Scansione su griglia uniforme ----
        grid = torch.linspace(0.0, 1.0, steps=max(3, n_grid), device=dev)

        if not per_class:
            best = {"tau": 0.0, "f1": -1.0, "acc": -1.0, "exit": -1.0}
            curve = {"tau": [], "f1": [], "acc": [], "exit": []}

            # prima: tra le tau che rispettano il vincolo, scegliamo quella con exit-rate max
            for t in grid:
                f1v, accv, exv = eval_thresholds_global(conf, pred1, pred4, y_all, float(t.item()))
                if return_curve:
                    curve["tau"].append(float(t.item()))
                    curve["f1"].append(f1v); curve["acc"].append(accv); curve["exit"].append(exv)
                if f1v >= f1_min and (exv > best["exit"] or (math.isclose(exv, best["exit"]) and f1v > best["f1"])):
                    best = {"tau": float(t.item()), "f1": f1v, "acc": accv, "exit": exv}

            # fallback: se il vincolo è impossibile, prendi F1 massimo (tie-break: exit più alto)
            if best["f1"] < 0:
                for t in grid:
                    f1v, accv, exv = eval_thresholds_global(conf, pred1, pred4, y_all, float(t.item()))
                    if (f1v > best["f1"]) or (math.isclose(f1v, best["f1"]) and exv > best["exit"]):
                        best = {"tau": float(t.item()), "f1": f1v, "acc": accv, "exit": exv}

            # aggiorna tau globale nel buffer + azzera tau per-classe
            self.set_tau(best["tau"])

            result = {
                "T1": T1,
                "tau1": best["tau"],
                "f1": best["f1"],
                "accuracy": best["acc"],
                "exit_rate": best["exit"],
                "constraint": constraint_str,
            }
            if return_curve:
                result["curve"] = curve
            return result

        else:
            # per-classe: stima tau_c indipendente, poi verifica vincolo globale e rifinisce
            C = self.num_target_classes
            taus_dict = {}
            for c in range(C):
                m = (pred1 == c)
                if m.sum() < 5:  # pochi esempi -> non uscire quasi mai
                    taus_dict[c] = 1.0
                    continue
                best_c = {"tau": 1.0, "f1": -1.0, "exit": -1.0}
                for t in grid:
                    exit_mask = (conf[m] >= float(t.item()))
                    final_local = pred4[m].clone()
                    final_local[exit_mask] = pred1[m][exit_mask]
                    f1_local = f1_score(y_all[m].detach().cpu().numpy(),
                                        final_local.detach().cpu().numpy(),
                                        average="weighted")
                    ex_local = float(exit_mask.float().mean().item())
                    if (f1_local > best_c["f1"]) or (math.isclose(f1_local, best_c["f1"]) and ex_local > best_c["exit"]):
                        best_c = {"tau": float(t.item()), "f1": f1_local, "exit": ex_local}
                taus_dict[c] = best_c["tau"]

            f1v, accv, exv = eval_thresholds_perclass(conf, pred1, pred4, y_all, taus_dict)

            #  se il vincolo non è soddisfatto
            if f1v < f1_min:
                for _ in range(10):
                    worst = None
                    worst_gain = None
                    grid_np = grid.detach().cpu().numpy().tolist()
                    for c in range(C):
                        current_tau = taus_dict.get(c, 1.0)
                        # prossimo step di griglia
                        idx = 0
                        while idx < len(grid_np) and grid_np[idx] <= current_tau + 1e-12:
                            idx += 1
                        if idx >= len(grid_np):
                            continue
                        tau_trial = float(grid_np[idx])
                        taus_try = dict(taus_dict)
                        taus_try[c] = tau_trial
                        f1_t, acc_t, ex_t = eval_thresholds_perclass(conf, pred1, pred4, y_all, taus_try)
                        gain = f1_t - f1v
                        if (worst_gain is None) or (gain > worst_gain) or (math.isclose(gain, worst_gain) and ex_t > (worst["ex_t"] if worst else -1)):
                            worst = {"c": c, "tau": tau_trial, "ex_t": ex_t, "f1_t": f1_t, "acc_t": acc_t}
                            worst_gain = gain
                    if worst is None:
                        break
                    taus_dict[worst["c"]] = worst["tau"]
                    f1v, accv, exv = eval_thresholds_perclass(conf, pred1, pred4, y_all, taus_dict)
                    if f1v >= f1_min:
                        break

            # Memorizza le tau per-classe per l'inferenza
            self.tau1_by_class = {int(k): float(v) for k, v in taus_dict.items()}
            # Mantieni anche tau1 globale come fallback 
            fallback = float(sum(taus_dict.values()) / max(len(taus_dict), 1))
            self.tau1.fill_(fallback)
            self.taus = [fallback]

            result = {
                "T1": T1,
                "tau1": self.tau1_by_class,     # dict per classe prevista
                "f1": f1v,
                "accuracy": accv,
                "exit_rate": exv,
                "constraint": constraint_str,
            }
            if return_curve:
                #da finire
                pass
            return result
