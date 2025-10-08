import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from neural_network import TabularDataset, NeuralNetwork
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import numpy as np
from copy import deepcopy
import time
import copy
from sklearn.metrics import f1_score



def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        it = pd.read_csv(file_path,chunksize=100000, low_memory=True)
        data = pd.concat(it, ignore_index=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def split_data(data, target_col, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to split.
    target_col: Name of target column.
    
    Returns:
    tuple: A tuple containing the training, valid and testing DataFrames.
    """
    train_df, temp_df = train_test_split(data, test_size=0.3, random_state=random_state, stratify=data[target_col])
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df[target_col])   
    
    return train_df, valid_df, test_df

def remove_rows(data, values_to_remove):
    
    for col, values in values_to_remove.items():
        for val in values:
            data = data[data[col] != val]

    return data

def keep_columns(data, columns_to_keep):
    """
    Remove unnecessary columns from the DataFrame.
    
    Parameters:
    data (pd.DataFrame): The DataFrame from which to remove columns.
    columns_to_keep (list): List of column names to keep.
    
    Returns:
    pd.DataFrame: The DataFrame with specified columns removed.
    """
    return data[columns_to_keep]

def get_X_num(data, numerical_cols):
    """
    Get a DataFrame containing only the specified numeric columns.
    
    Parameters:
    data (pd.DataFrame): The original DataFrame.
    numerical_cols (list): List of numeric column names to include.
    
    Returns:
    pd.DataFrame: A DataFrame with only the specified numeric columns.
    """
    return data[numerical_cols]

def get_X_cat(data, categorical_cols):
    """
    Get a DataFrame containing only the specified categorical columns.
    
    Parameters:
    data (pd.DataFrame): The original DataFrame.
    categorical_cols (list): List of categorical column names to include.
    
    Returns:
    pd.DataFrame: A DataFrame with only the specified categorical columns.
    """
    return data[categorical_cols]


def get_weights(y):
    class_weights = compute_class_weight(class_weight="balanced",
                                     classes=np.unique(y.numpy()),
                                     y=y.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights



def load_and_prepare_data(file_path, target_col, numerical_cols, categorical_cols, rows_to_remove, batch_size=512):
    """
    Load and prepare the data for modeling.
    
    Parameters:
    file_path (str): The path to the CSV file.
    target_col: Name of target column.
    numerical_cols (list): List of numeric column names.
    categorical_cols (list): List of categorical column names.
    columns_to_keep (list): List of columns to keep in the final DataFrame.
    """

    data = load_data(file_path)
    if data is None:
        return None, None, None
    
    data = keep_columns(data, numerical_cols + categorical_cols + [target_col])
    data = remove_rows(data, rows_to_remove)

    data[target_col] = data[target_col].astype('category')
    class_names = data[target_col].cat.categories.tolist()

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    data[target_col] = data[target_col].cat.codes

    cat_cardinalities = [data[col].nunique() for col in categorical_cols]

    train_df, valid_df, test_df = split_data(data, target_col)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_cols])
    X_valid_num = scaler.transform(valid_df[numerical_cols])
    X_test_num  = scaler.transform(test_df[numerical_cols])
    
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_valid_num = torch.tensor(X_valid_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    
    X_train_cat = torch.tensor(get_X_cat(train_df, categorical_cols).values, dtype=torch.long)
    X_valid_cat = torch.tensor(get_X_cat(valid_df, categorical_cols).values, dtype=torch.long)
    X_test_cat = torch.tensor(get_X_cat(test_df, categorical_cols).values, dtype=torch.long)

    y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df[target_col].values, dtype=torch.long)
    y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    valid_dataset = TabularDataset(X_valid_num, X_valid_cat, y_valid)
    test_dataset = TabularDataset(X_test_num, X_test_cat, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names


def load_and_prepare_nb15(file_path, target_col, numerical_cols, categorical_cols, batch_size=512):
    """
    Load and prepare the data for modeling.
    
    Parameters:
    file_path (str): The path to the CSV file.
    target_col: Name of target column.
    numerical_cols (list): List of numeric column names.
    categorical_cols (list): List of categorical column names.
    columns_to_keep (list): List of columns to keep in the final DataFrame.
    """
    data = load_data(file_path)
    if data is None:
        return None, None, None
    
    data = keep_columns(data, numerical_cols + categorical_cols + [target_col])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data[target_col] = data[target_col].astype('category')
    class_names = data[target_col].cat.categories.tolist()

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    data[target_col] = data[target_col].cat.codes

    cat_cardinalities = [data[col].nunique() for col in categorical_cols]

    train_df, valid_df, test_df = split_data(data, target_col)



    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numerical_cols])
    X_valid_num = scaler.transform(valid_df[numerical_cols])
    X_test_num  = scaler.transform(test_df[numerical_cols])
    
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_valid_num = torch.tensor(X_valid_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    
    X_train_cat = torch.tensor(get_X_cat(train_df, categorical_cols).values, dtype=torch.long)
    X_valid_cat = torch.tensor(get_X_cat(valid_df, categorical_cols).values, dtype=torch.long)
    X_test_cat = torch.tensor(get_X_cat(test_df, categorical_cols).values, dtype=torch.long)

    y_train = torch.tensor(train_df[target_col].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df[target_col].values, dtype=torch.long)
    y_test = torch.tensor(test_df[target_col].values, dtype=torch.long)

    train_dataset = TabularDataset(X_train_num, X_train_cat, y_train)
    valid_dataset = TabularDataset(X_valid_num, X_valid_cat, y_valid)
    test_dataset = TabularDataset(X_test_num, X_test_cat, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names


def _get_linear_layers(seq):
    idx_and_layers = [(i, m) for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    # scorre 'seq' (tipicamente nn.Sequential), raccoglie (indice, modulo) SOLO per i layer Linear
    layers = [m for _, m in idx_and_layers]
    # estrae solo i moduli Linear (senza gli indici) in una lista parallela
    return idx_and_layers, layers
    # ritorna sia (indice, layer) sia solo i layer, comodo per mappare vecchio->nuovo

def _topk_indices_by_row_l1(weight, k):
    # weight: [out_features, in_features]  -> ogni riga è un neurone in uscita
    scores = weight.abs().sum(dim=1)      # calcola l'L1 per riga (importanza del neurone)
    k = max(1, min(k, weight.shape[0]))   # numero neuroni da tenere
    topk = torch.topk(scores, k, largest=True)
    # prende gli indici delle k righe con punteggio più alto (neuroni da TENERE)
    keep_idx = torch.sort(topk.indices).values
    # ordina gli indici crescenti per avere slicing stabile (mantiene l’ordine)
    return keep_idx                        # restituisce gli indici dei neuroni da tenere

def structured_prune_hidden(model: NeuralNetwork, keep_ratio=0.7, keep_ratios_per_layer=None, device="cpu"):
    """
    Pruna in modo strutturato i layer nascosti (nn.Linear) della tua rete.
    - keep_ratio: frazione di neuroni da TENERE per ogni hidden layer (es. 0.7).
    - keep_ratios_per_layer: lista opzionale di ratio, uno per hidden layer (ha priorità su keep_ratio).

    Ritorna: nuovo modello con hidden ridotti + copia dei pesi prunati.
    """
    model = deepcopy(model).to(device)     # clona il modello e spostalo su device (evita di toccare l’originale)
    model.eval()                           # modalità valutazione (disattiva dropout/batchnorm)
    with torch.no_grad():                  # no grad: non servono gradienti durante il pruning/copia pesi
        # 1) individua i Linear nella sequenza
        lin_idx_and_layers, lin_layers = _get_linear_layers(model.network)
        # estrae tutti i nn.Linear dalla tua 'nn.Sequential' (incluso output)
        if len(lin_layers) < 2:
            raise ValueError("Mi aspetto almeno un hidden e un output linear.")
        # se c’è meno di 2 Linear, non avrebbe senso (serve almeno 1 hidden + 1 output)
        hidden_layers = lin_layers[:-1]    # escludi l'ultimo (output) dai layer da potare
        output_layer = lin_layers[-1]      # riferimento (non usato direttamente) al layer di output

        # 2) calcola nuove dimensioni per gli hidden
        old_hidden_sizes = [L.out_features for L in hidden_layers]
        # raccoglie per ogni hidden il numero di neuroni attuale
        if keep_ratios_per_layer is None:
            keep_ratios_per_layer = [keep_ratio] * len(old_hidden_sizes)
        # se non hai passato ratio per layer, usa lo stesso keep_ratio per tutti
        assert len(keep_ratios_per_layer) == len(old_hidden_sizes)
        # controllo: dev’esserci un ratio per ogni hidden

        new_hidden_sizes = [max(1, int(round(h * r))) for h, r in zip(old_hidden_sizes, keep_ratios_per_layer)]
        # calcola quanti neuroni TENERE per ciascun hidden: round(h * r), minimo 1

        # 3) costruisci un nuovo modello con hidden ridotti (embedding invariati)
        new_model = NeuralNetwork(
            hidden_layers_sizes=new_hidden_sizes,                            # nuove dimensioni potate
            cat_cardinalities=model.config['cat_cardinalities'],             # parametri originali
            embedding_dims=model.config['embedding_dims'],
            num_numerical_features=model.config['num_numerical_features'],
            num_target_classes=model.config['num_target_classes']
        ).to(device)
        # istanzia la stessa architettura ma con hidden più piccoli
        new_model.eval()                 # modalità eval anche per il nuovo modello

        # copia pesi embedding 1:1
        for old_emb, new_emb in zip(model.embeddings, new_model.embeddings):
            new_emb.weight.data.copy_(old_emb.weight.data)
        # gli embedding non si potano, copia diretta dei pesi

        # 4) ricostruisci e copia i pesi 
        old_lin_layers = lin_layers
        # lista dei linear del vecchio modello (hidden + output)
        new_lin_layers = _get_linear_layers(new_model.network)[1]
        # lista dei linear del nuovo modello (con hidden ridotti)

        prev_keep_idx = None  # mapping delle uscite del layer precedente
        # servirà a selezionare le COLONNE (ingressi) del prossimo layer in base ai neuroni tenuti prima

        for i, (old_lin, new_lin) in enumerate(zip(old_lin_layers, new_lin_layers)):
            # scorre in parallelo i linear vecchi e nuovi allo stesso livello
            old_W = old_lin.weight.data
            #matrice pesi del vecchio layer: shape [out_old, in_old]
            old_b = old_lin.bias.data if old_lin.bias is not None else None
            # bias del vecchio layer (può essere None)

            # adegua le COLONNE in base ai neuroni tenuti dallo strato precedente
            if prev_keep_idx is not None:
                W_in = old_W[:, prev_keep_idx]
                # se prima abbiamo tenuto solo alcune uscite, qui selezioniamo le colonne corrispondenti
                # (quelle uscite diventano ingressi di questo layer)
            else:
                W_in = old_W  # primo Linear del MLP (dopo concat input)
                # nel primissimo Linear del MLP non c’è pruning “a monte”: usiamo tutte le colonne

            if i < len(new_lin_layers) - 1:
                # è un hidden: scegli quali neuroni (righe) tenere
                keep_count = new_lin.out_features
                # quante righe deve avere questo layer nel nuovo modello (deciso prima)
                keep_idx = _topk_indices_by_row_l1(W_in, keep_count)
                # scegli gli indici delle righe (neuroni) con L1 più grande (importanti)
                W_out = W_in[keep_idx, :]
                # tieni solo quelle righe (pruning strutturato delle uscite)
                b_out = old_b[keep_idx] if old_b is not None else None
                # idem per il bias, se presente
                prev_keep_idx = keep_idx  # servirà per il prossimo layer (colonne)
                # memorizza quali uscite sono rimaste, così il prossimo layer potrà ridurre le sue colonne
            else:
                # layer di output: non pruniamo le classi
                W_out = W_in
                # manteniamo tutte le righe (tutte le classi in uscita)
                b_out = old_b
                prev_keep_idx = None  # non serve più
                # dopo l’output non c’è un layer successivo che debba selezionare colonne

            # crea peso/bias nel nuovo layer (consistenza delle forme)
            assert new_lin.in_features == W_out.shape[1], \
                f"in_features attesi {new_lin.in_features}, trovati {W_out.shape[1]}"
            # controlla che il numero di colonne coincidano con gli in_features del nuovo layer
            assert new_lin.out_features == W_out.shape[0], \
                f"out_features attesi {new_lin.out_features}, trovati {W_out.shape[0]}"
            # controlla che il numero di righe coincidano con gli out_features del nuovo layer

            new_lin.weight.data.copy_(W_out.contiguous())
            # copia la matrice dei pesi potata nel nuovo layer (contiguous per sicurezza di layout)
            if new_lin.bias is not None and b_out is not None:
                new_lin.bias.data.copy_(b_out.contiguous())
            # copia il bias corrispondente, se presente

        # copia anche il resto dei moduli non-lineari (ReLU/Dropout) è già a posto nella nuova architettura
        return new_model


def benchmark_cpu(model, dataloader, num_threads=1, warmup=30, iters=200, device="cpu"):
    old_nt = torch.get_num_threads()
    dev = device if isinstance(device, torch.device) else torch.device(device)
    is_cuda = (dev.type == "cuda")

    if not is_cuda:
        torch.set_num_threads(num_threads)

    model.to(dev).eval()

    # prendi un batch rappresentativo
    it = iter(dataloader)
    x_num, x_cat, _ = next(it)
    x_num = x_num.to(dev)
    x_cat = x_cat.to(dev)

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x_num, x_cat)
        if is_cuda:
            torch.cuda.synchronize()

    # misure
    times = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x_num, x_cat)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    if not is_cuda:
        torch.set_num_threads(old_nt)

    times = np.array(times)
    bs = x_num.shape[0]
    total = times.sum()
    return {
        "batch_size": bs,
        "num_threads": (num_threads if not is_cuda else None),
        "median_ms": float(np.median(times) * 1e3),
        "p95_ms": float(np.percentile(times, 95) * 1e3),
        "throughput_sps": float((iters * bs) / total),
        "device": dev.type,
    }



def hidden_sizes(model: NeuralNetwork):
    lin_layers = _get_linear_layers(model.network)[1]
    assert len(lin_layers) >= 2, "Attesi almeno un hidden + output"
    return [L.out_features for L in lin_layers[:-1]]


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def model_size_megabytes(model: nn.Module, dtype_bytes=None):
    # stima: somma dei parametri * bytes per elemento (default: da primo parametro)
    ps = [p for p in model.parameters()]
    if dtype_bytes is None:
        dtype_bytes = torch.finfo(ps[0].dtype).bits // 8 if ps[0].is_floating_point() else torch.iinfo(ps[0].dtype).bits // 8
    total_elems = sum(p.numel() for p in ps)
    return total_elems * dtype_bytes / (1024**2)

def total_hidden_neurons(model: NeuralNetwork):
    return sum(hidden_sizes(model))

def pct_neurons_kept(orig_hidden, pruned_hidden):
    return 100.0 * (sum(pruned_hidden) / sum(orig_hidden))

# ---- Valutazione F1 su un dataloader ----
def eval_f1_macro(model, dataloader, device="cpu"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for x_num, x_cat, y in dataloader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return f1_score(all_labels, all_preds, average="macro")

# ---- Sweep pruning 100% -> 10% a step di 10% ----
def pruning_sweep(
    base_model: NeuralNetwork,
    train_loader, valid_loader, test_loader,
    class_weights=None,
    device="cpu",
    finetune_epochs=3,    # piccolo fine-tuning post-pruning (0 per saltare)
    finetune_lr=5e-4,
    num_threads_bench=1,
    warmup=30, iters=200
):
    results = []
    orig_hidden = hidden_sizes(base_model)
    orig_params = count_params(base_model)
    orig_size_mb = model_size_megabytes(base_model)
    orig_neurons_sum = sum(orig_hidden)

    # baseline (100% = nessun pruning + (opz) fine-tuning breve)
    base = copy.deepcopy(base_model)
    if finetune_epochs > 0:
        base.fit(train_loader, valid_loader, device=device, epochs=finetune_epochs, lr=finetune_lr, weights=class_weights)
    f1_base = eval_f1_macro(base, test_loader, device=device)
    t_base = benchmark_cpu(base, test_loader, num_threads=num_threads_bench, warmup=warmup, iters=iters, device=device)
    res0 = {
        "keep_ratio": 1.0,
        "neurons_kept_pct": 100.0,
        "hidden_shape": orig_hidden,
        "params_M": orig_params/1e6,
        "model_MB": orig_size_mb,
        "f1_macro": f1_base,
        "median_ms": t_base["median_ms"],
        "p95_ms": t_base["p95_ms"],
        "throughput_sps": t_base["throughput_sps"]
    }
    results.append(res0)

    for r in range(90, 0, -10):  # 90,80,...,10
        keep_ratio = r / 100.0
        pruned = structured_prune_hidden(base_model, keep_ratio=keep_ratio, device=device)
        # piccolo fine-tuning per recuperare performance
        if finetune_epochs > 0:
            pruned.fit(train_loader, valid_loader, device=device, epochs=finetune_epochs, lr=finetune_lr, weights=class_weights)

        f1 = eval_f1_macro(pruned, test_loader, device=device)
        t = benchmark_cpu(pruned, test_loader, num_threads=num_threads_bench, warmup=warmup, iters=iters, device=device)

        pruned_hidden = hidden_sizes(pruned)
        params = count_params(pruned)
        size_mb = model_size_megabytes(pruned)
        kept_pct = pct_neurons_kept(orig_hidden, pruned_hidden)

        results.append({
            "keep_ratio": keep_ratio,
            "neurons_kept_pct": kept_pct,
            "hidden_shape": pruned_hidden,
            "params_M": params/1e6,
            "model_MB": size_mb,
            "f1_macro": f1,
            "median_ms": t["median_ms"],
            "p95_ms": t["p95_ms"],
            "throughput_sps": t["throughput_sps"]
        })

    df = pd.DataFrame(results).sort_values("neurons_kept_pct", ascending=False).reset_index(drop=True)
    return df

