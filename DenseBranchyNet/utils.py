import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from neural_network import TabularDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import numpy as np
import time

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



def load_and_prepare_data(file_path, target_col, numerical_cols, categorical_cols, rows_to_remove, batch_size=512, test_batch_size=128):
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
    test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False  # in test è meglio tenerlo disattivato per risultati ripetibili
)

    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names


def load_and_prepare_nb15(file_path, target_col, numerical_cols, categorical_cols, batch_size=512, test_batch_size=128):
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
    test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False  # in test è meglio tenerlo disattivato per risultati ripetibili
)


    cw = get_weights(y_train)


    
    return train_dataloader, valid_dataloader, test_dataloader, cat_cardinalities, cw, class_names



@torch.inference_mode()
def benchmark_cpu(
    model,
    dataloader,
    branchy: bool,
    use_margin: bool = True,
    warmup_iters: int = 20,
    max_samples: int | None = None,
    num_threads: int = 1,
):
    """
    Misura la latenza media per istanza (batch=1) su CPU cronometrando SOLO il passaggio nella rete.
    - branchy=True  -> usa predict_using_margin / predict_using_entropy
    - branchy=False -> usa il forward standard
    """
    dev = torch.device("cpu")
    model = model.to(dev).eval()

    # imposta numero di threads
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    # verifica batch=1
    try:
        x_num0, x_cat0, _ = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader vuoto.")
    assert x_num0.size(0) == 1, "Questo benchmark richiede DataLoader con batch_size=1."

    # funzione di inferenza
    def infer_fn(xn, xc):
        if branchy:
            if use_margin and hasattr(model, "predict_using_margin"):
                return model.predict_using_margin(xn, xc)
            elif not use_margin and hasattr(model, "predict_using_entropy"):
                return model.predict_using_entropy(xn, xc)
            return model(xn, xc, branchy=True, use_margin=use_margin, return_logits=False)
        else:
            return model(xn, xc)

    # warmup
    it = iter(dataloader)
    for _ in range(warmup_iters):
        try:
            xn, xc, _ = next(it)
        except StopIteration:
            it = iter(dataloader)
            xn, xc, _ = next(it)
        _ = infer_fn(xn, xc)

    # misura
    times_ms = []
    n = 0
    for xn, xc, _ in dataloader:
        if max_samples is not None and n >= max_samples:
            break
        t0 = time.perf_counter()
        _ = infer_fn(xn, xc)
        dt_ms = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt_ms)
        n += 1

    # ripristina threads originali
    torch.set_num_threads(old_threads)

    times = np.array(times_ms, dtype=np.float64)
    return {
        "samples": int(n),
        "branchy": bool(branchy),
        "use_margin": bool(use_margin),
        "num_threads": num_threads,
        "mean_ms": float(times.mean()) if n else float("nan"),
        "median_ms": float(np.median(times)) if n else float("nan"),
        "p95_ms": float(np.percentile(times, 95)) if n else float("nan"),
        "min_ms": float(times.min()) if n else float("nan"),
        "max_ms": float(times.max()) if n else float("nan"),
    }



import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import f1_score

def plot_latency_vs_f1_branchynet(
    model,
    branchynet,
    valid_dataloader,
    test_dataloader,
    y_true,
    device,
    baseline_start=0.88,
    baseline_end=0.99,
    baseline_step=0.01,
    f1_drop=0.005,
    relative=True,
    per_class=False,
    n_grid=101,
    f1_average="weighted",
    num_threads=None,
    annotate_points=True,
    title="Tempo vs F1 • BranchyNet: margini vs softmax+entropia",
):
    """
    Disegna un grafico unico: tempo medio (ms) vs F1(test) per BranchyNet (margini/entropia),
    con sweep di baseline_f1 in [baseline_start, baseline_end] a step baseline_step.
    Aggiunge anche il modello standard (linea verticale F1, punto e linea orizzontale tempo).

    Ritorna un dict con i dati tracciati.
    """
    if num_threads is None:
        num_threads = max(1, os.cpu_count() // 2)

    # -- helper robusto per convertire y_true e pred in numpy
    def to_numpy(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    y_true_np = to_numpy(y_true)

    # ---- Benchmark + F1 del modello standard 
    dur_model = benchmark_cpu(
        model, test_dataloader,
        branchy=False, use_margin=False, num_threads=num_threads
    )
    standard_time = dur_model["mean_ms"]

    model_preds = model.predict(test_dataloader, device)
    model_preds_np = to_numpy(model_preds)
    standard_f1 = f1_score(y_true_np, model_preds_np, average=f1_average)

    # ---- Sweep baseline_f1
    baseline_grid = np.round(np.arange(baseline_start, baseline_end + 1e-12, baseline_step), 2)

    #per evitare side effects tra i punti
    base_bn = deepcopy(branchynet).eval()

    pts_margin, pts_entropy = [], []   # ciascun punto: (f1_test, mean_ms, baseline_f1)

    for b in baseline_grid:
        # ---------- BranchyNet con margini -------
        print(f'Starting margin for {b}')
        bn = deepcopy(base_bn).eval()
        _ = bn.calibrate_tau(
            valid_dataloader, device,
            mode="margin", baseline_f1=float(b),
            f1_drop=f1_drop, relative=relative,
            per_class=per_class, n_grid=n_grid, return_curve=False
        )
        dur_bn_m = benchmark_cpu(
            bn, test_dataloader,
            branchy=True, use_margin=True, num_threads=num_threads
        )
        preds_bn_m = bn.predict_all(test_dataloader, use_margin=True)
        f1_bn_m = f1_score(y_true_np, to_numpy(preds_bn_m), average=f1_average)
        pts_margin.append((f1_bn_m, dur_bn_m["mean_ms"], float(b)))

        # ------- BranchyNet con entropia ----------
        print(f'Starting softmax + entropy for {b}')
        bn = deepcopy(base_bn).eval()
        _ = bn.calibrate_tau(
            valid_dataloader, device,
            mode="entropy", baseline_f1=float(b),
            f1_drop=f1_drop, relative=relative,
            per_class=per_class, n_grid=n_grid, return_curve=False
        )
        dur_bn_e = benchmark_cpu(
            bn, test_dataloader,
            branchy=True, use_margin=False, num_threads=num_threads
        )
        preds_bn_e = bn.predict_all(test_dataloader, use_margin=False)
        f1_bn_e = f1_score(y_true_np, to_numpy(preds_bn_e), average=f1_average)
        pts_entropy.append((f1_bn_e, dur_bn_e["mean_ms"], float(b)))
        print('--------------------------------')


    plt.figure(figsize=(9, 6))

    # Ordina per F1 per linee più leggibili
    pts_margin.sort(key=lambda t: t[0])
    pts_entropy.sort(key=lambda t: t[0])

    xm = [p[0] for p in pts_margin]
    ym = [p[1] for p in pts_margin]
    bm = [p[2] for p in pts_margin]

    xe = [p[0] for p in pts_entropy]
    ye = [p[1] for p in pts_entropy]
    be = [p[2] for p in pts_entropy]

    plt.plot(xm, ym, marker="o", linewidth=1.8, label="Branchy (margini)")
    plt.plot(xe, ye, marker="s", linewidth=1.8, label="Branchy (softmax + entropia)")

    # Annotazioni dei punti con baseline_f1
    if annotate_points:
        for x, y, b in zip(xm, ym, bm):
            plt.annotate(f"b={b:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for x, y, b in zip(xe, ye, be):
            plt.annotate(f"b={b:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    # Modello standard: linea orizzontale tempo, verticale F1, punto
    plt.axhline(standard_time, linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"Tempo standard (~{standard_time:.1f} ms)")
    plt.axvline(standard_f1, linestyle="--", linewidth=1.2, alpha=0.6,
                label=f"F1 standard (~{standard_f1:.3f})")
    plt.scatter([standard_f1], [standard_time], marker="X", s=80, zorder=5, label="Punto modello standard")
    plt.annotate(f"({standard_f1:.3f}, {standard_time:.1f} ms)",
                 (standard_f1, standard_time), textcoords="offset points", xytext=(6, -12), fontsize=8)

    plt.xlabel(f"F1 ({f1_average}) su test")
    plt.ylabel("Tempo medio di inferenza (ms) — batch=1, CPU")
    plt.title(f"{title}\n(sweep baseline_f1: {baseline_start:.2f} → {baseline_end:.2f}, step {baseline_step:.2f})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "standard": {"f1": standard_f1, "mean_ms": standard_time},
        "margin":   {"points": pts_margin},   # list of (f1_test, mean_ms, baseline_f1)
        "entropy":  {"points": pts_entropy},  # list of (f1_test, mean_ms, baseline_f1)
        "config": {
            "f1_drop": f1_drop, "relative": relative, "per_class": per_class,
            "baseline_range": (baseline_start, baseline_end, baseline_step),
            "n_grid": n_grid, "num_threads": num_threads, "f1_average": f1_average
        }
    }

