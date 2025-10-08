import pandas as pd
from sklearn.model_selection import train_test_split
import torch, platform
from neural_network import TabularDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
import numpy as np
import torch.nn as nn
import torch.ao.quantization as tq
import time
from copy import deepcopy


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


def quantize_dynamic_model(model_fp32: nn.Module) -> nn.Module:
    print('Platform: ' + platform.machine().lower())
    if platform.machine().lower() in ("x86_64","amd64"):
        torch.backends.quantized.engine = "fbgemm"
    else:
        torch.backends.quantized.engine = "qnnpack"
    model_fp32.eval()
    qdyn = tq.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8  # INT8 pesi
    )
    return qdyn


def quantize_hidden_only(model: nn.Module):
    m = deepcopy(model).eval()
    # trova gli indici dei Linear in m.network (Linear, ReLU, Dropout, ..., Linear[OUT])
    lin_idx = [i for i, mod in enumerate(m.network) if isinstance(mod, nn.Linear)]
    if len(lin_idx) <= 2:
        return m.eval()  # niente hidden da quantizzare

    # quantizza tutto (Linear)
    q = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval()
    # ripristina primo e ultimo Linear in float
    q.network[lin_idx[0]]  = m.network[lin_idx[0]]   # first FP32
    q.network[lin_idx[-1]] = m.network[lin_idx[-1]]  # last  FP32
    return q.eval()



def benchmark_cpu(model, dataloader, num_threads=1, warmup=30, iters=200, device="cpu"):
    old_nt = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    model.to(device).eval()

    # prendi un batch rappresentativo dal tuo dataloader
    it = iter(dataloader)
    x_num, x_cat, _ = next(it)
    x_num = x_num.to(device)
    x_cat = x_cat.to(device)

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x_num, x_cat)

    # misure
    times = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x_num, x_cat)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            times.append(time.perf_counter() - t0)

    torch.set_num_threads(old_nt)

    times = np.array(times)
    bs = x_num.shape[0]
    total = times.sum()
    return {
        "batch_size": bs,
        "num_threads": num_threads,
        "median_ms": float(np.median(times) * 1e3),
        "p95_ms": float(np.percentile(times, 95) * 1e3),
        "throughput_sps": float((iters * bs) / total)
    }

def quantize_all_but_output(model_fp32: nn.Module) -> nn.Module:
    """
    Quantizza dinamicamente TUTTI i layer Linear tranne l'ultimo (output),
    che rimane in FP32. Mantiene la struttura del modello in eval().
    Richiede che il modello abbia un attributo `network` di tipo nn.Sequential.
    """
    print('Platform:', platform.machine().lower())
    if platform.machine().lower() in ("x86_64", "amd64"):
        torch.backends.quantized.engine = "fbgemm"
    else:
        torch.backends.quantized.engine = "qnnpack"

    # copia e metti in eval
    m = deepcopy(model_fp32).eval()

    # individua i Linear dentro m.network (come nel tuo codice)
    if not hasattr(m, "network") or not isinstance(m.network, nn.Sequential):
        raise ValueError("Atteso m.network come nn.Sequential per applicare la sostituzione dell'output.")

    lin_idx = [i for i, mod in enumerate(m.network) if isinstance(mod, nn.Linear)]
    if len(lin_idx) < 1:
        # nessun Linear: niente da fare
        return m.eval()

    # quantizza dinamicamente tutti i Linear
    q = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval()

    # ripristina SOLO l'ultimo Linear (output) in FP32
    last_linear_idx = lin_idx[-1]
    q.network[last_linear_idx] = m.network[last_linear_idx]  # output FP32

    # (opzionale) logging minimale dei layer effettivamente quantizzati
    # print("Linear quantizzati (INT8):", [i for i in lin_idx[:-1]])
    # print("Output rimasto FP32:", last_linear_idx)

    return q.eval()


import platform
import torch
import torch.nn as nn
from copy import deepcopy

# ---------- IMPORT SHIM (gestisce differenze tra versioni PyTorch) ----------
try:
    # Nuove versioni
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
except ImportError:
    # Alcune build hanno ancora l'alias diretto
    from torch.ao.quantization.fx import prepare_fx, convert_fx  # type: ignore

try:
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
except ImportError:
    # Fallback: su versioni vecchie non esiste QConfigMapping → usiamo Mapping semplice
    QConfigMapping = None  # lo gestiamo sotto

try:
    from torch.ao.quantization import get_default_qconfig
except ImportError:
    from torch.ao.quantization.qconfig import get_default_qconfig  # type: ignore

try:
    from torch.ao.quantization import float16_dynamic_qconfig
except ImportError:
    from torch.ao.quantization.qconfig import float16_dynamic_qconfig  # type: ignore

import torch.ao.quantization as tq
from copy import deepcopy

def quantize_all_but_output(model_fp32: nn.Module) -> nn.Module:
    """
    Quantizza dinamicamente TUTTI i layer Linear tranne l'ultimo (output),
    che rimane in FP32. Mantiene la struttura del modello in eval().
    Richiede che il modello abbia un attributo `network` di tipo nn.Sequential.
    """
    print('Platform:', platform.machine().lower())
    if platform.machine().lower() in ("x86_64", "amd64"):
        torch.backends.quantized.engine = "fbgemm"
    else:
        torch.backends.quantized.engine = "qnnpack"

    # copia e metti in eval
    m = deepcopy(model_fp32).eval()

    # individua i Linear dentro m.network (come nel tuo codice)
    if not hasattr(m, "network") or not isinstance(m.network, nn.Sequential):
        raise ValueError("Atteso m.network come nn.Sequential per applicare la sostituzione dell'output.")

    lin_idx = [i for i, mod in enumerate(m.network) if isinstance(mod, nn.Linear)]
    if len(lin_idx) < 1:
        # nessun Linear: niente da fare
        return m.eval()

    # quantizza dinamicamente tutti i Linear
    q = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8).eval()

    # ripristina SOLO l'ultimo Linear (output) in FP32
    last_linear_idx = lin_idx[-1]
    q.network[last_linear_idx] = m.network[last_linear_idx]  # output FP32

    # (opzionale) logging minimale dei layer effettivamente quantizzati
    # print("Linear quantizzati (INT8):", [i for i in lin_idx[:-1]])
    # print("Output rimasto FP32:", last_linear_idx)

    return q.eval()

def _linear_positions(seq: nn.Sequential):
    """Indici (0..N-1) dei SOLI nn.Linear dentro il tuo Sequential."""
    return [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]

def quantize_fp16_dynamic(model: nn.Module,
                          exclude_idx=(0, -1)):
    """
    Quantizzazione *dinamica* FP16 (weights-only) per MLP tabellare:
      - Converte i Linear in FP16 via torch.ao.quantization.quantize_dynamic(dtype=torch.float16)
      - Mantiene in FP32 i Linear indicati da fp32_linear_idx (tra i SOLI Linear).
    Default: tieni FP32 primo(0), hidden #2 (2) e ultimo(-1).

    Ritorna: modello (CPU, eval)
    """
    m = deepcopy(model).cpu().eval()

    # 1) applica dynamic quantization FP16 (pesi FP16, attivazioni FP32)
    q = tq.quantize_dynamic(m, {nn.Linear}, dtype=torch.float16).eval()

    # 2) ripristina in FP32 i Linear esclusi (per indice tra i SOLI Linear)
    lin_pos = _linear_positions(m.network)           # posizioni nel Sequential
    L = len(lin_pos)
    keep = set(i if i >= 0 else (L + i) for i in exclude_idx)

    # copia i layer float originali nei punti esclusi
    for i_lin, seq_idx in enumerate(lin_pos):
        if i_lin in keep:
            q.network[seq_idx] = m.network[seq_idx]  # rimette FP32

    return q.eval()
