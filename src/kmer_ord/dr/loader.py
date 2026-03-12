# src/kmer_ord/dr/loader.py
from pathlib import Path
import pandas as pd

def load_matrix(matrix_path: Path) -> pd.DataFrame:
    """
    Load k-mer matrix from TSV, CSV, or NPY file.
    First column is sample IDs; rest are numeric features.
    Returns pd.DataFrame with sample IDs as index.
    """
    import numpy as np
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"K-mer matrix not found: {matrix_path}")

    suffix = matrix_path.suffix.lower()

    if suffix in [".tsv", ".csv"]:
        df = pd.read_csv(matrix_path, sep="\t" if suffix == ".tsv" else ",", index_col=0)
        #df = pd.read_csv(matrix_path, sep="\t" if suffix == ".tsv" else ",", index_col=None)
    elif suffix == ".npy":
        arr = np.load(matrix_path)
        df = pd.DataFrame(arr)
    else:
        raise ValueError(f"Unsupported matrix format: {suffix}")

    if df.shape[0] < 2:
        raise ValueError("Matrix must contain at least 2 samples for DR.")

    # ensure all remaining columns are numeric
    df = df.apply(pd.to_numeric, errors="raise")

    return df