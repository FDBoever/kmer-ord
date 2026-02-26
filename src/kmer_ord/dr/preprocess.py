# src/kmer_ord/dr/preprocess.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply normalization to k-mer matrix (DataFrame of numeric k-mer counts).
    Returns a DataFrame (rows = samples, columns = features), preserving sample IDs.
    """

    X = df.copy().astype(np.float32)  # ensure float32

    if method == "raw":
        return X

    elif method == "relative":
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X.div(row_sums.flatten(), axis=0)
        return X

    elif method == "log":
        X = np.log1p(X)
        return pd.DataFrame(X, index=df.index, columns=df.columns)

    elif method == "clr":
        # Add small pseudocount to avoid log(0)
        X += 1e-9
        geometric_mean = np.exp(np.mean(np.log(X), axis=1))
        X = np.log(X.div(geometric_mean, axis=0))
        return X

    elif method == "zscore":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, index=df.index, columns=df.columns)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def reduce_dimensions_with_pca(df: pd.DataFrame,
                               keep_pcs: int | None = None,
                               keep_variance: float | None = None) -> pd.DataFrame:
    """
    Apply PCA reduction to a DataFrame either by fixed number of PCs
    or by cumulative variance threshold. Returns DataFrame with sample IDs as index.
    """

    if keep_pcs is None and keep_variance is None:
        raise ValueError("Either keep_pcs or keep_variance must be specified.")

    if keep_pcs is not None:
        pca = PCA(n_components=keep_pcs)
    else:
        # fit PCA with all components to compute cumulative variance
        pca_full = PCA()
        X_full = pca_full.fit_transform(df.values)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        keep_pcs = np.searchsorted(cumulative_variance, keep_variance) + 1
        pca = PCA(n_components=keep_pcs)

    X_pca = pca.fit_transform(df.values)
    columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, index=df.index, columns=columns)