# kmer_ord/cluster/dbscan.py
import numpy as np

def run_dbscan(X, eps=0.5, min_samples=5, metric="euclidean"):
    """
    Run DBSCAN clustering on data matrix.

    Returns:
        labels : np.ndarray
    """
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = db.fit_predict(X)
    return labels