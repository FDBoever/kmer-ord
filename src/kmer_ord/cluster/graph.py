# kmer_ord/cluster/graph.py
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(X, k=15, metric="euclidean"):
    """
    Build a k-nearest neighbor graph (sparse adjacency matrix) from data.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (samples × features)
    k : int
        Number of neighbors
    metric : str
        Distance metric ('euclidean' or 'cosine')

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix
    """
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)  # safety check

    nn = NearestNeighbors(n_neighbors=k+1, metric=metric, n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # exclude self-loops
    rows = np.repeat(np.arange(n_samples), k)
    cols = indices[:, 1:].flatten()
    data = np.ones(len(rows), dtype=np.float32)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    # Symmetrize
    A = A.maximum(A.transpose())
    return A