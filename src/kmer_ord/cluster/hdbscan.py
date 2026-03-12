# kmer_ord/cluster/hdbscan.py

def run_hdbscan(X, min_cluster_size=15, min_samples=None, metric="euclidean"):
    """
    Run HDBSCAN clustering on data matrix.
    
    Returns:
        labels : np.ndarray
    """
    import hdbscan
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    return labels