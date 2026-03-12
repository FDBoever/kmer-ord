# kmer_ord/cluster/leiden.py

def run_leiden(A, resolution=1.0, n_iterations=-1, seed=42):
    """
    Run Leiden clustering on sparse adjacency matrix.

    Returns:
        labels : np.ndarray
        modularity : float
    """
    import numpy as np
    import leidenalg
    import igraph as ig
    
    if A.nnz == 0:
        print("[Warning] Graph has no edges. Cannot run Leiden.")
        return np.array([]), np.nan

    # Convert to igraph
    sources, targets = A.nonzero()
    g = ig.Graph(directed=False)
    g.add_vertices(A.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = A.data

    try:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed)
        labels = np.array(partition.membership)
        modularity = g.modularity(labels, weights=g.es["weight"])
        return labels, modularity
    except Exception as e:
        print(f"[Error] Leiden clustering failed: {e}")
        return np.array([]), np.nan


def leiden_resolution_sweep(A, resolutions=None, n_iterations=-1, seed=42):
    """
    Sweep over multiple resolutions for Leiden clustering.

    Returns:
        results : dict
            resolution -> (labels, modularity)
    """
    import numpy as np
    from tqdm import tqdm

    if resolutions is None:
        resolutions = np.linspace(0.01, 1.0, 10)

    results = {}
    for r in tqdm(resolutions, desc="Resolution sweep"):
        labels, modularity = run_leiden(A, resolution=r, n_iterations=n_iterations, seed=seed)
        results[r] = (labels, modularity)

    return results