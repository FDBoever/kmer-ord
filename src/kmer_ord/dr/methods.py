# src/kmer_ord/dr/methods.py
import numpy as np
import pandas as pd
from pathlib import Path
from kmer_ord.utils.logging_utils import section, info, warn, console
#from umap import UMAP

# methods supporting parameter screening
SCREENABLE_METHODS = {"umap", "tsne", "trimap", "pacmap", "localmap"}

#current hyperparameters depending on dataset size (scale)
DR_HYPERPARAMS = {
    'umap': {
        'default': {'n_neighbors': 15, 'min_dist': 0.1},
        'small':   {'n_neighbors': 50, 'min_dist': 0.05},
        'medium':  {'n_neighbors': 150, 'min_dist': 0.1},
        'large':   {'n_neighbors': 200, 'min_dist': 0.1}
    },
    'tsne': {
        'default': {'init': 'pca', 'random_state': 42},
        'small':   {'perplexity': 30, 'init': 'pca', 'random_state': 42, 'learning_rate': 10},
        'medium':  {'perplexity': 100, 'init': 'pca', 'random_state': 42, 'learning_rate': 10},
        'large':   {'perplexity': 200, 'init': 'pca', 'random_state': 42, 'learning_rate': 10}
    },
    'trimap': {
        'default': {'n_inliers': 10, 'weight_temp': 0.5},
        'small':   {'n_inliers': 50, 'weight_temp': 0.3},
        'medium':  {'n_inliers': 100, 'weight_temp': 0.4},
        'large':   {'n_inliers': 150, 'weight_temp': 0.5}
    },
    'pacmap': {
        'default': {'MN_ratio': 0.5, 'FP_ratio': 2},
        'small':   {'n_neighbors': 15, 'MN_ratio': 0.5, 'FP_ratio': 2},
        'medium':  {'n_neighbors': 100, 'MN_ratio': 0.5, 'FP_ratio': 3},
        'large':   {'n_neighbors': 200, 'MN_ratio': 0.5, 'FP_ratio': 5}
    },
    'localmap': {
        'default': {'MN_ratio': 0.5, 'FP_ratio': 0.5},
        'small':   {'n_neighbors': 15, 'MN_ratio': 0.3, 'FP_ratio': 0.5},
        'medium':  {'n_neighbors': 100, 'MN_ratio': 0.5, 'FP_ratio': 1.0},
        'large':   {'n_neighbors': 200, 'MN_ratio': 0.7, 'FP_ratio': 1.0}
    },
    'pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'sparse_pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'kernel_pca': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
    'lle': {'default': {}, 'small': {}, 'medium': {}, 'large': {}},
}

ALL_METHODS = list(DR_HYPERPARAMS.keys())


def _run_single_method(
    X: np.ndarray,
    method: str,
    dims: int,
    seed: int,
    scale: str = "default",
    n_jobs: int = 1,
):
    import pandas as pd
    import scipy.sparse as sparse
    from sklearn.decomposition import PCA, KernelPCA, SparsePCA
    from sklearn.manifold import TSNE, LocallyLinearEmbedding

    try:
        import umap
    except ImportError:
        umap = None

    #from trimap import TRIMAP
    #from pacmap import PaCMAP
    #from pacmap.pacmap import LocalMAP
    method = method.lower()
    params = DR_HYPERPARAMS.get(method, {}).get(scale, {})
    graph = None

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=dims, **params)
        embedding = model.fit_transform(X)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=dims, random_state=seed, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    elif method == "umap":
        import umap
        warn("UMAP random seed is disabled to allow parallel execution.")
        model = umap.UMAP(n_components=dims, random_state=None, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "trimap":
        from trimap import TRIMAP
        model = TRIMAP(n_dims=dims, **params)
        embedding = model.fit_transform(X)

    elif method == "pacmap":
        from pacmap import PaCMAP
        model = PaCMAP(n_components=dims, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "localmap":
        from pacmap.pacmap import LocalMAP
        model = LocalMAP(n_components=dims, **params)
        embedding = model.fit_transform(X)
        graph = getattr(model, "graph_", None)

    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = params.get("n_neighbors", 10)
        model = LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=dims,
            n_jobs=n_jobs,
        )
        embedding = model.fit_transform(X)

    elif method == "sparse_pca":
        from sklearn.decomposition import SparsePCA
        model = SparsePCA(n_components=dims, random_state=seed, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    elif method == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=dims, n_jobs=n_jobs, **params)
        embedding = model.fit_transform(X)

    else:
        raise ValueError(f"Unsupported DR method: {method}")

    # ensure graph is sparse if present
    if graph is not None and not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)

    return embedding, graph


def run_dr_methods(
    X: np.ndarray | pd.DataFrame,
    methods: list[str],
    dims: int,
    seed: int,
    scale: str,
    screen_params: bool,
    output_dir: Path,
    normalisation: str,
    input_name: str,
    sequence_ids: list | pd.Index | None = None,
    n_jobs: int = 1,
) -> tuple[Path, list[Path]]:
    """
    Run selected DR methods for a single normalisation.

    Adds sequence_id column to all embeddings for downstream merging.
    Saves graph objects if produced by a method.

    Returns
    -------
    merged_file : Path
        Path to merged embeddings file
    graph_paths : list[Path]
        List of graph files saved (may be empty)
    """
    import pandas as pd
    import numpy as np
    import scipy.sparse as sparse
    # -----------------------------
    # Sequence IDs
    # -----------------------------
    if sequence_ids is None:
        if isinstance(X, pd.DataFrame):
            sequence_ids = X.index
        else:
            sequence_ids = np.arange(X.shape[0])

    if "all" in methods:
        methods = ALL_METHODS

    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    graph_paths: list[Path] = []

    # -----------------------------
    # Per-method execution
    # -----------------------------
    for method in methods:

        method = method.lower()

        method_dir = output_dir / normalisation / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Parameter screening
        # -------------------------
        if screen_params and method in SCREENABLE_METHODS:
            screen_dir = method_dir / "parameter_screen"
            screen_dir.mkdir(parents=True, exist_ok=True)

            _run_parameter_screen(
                X=X,
                method=method,
                dims=dims,
                seed=seed,
                scale=scale,
                output_dir=screen_dir,
                normalisation=normalisation,
                input_name=input_name,
                sequence_ids=sequence_ids,
                n_jobs=n_jobs,
            )

        # -------------------------
        # Default embedding
        # -------------------------
        embedding, graph = _run_single_method(
            X=X,
            method=method,
            dims=dims,
            seed=seed,
            scale=scale,
            n_jobs=n_jobs,
        )

        # save embedding
        columns = [f"{method}_{i+1}" for i in range(dims)]
        df_embed = pd.DataFrame(embedding, columns=columns)
        df_embed.insert(0, "sequence_id", sequence_ids)

        out_file = (method_dir / f"{input_name}_{normalisation}_{method}_{dims}D.tsv")

        df_embed.to_csv(out_file, sep="\t", index=False)
        info(f"saved  {normalisation}/{method}  {out_file}")

        dfs.append(df_embed)

        # Save graph if available
        if graph is not None:
            graph_file = (method_dir / f"{input_name}_{normalisation}_{method}_graph.npz")

            sparse.save_npz(graph_file, graph)
            graph_paths.append(graph_file)

    # Merge embeddings across methods
    merged_df = pd.concat(dfs, axis=1)

    # ensure only one sequence_id column
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_file = (output_dir / normalisation /f"{input_name}_{normalisation}_{dims}D_merged_embeddings.tsv")

    merged_df.to_csv(merged_file, sep="\t", index=False)
    info(f"merged  {normalisation}  {merged_file}")

    return merged_file, graph_paths


def _run_parameter_screen(
    X: pd.DataFrame,
    method: str,
    dims: int,
    seed: int,
    scale: str,
    output_dir: Path,
    normalisation: str,
    input_name: str,
    sequence_ids: list | pd.Index,
    n_jobs: int = 1,
) -> list[Path]:
    """
    Perform parameter screening for a given DR method.
    Saves individual files for each parameter combination with sequence_id column.
    Returns list of saved paths.
    """

    import pandas as pd
    import numpy as np
    import scipy.sparse as sparse
    from sklearn.decomposition import PCA, KernelPCA, SparsePCA
    from sklearn.manifold import TSNE, LocallyLinearEmbedding

    try:
        import umap
    except ImportError:
        umap = None

    from trimap import TRIMAP
    from pacmap import PaCMAP
    from pacmap.pacmap import LocalMAP

    output_paths = []

    def save_embedding(embedding: np.ndarray, param_str: str):
        """Helper to save a DataFrame with sequence_id."""
        df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
        df.insert(0, "sequence_id", sequence_ids)
        out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
        df.to_csv(out_file, sep="\t", index=False)
        output_paths.append(out_file)

    if method == "umap":
        n_neighbors_values = [5, 10, 50, 100, 150]
        min_dist_values = [0, 0.1, 0.25, 0.5, 1.0]
        for n in n_neighbors_values:
            for m in min_dist_values:
                info(f"UMAP  n_neighbors={n}  min_dist={m}")
                model = umap.UMAP(n_components=dims, n_neighbors=n, min_dist=m, n_jobs=n_jobs)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_min{m}")

    elif method == "tsne":
        perplexity_values = [5, 10, 30, 50, 100]
        learning_rate_values = [10, 100, 200, 500]
        for p in perplexity_values:
            for lr in learning_rate_values:
                info(f"t-SNE  perplexity={p}  learning_rate={lr}")
                model = TSNE(n_components=dims, perplexity=p, learning_rate=lr,
                             max_iter=1000, random_state=seed, n_jobs=n_jobs)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"p{p}_lr{lr}")

    elif method == "trimap":
        n_inliers_values = [10, 25, 50, 100, 150]
        weight_temp_values = [0.1, 0.5, 1.0, 2.0, 2.5]
        for n in n_inliers_values:
            for w in weight_temp_values:
                info(f"TRIMAP  n_inliers={n}  weight_temp={w}")
                model = TRIMAP(n_dims=dims, n_inliers=n, weight_temp=w)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"inliers{n}_weighttemp{w}")

    elif method == "pacmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]
        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                info(f"PaCMAP  n_neighbors={n}  FP_ratio={fp}")
                model = PaCMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")

    elif method == "localmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]
        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                info(f"LocalMAP  n_neighbors={n}  FP_ratio={fp}")
                model = LocalMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")

    else:
        raise ValueError(f"Parameter screening not implemented for method: {method}")

    return output_paths