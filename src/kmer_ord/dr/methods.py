# src/kmer_ord/dr/methods.py
import numpy as np
import pandas as pd
from pathlib import Path
from kmer_ord.utils.logging_utils import section, info, warn, divider, console


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def _resolve_scale(scale: str, n_seq: int) -> str:
    """Map 'auto' to a named scale tier based on dataset size."""
    if scale != "auto":
        return scale
    if n_seq < 5_000:
        return "small"
    if n_seq < 50_000:
        return "medium"
    return "large"

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
        'default': {'init': 'pca'},
        'small':   {'perplexity': 30,  'init': 'pca', 'learning_rate': 10},
        'medium':  {'perplexity': 100, 'init': 'pca', 'learning_rate': 10},
        'large':   {'perplexity': 200, 'init': 'pca', 'learning_rate': 10}
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

ALL_METHODS = ["umap", "tsne", "trimap", "pacmap", "localmap", "pca"]


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
    import time

    # Sequence IDs
    if sequence_ids is None:
        if isinstance(X, pd.DataFrame):
            sequence_ids = X.index
        else:
            sequence_ids = np.arange(X.shape[0])

    if "all" in methods:
        methods = ALL_METHODS

    output_dir.mkdir(parents=True, exist_ok=True)

    n_seq, n_feat = X.shape
    resolved_scale = _resolve_scale(scale, n_seq)
    w = 16
    section(f"dimensionality reduction  ·  {normalisation}")
    info(f"{'input':<{w}}  {n_seq:,} sequences  ×  {n_feat:,} features")
    info(f"{'methods':<{w}}  {', '.join(methods)}")
    scale_label = f"{resolved_scale}  (auto)" if scale == "auto" else resolved_scale
    info(f"{'scale / dims':<{w}}  {scale_label}  /  {dims}D")
    if n_jobs > 1:
        info(f"{'threads':<{w}}  {n_jobs}")

    divider()

    dfs = []
    graph_paths: list[Path] = []

    for method in methods:

        method = method.lower()

        method_dir = output_dir / normalisation / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Parameter screening
        if screen_params and method in SCREENABLE_METHODS:
            screen_dir = method_dir / "parameter_screen"
            screen_dir.mkdir(parents=True, exist_ok=True)

            _run_parameter_screen(
                X=X,
                method=method,
                dims=dims,
                seed=seed,
                scale=resolved_scale,
                output_dir=screen_dir,
                normalisation=normalisation,
                input_name=input_name,
                sequence_ids=sequence_ids,
                n_jobs=n_jobs,
            )

        # Default embedding
        m = 14
        params = DR_HYPERPARAMS.get(method, {}).get(resolved_scale, {})
        info(f"{method:<{m}}  running")
        if params:
            params_str = "  ".join(f"{k}={v}" for k, v in params.items())
            info(f"{'':>{m}}  {params_str}")
        t0 = time.perf_counter()

        embedding, graph = _run_single_method(
            X=X,
            method=method,
            dims=dims,
            seed=seed,
            scale=resolved_scale,
            n_jobs=n_jobs,
        )

        elapsed = time.perf_counter() - t0

        # Save embedding
        columns = [f"{method}_{i+1}" for i in range(dims)]
        df_embed = pd.DataFrame(embedding, columns=columns)
        df_embed.insert(0, "sequence_id", sequence_ids)

        out_file = method_dir / f"{input_name}_{normalisation}_{method}_{dims}D.tsv"
        df_embed.to_csv(out_file, sep="\t", index=False)

        # Save graph if available
        graph_note = ""
        if graph is not None:
            graph_file = method_dir / f"{input_name}_{normalisation}_{method}_graph.npz"
            sparse.save_npz(graph_file, graph)
            graph_paths.append(graph_file)
            graph_note = "  (+ graph)"

        info(f"{'':>{m}}  done  {_fmt_time(elapsed)}{graph_note}")
        divider()

        dfs.append(df_embed)

    # Merge embeddings across methods
    merged_df = pd.concat(dfs, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_file = output_dir / normalisation / f"{input_name}_{normalisation}_{dims}D_merged_embeddings.tsv"
    merged_df.to_csv(merged_file, sep="\t", index=False)
    info(f"merged  →  {normalisation}/{merged_file.name}")

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

    import time as _time

    pw = 36  # fixed width for params column so elapsed times align

    if method == "umap":
        n_neighbors_values = [5, 10, 50, 100, 150]
        min_dist_values = [0, 0.1, 0.25, 0.5, 1.0]
        section(f"parameter screen  ·  umap  ({len(n_neighbors_values) * len(min_dist_values)} combinations)")
        for n in n_neighbors_values:
            for m in min_dist_values:
                t0 = _time.perf_counter()
                model = umap.UMAP(n_components=dims, n_neighbors=n, min_dist=m, n_jobs=n_jobs)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_min{m}")
                info(f"{'n_neighbors=' + str(n) + '  min_dist=' + str(m):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "tsne":
        perplexity_values = [5, 10, 30, 50, 100]
        learning_rate_values = [10, 100, 200, 500]
        section(f"parameter screen  ·  tsne  ({len(perplexity_values) * len(learning_rate_values)} combinations)")
        for p in perplexity_values:
            for lr in learning_rate_values:
                t0 = _time.perf_counter()
                model = TSNE(n_components=dims, perplexity=p, learning_rate=lr,
                             max_iter=1000, random_state=seed, n_jobs=n_jobs)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"p{p}_lr{lr}")
                info(f"{'perplexity=' + str(p) + '  learning_rate=' + str(lr):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "trimap":
        n_inliers_values = [10, 25, 50, 100, 150]
        weight_temp_values = [0.1, 0.5, 1.0, 2.0, 2.5]
        section(f"parameter screen  ·  trimap  ({len(n_inliers_values) * len(weight_temp_values)} combinations)")
        for n in n_inliers_values:
            for w in weight_temp_values:
                t0 = _time.perf_counter()
                model = TRIMAP(n_dims=dims, n_inliers=n, weight_temp=w)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"inliers{n}_weighttemp{w}")
                info(f"{'n_inliers=' + str(n) + '  weight_temp=' + str(w):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "pacmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]
        section(f"parameter screen  ·  pacmap  ({len(n_neighbors_values) * len(FP_ratio_values)} combinations)")
        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                t0 = _time.perf_counter()
                model = PaCMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")
                info(f"{'n_neighbors=' + str(n) + '  FP_ratio=' + str(fp):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    elif method == "localmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]
        section(f"parameter screen  ·  localmap  ({len(n_neighbors_values) * len(FP_ratio_values)} combinations)")
        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                t0 = _time.perf_counter()
                model = LocalMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                save_embedding(embedding, param_str=f"n{n}_FPratio{fp}")
                info(f"{'n_neighbors=' + str(n) + '  FP_ratio=' + str(fp):<{pw}}  {_fmt_time(_time.perf_counter() - t0)}")

    else:
        raise ValueError(f"Parameter screening not implemented for method: {method}")

    return output_paths