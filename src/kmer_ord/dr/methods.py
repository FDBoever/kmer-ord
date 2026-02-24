# src/kmer_ord/dr/methods.py

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

try:
    import umap
except ImportError:
    umap = None

from trimap import TRIMAP
from pacmap import PaCMAP
from pacmap.pacmap import LocalMAP
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


def _run_single_method(X: np.ndarray, method: str, dims: int, seed: int,scale: str = "default",):
    method = method.lower()
    params = DR_HYPERPARAMS.get(method, {}).get(scale, {})
    
    if method == "pca":
        model = PCA(n_components=dims, **params)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        model = TSNE(n_components=dims, random_state=seed, **params)
    elif method == "umap":
        import umap
        #model = umap.UMAP(n_components=dims, random_state=seed, **params)
        model = umap.UMAP(n_components=dims, **params)
    elif method == "trimap":
        from trimap import TRIMAP
        model = TRIMAP(n_dims=dims, **params)
    elif method == "pacmap":
        from pacmap import PaCMAP
        model = PaCMAP(n_components=dims, **params)
    elif method == "localmap":
        from pacmap.pacmap import LocalMAP
        model = LocalMAP(n_components=dims, **params)
    elif method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        n_neighbors = params.get("n_neighbors", 10)
        model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dims)
    elif method == "sparse_pca":
        from sklearn.decomposition import SparsePCA
        model = SparsePCA(n_components=dims, **params)
    elif method == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=dims, **params)
    else:
        raise ValueError(f"Unsupported DR method: {method}")

    return model.fit_transform(X)


def run_dr_methods(X: np.ndarray,
                   methods: list[str],
                   dims: int,
                   seed: int,
                   scale: str,
                   screen_params: bool,
                   output_dir: Path,
                   normalisation: str,
                   input_name: str,) -> Path:
    """
    Run selected DR methods for a single normalisation.

    - Saves individual method files
    - Optionally performs parameter screening
    - Creates ONE merged file across methods (same normalisation)
    - Returns merged file path
    """

    if "all" in methods:
        methods = ALL_METHODS

    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for method in methods:

        method_dir = output_dir / normalisation / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # parameter screening
        # --------------------------------------------------
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
            )

        # Default embedding
        # --------------------------------------------------
        embedding = _run_single_method(
            X=X,
            method=method,
            dims=dims,
            seed=seed,
            scale=scale,
        )

        columns = [f"{method}_{i+1}" for i in range(dims)]
        df_embed = pd.DataFrame(embedding, columns=columns)

        out_file = (
            method_dir
            / f"{input_name}_{normalisation}_{method}_{dims}D.tsv"
        )

        df_embed.to_csv(out_file, sep="\t", index=False)
        print(f"Saved {method} ({normalisation}) > {out_file}")

        dfs.append(df_embed)

    # merge across methods (same normalisation)
    # --------------------------------------------------
    merged_df = pd.concat(dfs, axis=1)

    merged_file = (
        output_dir
        / normalisation
        / f"{input_name}_{normalisation}_merged_embeddings.tsv"
    )

    merged_df.to_csv(merged_file, sep="\t", index=False)
    print(f"Saved merged embeddings ({normalisation}) > {merged_file}")

    return merged_file


def _run_parameter_screen(X: pd.DataFrame,
                          method: str,
                          dims: int,
                          seed: int,
                          scale: str,
                          output_dir: Path,
                          normalisation: str,
                          input_name: str,) -> list[Path]:
    """
    Perform parameter screening for a given DR method.
    Saves individual files for each parameter combination, returns list of saved paths.
    """
    output_paths = []

    if method == "umap":
        n_neighbors_values = [5, 10, 50, 100, 150]
        min_dist_values = [0, 0.1, 0.25, 0.5, 1.0]

        for n in n_neighbors_values:
            for m in min_dist_values:
                print(f"     ... UMAP with n_neighbors={n}, min_dist={m}", flush=True)
                #model = UMAP(n_components=dims, n_neighbors=n, min_dist=m, random_state=seed)
                model = umap.UMAP(n_components=dims, n_neighbors=n, min_dist=m)
                embedding = model.fit_transform(X)
                df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
                param_str = f"n{n}_min{m}"
                out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
                df.to_csv(out_file, sep="\t", index=False)
                output_paths.append(out_file)

    elif method == "tsne":
        perplexity_values = [5, 10, 30, 50, 100]
        learning_rate_values = [10, 100, 200, 500]

        for p in perplexity_values:
            for lr in learning_rate_values:
                print(f"     ... t-SNE with perplexity={p}, learning_rate={lr}", flush=True)
                model = TSNE(n_components=dims, perplexity=p, learning_rate=lr, max_iter=1000, random_state=seed)
                embedding = model.fit_transform(X)
                df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
                param_str = f"p{p}_lr{lr}"
                out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
                df.to_csv(out_file, sep="\t", index=False)
                output_paths.append(out_file)

    elif method == "trimap":
        n_inliers_values = [10, 25, 50, 100, 150]
        weight_temp_values = [0.1, 0.5, 1.0, 2.0, 2.5]

        for n in n_inliers_values:
            for w in weight_temp_values:
                print(f"     ... TRIMAP with n_inliers={n}, weight_temp={w}", flush=True)
                model = TRIMAP(n_dims=dims, n_inliers=n, weight_temp=w)
                embedding = model.fit_transform(X)
                df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
                param_str = f"inliers{n}_weighttemp{w}"
                out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
                df.to_csv(out_file, sep="\t", index=False)
                output_paths.append(out_file)

    elif method == "pacmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]

        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                print(f"     ... PaCMAP with n_neighbors={n}, FP_ratio={fp}", flush=True)
                model = PaCMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
                param_str = f"n{n}_FPratio{fp}"
                out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
                df.to_csv(out_file, sep="\t", index=False)
                output_paths.append(out_file)

    elif method == "localmap":
        n_neighbors_values = [10, 25, 50, 100, 150]
        FP_ratio_values = [0.1, 0.5, 1.0, 2.0, 5]

        for n in n_neighbors_values:
            for fp in FP_ratio_values:
                print(f"     ... LocalMAP with n_neighbors={n}, FP_ratio={fp}", flush=True)
                model = LocalMAP(n_components=dims, n_neighbors=n, FP_ratio=fp)
                embedding = model.fit_transform(X)
                df = pd.DataFrame(embedding, columns=[f"{method}_{i+1}" for i in range(dims)])
                param_str = f"n{n}_FPratio{fp}"
                out_file = output_dir / f"{input_name}_{normalisation}_{method}_{param_str}_{dims}D.tsv"
                df.to_csv(out_file, sep="\t", index=False)
                output_paths.append(out_file)

    else:
        raise ValueError(f"Parameter screening not implemented for method: {method}")

    return output_paths