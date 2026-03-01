# src/kmer_ord/vis/embedding_plots.py

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

import datashader as ds
import datashader.transfer_functions as tf
import datashader.colors as dc

from colorcet import fire, CET_L17
from datashader.utils import export_image
from PIL import Image, ImageDraw, ImageFont

from kmer_ord.dr import methods
from kmer_ord.utils.logging_utils import section, info, warn

def _connect_spatialite(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)

    try:
        conn.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        # On macOS / conda sometimes name differs
        try:
            conn.execute("SELECT load_extension('mod_spatialite.so');")
        except Exception as e:
            raise RuntimeError(
                "SpatiaLite extension could not be loaded. "
                "Ensure mod_spatialite is installed."
            ) from e

    return conn


def plot_embeddings_from_db(db_path: Path, output_root: Path, mode: str = "all"):
    conn = _connect_spatialite(db_path)
    methods = _discover_geometry_methods(conn)
    if not methods:
        return

    features_df = _load_features(conn)
    cluster_df = _load_clusters(conn)

    # Store precomputed images per method/feature
    per_method_feature = {method: {"categorical": {}, "continuous": {}} for method in methods}

    for method in methods:
        emb_df = _extract_method_coordinates(conn, method)
        if emb_df.empty:
            continue

        merged = _merge_tables(emb_df, features_df, cluster_df)
        xcol = f"{method}_1"
        ycol = f"{method}_2"
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        # Compute and save density plots
        if mode in ("density", "all"):
            _render_density(merged, xcol, ycol, method_dir, method)

        # Loop over categorical features
        if mode in ("categorical", "all"):
            for col in features_df.select_dtypes(include=["object", "category"]).columns:
                img = _render_feature_to_image(merged, xcol, ycol, col, mode="categorical")
                if img is not None:
                    per_method_feature[method]["categorical"][col] = img
                    export_image(img, filename=f"{method}_by_{col}", export_path=str(method_dir))

        # Loop over continuous features
        if mode in ("continuous", "all"):
            for col in features_df.select_dtypes(include=[np.number]).columns:
                img = _render_feature_to_image(merged, xcol, ycol, col, mode="continuous")
                if img is not None:
                    per_method_feature[method]["continuous"][col] = img
                    export_image(img, filename=f"{method}_by_{col}", export_path=str(method_dir))

    # Build feature composites from cached images
    for col in features_df.select_dtypes(include=["object", "category"]).columns:
        _render_feature_composite_from_cache(per_method_feature, col, output_root / "feature_composites", mode="categorical")

    for col in features_df.select_dtypes(include=[np.number]).columns:
        _render_feature_composite_from_cache(per_method_feature, col, output_root / "feature_composites", mode="continuous")

    conn.close()


def _discover_geometry_methods(conn):
    query = """
        SELECT f_geometry_column
        FROM geometry_columns
        WHERE f_table_name = 'coordinates';
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        return []

    return df["f_geometry_column"].tolist()

def _extract_method_coordinates(conn, method):
    select_sql = f"""
        SELECT
            sequence_id,
            ST_X({method}) AS {method}_1,
            ST_Y({method}) AS {method}_2
        FROM coordinates;
    """
    df = pd.read_sql_query(select_sql, conn)
    return df.dropna(subset=[f"{method}_1", f"{method}_2"])


def _load_features(conn):
    try:
        return pd.read_sql_query("SELECT * FROM features;", conn)
    except Exception:
        return None


def _load_clusters(conn):
    try:
        return pd.read_sql_query("SELECT * FROM clusters;", conn)
    except Exception:
        return None


def _merge_tables(emb_df, features_df, cluster_df=None):
    df = emb_df.copy()
    if features_df is not None:
        df = df.merge(features_df, on="sequence_id", how="left") 
    if cluster_df is not None:
        df = df.merge(cluster_df, on="sequence_id", how="left")
    
    return df


# ============================================================

def _base_canvas(df, xcol, ycol, width=1200, height=1000):
    return ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(df[xcol].min(), df[xcol].max()),
        y_range=(df[ycol].min(), df[ycol].max()),
    )


def _render_density(df, xcol, ycol, outdir, method):
    info(f"plotting datashader density for method: {method}")

    cvs = _base_canvas(df, xcol, ycol)
    agg = cvs.points(df, xcol, ycol)
    img = tf.shade(agg, cmap=fire)
    img = tf.dynspread(img)

    export_image(img, filename=f"{method}_density", export_path=str(outdir))


def _render_feature_to_image(df, xcol, ycol, feature_col, mode="categorical", width=400, height=400):
    """
    Generate a datashader image for a single feature (categorical or continuous) for one DR method.
    Returns a datashader.Image (can be exported later or converted to PIL for composites).
    """
    df_local = df[[xcol, ycol, feature_col]].dropna().copy()
    if df_local.empty:
        return None

    cvs = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(df_local[xcol].min(), df_local[xcol].max()),
        y_range=(df_local[ycol].min(), df_local[ycol].max())
    )

    if mode == "categorical":
        df_local[feature_col] = df_local[feature_col].astype(str).astype("category")
        agg = cvs.points(df_local, xcol, ycol, ds.by(feature_col, ds.count()))
        categories = list(df_local[feature_col].cat.categories)

        if not categories:
            return None

        if len(categories) == 1:
            color_key = {categories[0]: "#FF0000"}
        else:
            palette = [fire[i % len(fire)] for i in range(len(categories))]
            color_key = dict(zip(categories, palette))

        img = tf.shade(agg, color_key=color_key, how="eq_hist", alpha=255)

    elif mode == "continuous":
        df_local[feature_col] = df_local[feature_col].astype(float)
        agg = cvs.points(df_local, xcol, ycol, ds.mean(feature_col))
        img = tf.shade(agg, cmap=CET_L17, how="eq_hist", alpha=255)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    img = tf.dynspread(img)
    return img


PREFERRED_DR_ORDER = ["pca", "trimap", "pacmap", "localmap", "umap", "tsne"]
IGNORE_FEATURES = {"sequence_id", "header"}

def _render_feature_composite_from_cache(per_method_feature, feature_col, outdir, mode, width=400, height=400, padding=10, border=2):
    """
    Combine per-method images into a horizontal composite with borders and DR labels.

    Parameters
    ----------
    per_method_feature : dict
        Dictionary of {method: {"categorical": {}, "continuous": {}}}.
    feature_col : str
        Feature to visualize.
    outdir : Path
        Output directory.
    mode : str
        "categorical" or "continuous".
    width, height : int
        Width/height of each subplot.
    padding : int
        Space between subplots.
    border : int
        Width of the border around each subplot.
    """
    if feature_col in IGNORE_FEATURES:
        # Skip internal columns like sequence_id
        return

    images = []
    methods_present = [m for m in PREFERRED_DR_ORDER if m in per_method_feature]
    
    for method in methods_present:
        img = per_method_feature[method][mode].get(feature_col)
        if img is not None:
            pil_img = img.to_pil().resize((width, height))
            images.append((method, pil_img))

    if not images:
        info(f"[!] No images for feature {feature_col}")
        return

    # Calculate composite size
    total_width = sum(img.width for _, img in images) + padding * (len(images) - 1)
    max_height = max(img.height for _, img in images) + 40  # extra space for labels

    composite = Image.new("RGBA", (total_width, max_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(composite)

    # Optional: load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    x_offset = 0
    for method, img in images:
        # Draw border
        border_box = [x_offset, 30, x_offset + img.width, 30 + img.height]
        draw.rectangle(border_box, outline="black", width=border)

        # Paste image inside border
        composite.paste(img, (x_offset, 30))

        # Draw method label
        try:
            bbox = draw.textbbox((0, 0), method, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(method)

        text_x = x_offset + (img.width - text_w) // 2
        draw.text((text_x, 5), method, fill="black", font=font)

        x_offset += img.width + padding

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"feature_composite_{feature_col}.png"
    composite.save(out_path)
    info(f"Saved : {out_path}")