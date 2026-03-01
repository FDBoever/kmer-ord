import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib as Path
from kmer_ord.utils.logging_utils import section, info, warn

sns.set_theme(style="ticks")
plt.rcParams.update({
    "figure.figsize": (5, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})

def plot_single_distribution(df: pd.DataFrame, column: str, output_dir):
    plt.figure()
    sns.histplot(df[column].dropna(), kde=False, bins=100)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / f"{column}_distribution.png", dpi=300)
    plt.close()
    info(f"Saved: {output_dir / f'{column}_distribution.png'}")


def plot_numeric_composite(df: pd.DataFrame, output_dir):
    numeric_cols = df.select_dtypes(include=["number"]).columns
    n = len(numeric_cols)
    if n == 0:
        return
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), bins=50, ax=axes[i])
        axes[i].set_title(col)
        sns.despine(ax=axes[i])
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_composite.png", dpi=300)
    plt.close()
    info(f"Saved: {output_dir / 'numeric_composite.png'}")


def plot_boxplot(df: pd.DataFrame, categorical: str, numeric: str, output_dir):
    plt.figure()
    order = df.groupby(categorical)[numeric].median().sort_values().index
    palette = sns.color_palette("Set2", n_colors=len(order))
    palette_dict = dict(zip(order, palette))
    sns.boxplot(
        data=df, x=categorical, y=numeric, order=order,
        hue=categorical, palette=palette_dict, width=0.35,
        showcaps=True, whiskerprops={"linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.4},
        showfliers=False, legend=False
    )
    sns.stripplot(
        data=df, x=categorical, y=numeric, order=order,
        hue=categorical, palette=palette_dict,
        jitter=True, edgecolor='k', linewidth=0.5, size=3
    )
    plt.title(f"{numeric} by {categorical}")
    plt.xlabel("")
    plt.ylabel(numeric)
    if len(order) > 3:
        plt.xticks(rotation=30, ha="right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / f"{numeric}_by_{categorical}_boxplot.png", dpi=300)
    plt.close()
    info(f"Saved: {output_dir / f'{numeric}_by_{categorical}_boxplot.png'}")


def plot_categorical_composite(df, categorical, numeric_cols, output_dir):
    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    if pd.api.types.is_categorical_dtype(df[categorical]):
        order = df[categorical].cat.categories
    else:
        order = df[categorical].dropna().unique()
    palette = sns.color_palette("Set2", n_colors=len(order))
    palette_dict = dict(zip(order, palette))
    for i, num_col in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(
            data=df, x=categorical, y=num_col, order=order,
            hue=categorical, palette=palette_dict, width=0.3,
            showcaps=True, whiskerprops={"linewidth": 1},
            medianprops={"color": "black", "linewidth": 1.4},
            showfliers=False, legend=False, ax=ax
        )
        sns.stripplot(
            data=df, x=categorical, y=num_col, order=order,
            hue=categorical, palette=palette_dict,
            jitter=0.075, size=2.5, alpha=0.7,
            edgecolor="black", linewidth=0.3, legend=False, ax=ax
        )
        ax.set_title(num_col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(output_dir / f"{categorical}_composite.png", dpi=300)
    plt.close()
    info(f"Saved: {output_dir / f'{categorical}_composite.png'}")


def read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load a table from a SQLite database into a DataFrame."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def plot_numeric_distributions(db_path: Path, table_name: str, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    df = read_table(db_path, table_name)
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) == 0:
        info("No numeric columns found.")
        return

    for col in numeric_cols:
        plot_single_distribution(df, col, output_dir)

    plot_numeric_composite(df, output_dir)


def plot_categorical_vs_numeric(db_path: Path, table_name: str, output_dir: Path, max_categories: int = 10):
    output_dir.mkdir(exist_ok=True, parents=True)
    df = read_table(db_path, table_name)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) == 0 or len(categorical_cols) == 0:
        info("No numeric or categorical columns found.")
        return

    for cat_col in categorical_cols:
        n_unique = df[cat_col].nunique()
        if n_unique == 0 or n_unique > max_categories:
            info(f"Skipping {cat_col} (unique values: {n_unique})")
            continue

        for num_col in numeric_cols:
            plot_boxplot(df, cat_col, num_col, output_dir)

        plot_categorical_composite(df, cat_col, numeric_cols, output_dir)