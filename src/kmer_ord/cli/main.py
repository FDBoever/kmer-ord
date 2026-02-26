# src/kmer_ord/cli/main.py
import typer
from pathlib import Path
from kmer_ord.workflow import context
from kmer_ord.workflow.context import Context
from kmer_ord.workflow.runner import Runner
from kmer_ord.utils.logging_utils import section, info, warn
from kmer_ord.cli.setup import setup_app

#app = typer.Typer(add_completion=False, rich_markup_mode=None)
app = typer.Typer(add_completion=False,
                  context_settings={"help_option_names": ["-h", "--help"]})

app.add_typer(setup_app)

# -----------------------------
# Pipeline
# -----------------------------
@app.command("run", rich_help_panel="Pipeline")
def run_pipeline(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta/fastq file (can be gzipped)"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    force: bool = typer.Option(False, "-f","--force", help="Force recomputation even if outputs exist"),
    kmer_length: int = typer.Option(6,"-k", "--kmer", help="K-mer length"),
    threads: int = typer.Option(4, "-t","--threads", help="Number of threads"),

    # --- DR options ---
    dr_methods: str = typer.Option("umap","--dr", help="Comma-separated DR methods (default: umap)"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method (raw, relative, log, clr, zscore)"),
    dims: int = typer.Option(2, "-d","--dims", help="Embedding dimensions"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None,"--keep-pcs", help="Number of principal components to retain"),
    keep_variance: float = typer.Option(None,"--keep-variance",help="Variance threshold for PCA (e.g. 0.9)"),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),
):
    """
    Run the full kmer-ord pipeline:
    fastq -> fasta -> sequence stats -> kmer-counting -> DR
    """
    print("-" * 70)
    section("Starting kmer-ord...")
    info("importing packages...")
    from kmer_ord.io.sequence import fastq_to_fasta
    from kmer_ord.io.summary import calculate_stats
    from kmer_ord.workflow.operations import (
        FastqToFasta, FastaStats, KmerCount, KmerMetrics, Tiara,
        DimensionalityReduction, FeatureMerge, SpatialiteDatabase)

    context = Context(input, output_dir, force=force)

    method_list = [m.strip().lower() for m in dr_methods.split(",")]
    norm_list = [n.strip().lower() for n in normalisation.split(",")]

    operations = [
        FastqToFasta(),
        FastaStats(),
        KmerCount(kmer_length=kmer_length, threads=threads),
        KmerMetrics(chunksize=1000, cpus=threads),
        Tiara(threads=threads),
        DimensionalityReduction(methods=method_list, normalisations=norm_list,
                                dims=dims, pca_dim_red=pca_pre, 
                                keep_pcs=keep_pcs, keep_variance=keep_variance,
                                screen_params=screen_params, 
                                scale=scale),
        FeatureMerge(),
        SpatialiteDatabase()]

    runner = Runner(operations)
    runner.run(context)

    # print all artifacts 
    print("\n")
    print("-" * 70)
    typer.echo("Generated output:")
    for name, path in context.artifacts.items():
        if isinstance(path, list):
            typer.echo(f"  {name}:")
            for p in path:
                typer.echo(f"    - {p}")
        else:
            typer.echo(f"  {name}: {path}")
    print("-" * 70)
    section("Done.")
    print("-" * 70)

 
@app.command("discover", rich_help_panel="Pipeline")
def discover_pipeline(
    input: Path = typer.Option(..., "-i", "--input", help="Input fasta/fastq file"),
    output_dir: Path = typer.Option(..., "-o", "--output", help="Output directory"),
    kmer_length: int = typer.Option(6, "-k", "--kmer"),
    dims: int = typer.Option(15, "-d", "--dims", help="High-dimensional embedding size"),
    dr_method: str = typer.Option("umap", "--dr"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    norm: str = typer.Option("clr", "--norm"),
    cluster_methods: str = typer.Option("hdbscan", "--cluster", help="Comma-separated clustering methods (leiden,hdbscan,dbscan)"),
    leiden_sweep: bool = typer.Option(False, "--leiden-sweep", help="Run Leiden resolution sweep"),
    hdbscan_sweep: bool = typer.Option(False, "--hdbscan-sweep", help="Run HDBSCAN min_cluster_size sweep"),
    dbscan_sweep: bool = typer.Option(False, "--dbscan-sweep", help="Run DBSCAN eps sweep"),
    threads: int = typer.Option(4, "-t", "--threads"),
    force: bool = typer.Option(False, "-f", "--force"),
    db_path: Path = typer.Option(None, "--db", help="Optional path to existing SQLite/SpatiaLite DB"),):
    """
    Structure discovery pipeline:
    High-D embedding + clustering + database integration.
    """

    from kmer_ord.workflow.operations import (
        FastqToFasta,
        FastaStats,
        KmerCount,
        KmerMetrics,
        DimensionalityReduction,
        Clustering,
        AddClusteringToDB,
    )

    section("Starting structure discovery...")

    context = Context(input, output_dir, force=force)

    cluster_list = [c.strip().lower() for c in cluster_methods.split(",")]

    # -----------------------------
    # Core operations
    # -----------------------------
    operations = [
        FastqToFasta(),
        FastaStats(),
        KmerCount(kmer_length=kmer_length, threads=threads),
        KmerMetrics(),
        DimensionalityReduction(
            methods=[dr_method],
            normalisations=[norm],
            dims=dims, scale=scale,
        ),
    ]

    # -----------------------------
    # Clustering operations
    # -----------------------------
    for method in cluster_list:

        if method == "leiden":
            operations.append(
                Clustering(
                    method="leiden",
                    sweep=leiden_sweep,
                )
            )

        elif method == "hdbscan":
            operations.append(
                Clustering(
                    method="hdbscan",
                    sweep=hdbscan_sweep,
                )
            )

        elif method == "dbscan":
            operations.append(
                Clustering(
                    method="dbscan",
                    sweep=dbscan_sweep,
                )
            )

        else:
            raise ValueError(f"Unknown clustering method: {method}")

    runner = Runner(operations)
    runner.run(context)

    # -----------------------------
    # Database integration
    # -----------------------------
    if db_path is None:
        db_path = output_dir / "discovery.sqlite"

    add_db_op = AddClusteringToDB(db_path=db_path, force=force)
    add_db_op.run(context)

    section(f"Discovery complete. Database saved at: {db_path}")


@app.command("embed", rich_help_panel="Modeling")
def embed_pipeline(
    input: Path = typer.Option(..., "-i", "--input"),
    output_dir: Path = typer.Option(..., "-o", "--output"),
    kmer_length: int = typer.Option(6, "-k", "--kmer"),
    dims: int = typer.Option(20, "-d", "--dims"),
    dr_method: str = typer.Option("umap", "--dr"),
    norm: str = typer.Option("clr", "--norm"),
    threads: int = typer.Option(4, "-t", "--threads"),
    force: bool = typer.Option(False, "-f", "--force")):
    """
    Generate high-dimensional embedding only.
    """

    from kmer_ord.workflow.operations import (
        FastqToFasta,
        FastaStats,
        KmerCount,
        KmerMetrics,
        DimensionalityReduction)

    context = Context(input, output_dir, force=force)

    operations = [
        FastqToFasta(),
        FastaStats(),
        KmerCount(kmer_length=kmer_length, threads=threads),
        KmerMetrics(),
        DimensionalityReduction(
            methods=[dr_method],
            normalisations=[norm],
            dims=dims,
        ),
    ]

    runner = Runner(operations)
    runner.run(context)

    info("Embedding complete.")


@app.command("cluster", rich_help_panel="Modeling")
def cluster_pipeline(
    input: Path = typer.Option(..., "-i", "--input", help="Input directory containing artifacts"),
    output_dir: Path = typer.Option(..., "-o", "--output"),
    method: str = typer.Option("hdbscan", "--method"),
    force: bool = typer.Option(False, "-f", "--force"),
):
    """
    Cluster sequences using existing embedding.
    """

    from kmer_ord.workflow.operations import (
        Clustering,
        SpatialiteDatabase,
    )

    context = Context(input, output_dir, force=force)

    operations = [
        Clustering(method=method),
        SpatialiteDatabase(),
    ]

    runner = Runner(operations)
    runner.run(context)

    info("Clustering complete.")

# -----------------------------
# fastq to fasta
@app.command("fastq-to-fasta")
def fastq_to_fasta_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fastq file (can be gzipped)"),
    output: Path = typer.Option(..., "-o","--output", help="Output fasta file"),
    force: bool = typer.Option(False, "-f","--force", help="Overwrite output if it exists"),
):
    """
    Convert fastq (or fastq.gz) to fasta.
    """
    from kmer_ord.workflow.operations import FastqToFasta
    from kmer_ord.io.sequence import fastq_to_fasta
    if output.exists() and not force:
        info(f"Skipping conversion, FASTA already exists: {output}")
    else:
        fastq_to_fasta(input, output)
        info(f"fastq -> fasta conversion done: {output}")


# -----------------------------
# FASTA stats
@app.command("fasta-stats")
def fasta_stats_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    force: bool = typer.Option(False, "-f","--force", help="Recalculate stats even if outputs exist"),
):
    """
    Calculate per-sequence and overall statistics from a fasta file.
    """
    from kmer_ord.io.summary import calculate_stats
   
    context = Context(input, output_dir, force=force)
    df, overall_file, tsv_file = calculate_stats(
        input_fasta=context.fasta,
        output_dir=context.output_dir / "summary"
    )
    info(f"Stats calculated. Sequence-level tsv: {tsv_file}, Overall: {overall_file}")

# -----------------------------
# K-mer counting
@app.command("kmer-count")
def kmer_count_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    kmer_length: int = typer.Option(6, "-k", "--kmer", help="K-mer length"),
    threads: int = typer.Option(1, "-t", "--threads", help="Number of threads for counting"),
    force: bool = typer.Option(False, "-f","--force", help="Recalculate even if output exists")):
    """
    Count k-mers for a fasta file and save tsv matrix.
    """
    context = Context(input, output_dir, force=force)
    from kmer_ord.workflow.operations import KmerCount

    operation = KmerCount(kmer_length=kmer_length, threads=threads)
    operation.run(context)

    info(f"K-mer counting complete. Matrix saved at: {context.get('kmer_matrix')}")

if __name__ == "__main__":
    app()


# -----------------------------
# kmer-metrics
@app.command("kmer-metrics")
def kmer_metrics_cmd(
    input: Path = typer.Option(..., "-i", "--input",help="Input k-mer matrix TSV"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    chunksize: int = typer.Option(1000, "--chunksize", help="Rows per chunk"),
    cpus: int = typer.Option(1, "--cpus", help="Number of worker processes"),
    force: bool = typer.Option(False, "-f","--force", help="Recompute even if output exists"),):
    """
    Compute per-sequence k-mer metrics (Shannon diversity, unique k-mers, etc.).
    """
    from kmer_ord.workflow.operations import KmerMetrics

    context = Context(input, output_dir, force=force)
    operation = KmerMetrics(chunksize=chunksize, cpus=cpus)
    operation.run(context)
    info(f"K-mer metrics saved at: {context.get('kmer_metrics')}")


# -----------------------------
# DR
@app.command("dr")
def dr_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input k-mer matrix tsv"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    methods: str = typer.Option(..., "-m","--methods", help="Comma-separated DR methods"),
    scale: str = typer.Option("auto", "-s","--scale", help="Dataset scale presets for DR hyperparameters (auto, small, medium, large, default)"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method"),
    dims: int = typer.Option(2,"-d", "--dims", help="Embedding dimensions"),
    force: bool = typer.Option(False, "-f","--force", help="Recompute even if output exists"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None, "--keep-pcs"),
    keep_variance: float = typer.Option(None, "--keep-variance"),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),):
    """
    Run dimensionality reduction on an existing k-mer matrix.
    """
    from kmer_ord.workflow.operations import DimensionalityReduction

    context = Context(input, output_dir, force=force)

    method_list = [m.strip().lower() for m in methods.split(",")]
    norm_list = [n.strip().lower() for n in normalisation.split(",")]

    operation = DimensionalityReduction(
        methods=method_list,
        normalisations=norm_list,
        dims=dims,
        pca_dim_red=pca_pre,
        keep_pcs=keep_pcs,
        keep_variance=keep_variance,
        screen_params=screen_params,
        scale=scale,
    )

    operation.run(context)

    info(f"DR embeddings saved at: {context.get('dr_embeddings')}")


@app.command("run-tiara")
def run_tiara_cmd(
    input: Path = typer.Option(..., "-i","--input", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory"),
    threads: int = typer.Option(1, "-t", help="Number of threads"),
    force: bool = typer.Option(False, "-f", "--force", help="Recompute even if output exists"),):
    """
    Run Tiara classification on a fasta file.
    """
    from kmer_ord.workflow.operations import Tiara

    context = Context(input, output_dir, force=force)

    operation = Tiara(threads=threads)
    operation.run(context)

    info(f"Tiara output saved at: {context.get('tiara')}")


@app.command("build-db")
def build_database(
    input: Path = typer.Option(..., "-i", "--input", help="Input directory containing artifacts"),
    output_dir: Path = typer.Option(..., "-o","--output", help="Output directory for database"),
    force: bool = typer.Option(False, "-f", "--force", help="Recompute even if output exists"),):
    """
    Build Spatialite database from available artifacts.
    """
    from kmer_ord.workflow.operations import (FeatureMerge, SpatialiteDatabase)

    context = Context(input, output_dir, force=force)

    operation = SpatialiteDatabase()
    operation.run(context)

    info(f"Database created at: {context.get('database')}")