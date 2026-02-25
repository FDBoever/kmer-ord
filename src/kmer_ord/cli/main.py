# src/kmer_ord/cli/main.py


from kmer_ord.utils.logging_utils import section, info, warn
from kmer_ord.workflow import context
import typer
from pathlib import Path
from kmer_ord.io.sequence import fastq_to_fasta
from kmer_ord.io.summary import calculate_stats

from kmer_ord.workflow.context import Context
from kmer_ord.workflow.runner import Runner
#from kmer_ord.workflow.operations import FastqToFasta, FastaStats, FeatureMerge, KmerCount, SpatialiteDatabase
#from kmer_ord.workflow.operations import DimensionalityReduction
#from kmer_ord.workflow.operations import KmerMetrics
#from kmer_ord.workflow.operations import Tiara

#load setup cli
from kmer_ord.cli.setup import setup_app

#app = typer.Typer(add_completion=False, rich_markup_mode=None)
app = typer.Typer(add_completion=False)

app.add_typer(setup_app)

# -----------------------------
# Pipeline
# -----------------------------
@app.command("run", rich_help_panel="Pipeline")
def run_pipeline(
    input: Path = typer.Option(..., "-i"),
    output_dir: Path = typer.Option(..., "-o"),
    force: bool = typer.Option(False, "--force", help="Force recomputation even if outputs exist"),
    kmer_length: int = typer.Option(6, "--kmer", help="K-mer length"),
    threads: int = typer.Option(4, "--threads", help="Number of threads"),
    #kmer_counter_path: str = typer.Option(None, help="Path to kmer-counter binary"),

    # --- DR options ---
    dr_methods: str = typer.Option("umap","--dr", help="Comma-separated DR methods (default: umap)"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method (raw, relative, log, clr, zscore)"),
    dims: int = typer.Option(2, "--dims", help="Embedding dimensions"),
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
    from kmer_ord.workflow.operations import (
        FastqToFasta, FastaStats, KmerCount, KmerMetrics, Tiara,
        DimensionalityReduction, FeatureMerge, SpatialiteDatabase
    )

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
                                screen_params=screen_params),
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

  
# -----------------------------
# fastq to fasta
@app.command("fastq-to-fasta")
def fastq_to_fasta_cmd(
    input: Path = typer.Option(..., "-i"),
    output: Path = typer.Option(..., "-o"),
    force: bool = typer.Option(False, "--force", help="Overwrite output if it exists"),
):
    """
    Convert fastq (or fastq.gz) to fasta.
    """
    from kmer_ord.workflow.operations import FastqToFasta
    if output.exists() and not force:
        info(f"Skipping conversion, FASTA already exists: {output}")
    else:
        fastq_to_fasta(input, output)
        info(f"fastq -> fasta conversion done: {output}")


# -----------------------------
# FASTA stats
@app.command("fasta-stats")
def fasta_stats_cmd(
    input: Path = typer.Option(..., "-i"),
    output_dir: Path = typer.Option(..., "-o"),
    force: bool = typer.Option(False, "--force", help="Recalculate stats even if outputs exist"),
):
    """
    Calculate per-sequence and overall statistics from a fasta file.
    """
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
    input: Path = typer.Option(..., "-i", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o", help="Output directory"),
    kmer_length: int = typer.Option(6, "--kmer", help="K-mer length"),
    threads: int = typer.Option(1, "-t", help="Number of threads for counting"),
    force: bool = typer.Option(False, "--force", help="Recalculate even if output exists")
):
    """
    Count k-mers for a fasta file and save tsv matrix.
    """
    context = Context(input, output_dir, force=force)

    operation = KmerCount(kmer_length=kmer_length, threads=threads)
    operation.run(context)

    info(f"K-mer counting complete. Matrix saved at: {context.get('kmer_matrix')}")

if __name__ == "__main__":
    app()


# -----------------------------
# kmer-metrics
@app.command("kmer-metrics")
def kmer_metrics_cmd(
    input: Path = typer.Option(..., "-i", help="Input k-mer matrix TSV"),
    output_dir: Path = typer.Option(..., "-o", help="Output directory"),
    chunksize: int = typer.Option(1000, "--chunksize", help="Rows per chunk"),
    cpus: int = typer.Option(1, "--cpus", help="Number of worker processes"),
    force: bool = typer.Option(False, "--force", help="Recompute even if output exists"),
):
    """
    Compute per-sequence k-mer metrics (Shannon diversity, unique k-mers, etc.).
    """
    context = Context(input, output_dir, force=force)
    operation = KmerMetrics(chunksize=chunksize, cpus=cpus)
    operation.run(context)
    info(f"K-mer metrics saved at: {context.get('kmer_metrics')}")


# -----------------------------
# DR
@app.command("dr")
def dr_cmd(
    input: Path = typer.Option(..., "-i", help="Input k-mer matrix tsv"),
    output_dir: Path = typer.Option(..., "-o", help="Output directory"),
    methods: str = typer.Option(..., "--methods", help="Comma-separated DR methods"),
    normalisation: str = typer.Option("clr", "--norm", help="Normalization method"),
    dims: int = typer.Option(2, "--dims", help="Embedding dimensions"),
    force: bool = typer.Option(False, "--force", help="Recompute even if output exists"),
    pca_pre: bool = typer.Option(False, "--pca-pre", help="Apply PCA before DR"),
    keep_pcs: int = typer.Option(None, "--keep-pcs"),
    keep_variance: float = typer.Option(None, "--keep-variance"),
    screen_params: bool = typer.Option(False, "--screen_params", help="Run parameter screening for supported DR methods"),
):
    """
    Run dimensionality reduction on an existing k-mer matrix.
    """

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
    )

    operation.run(context)

    info(f"DR embeddings saved at: {context.get('dr_embeddings')}")


@app.command("run-tiara")
def run_tiara_cmd(
    input: Path = typer.Option(..., "-i", help="Input fasta file"),
    output_dir: Path = typer.Option(..., "-o", help="Output directory"),
    threads: int = typer.Option(1, "-t", help="Number of threads"),
    force: bool = typer.Option(False, "--force", help="Recompute even if output exists"),
):
    """
    Run Tiara classification on a fasta file.
    """
    context = Context(input, output_dir, force=force)

    operation = Tiara(threads=threads)
    operation.run(context)

    info(f"Tiara output saved at: {context.get('tiara')}")


@app.command("build-db")
def build_database(
    input: Path = typer.Option(..., "-i"),
    output_dir: Path = typer.Option(..., "-o"),
    force: bool = typer.Option(False, "--force"),):
    """
    Build Spatialite database from available artifacts.
    """
    context = Context(input, output_dir, force=force)

    operation = SpatialiteDatabase()
    operation.run(context)

    info(f"Database created at: {context.get('database')}")