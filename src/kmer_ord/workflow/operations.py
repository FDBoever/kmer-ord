# src/kmer_ord/workflow/operations.py
from pathlib import Path
from kmer_ord.workflow import context
import typer
import pandas as pd
import importlib.resources as pkg_resources

from kmer_ord.io.summary import calculate_stats
from kmer_ord.io.kmer_counter import run_kmer_counter, get_embedded_kmer_counter_path
from .operation import Operation
from kmer_ord.utils.benchmark import BenchmarkTimer  # <-- import BenchmarkTimer
from kmer_ord import data

class FastqToFasta(Operation):
    name = "fastq_to_fasta"
    produces = ["fasta"]

    def run(self, context):
        with BenchmarkTimer(label=self.name, input_file=context.input_file):
            # No need to convert again — context has already created canonical FASTA.
            # Just register it explicitly.
            fasta = context.fasta
            context.register("fasta", fasta)


class FastaStats(Operation):
    name = "fasta_stats"
    requires = ["fasta"]
    produces = ["summary_overall", "summary_per_sequence"]

    def run(self, context):
        overall_path = context.artifact_path("overall_stats", subdir="summary", suffix=".txt")
        per_seq_path = context.artifact_path("stats_per_sequence", subdir="summary", suffix=".tsv")

        with BenchmarkTimer(label=self.name, input_file=context.fasta):
            if overall_path.exists() and per_seq_path.exists() and not context.force:
                context.logger.info("Skipping FastaStats, summary output files already exist.")
                typer.echo("Skipping FastaStats, summary output files already exist.")
            else:
                df, overall_file, tsv_file = calculate_stats(context)

        # Register artifacts even if skipping
        context.register("summary_overall", overall_path)
        context.register("summary_per_sequence", per_seq_path)

class KmerCount(Operation):
    name = "kmer_count"
    requires = ["fasta"]
    produces = ["kmer_matrix"]

    def __init__(self, kmer_length, threads=1, kmer_counter_path=None):
        self.kmer_length = kmer_length
        self.threads = threads
        # If no path is given, use embedded binary
        self.kmer_counter_path = kmer_counter_path or get_embedded_kmer_counter_path()

    def run(self, context):
        output_path = context.artifact_path(f"{self.kmer_length}mer_matrix",
                                            subdir="kmer", suffix=".tsv")

        with BenchmarkTimer(label=f"{self.name}_{self.kmer_length}mer",
                            input_file=context.get("fasta"),
                            input_args=f"kmer_length={self.kmer_length}, threads={self.threads}"):
            if output_path.exists() and not context.force:
                typer.echo(f"Skipping KmerCount, matrix already exists: {output_path}")
                context.logger.info(f"Skipping KmerCount, matrix already exists: {output_path}")
            else:
                run_kmer_counter(
                    input_file=context.get("fasta"),
                    output_tsv=output_path,
                    kmer_length=self.kmer_length,
                    num_threads=self.threads,
                    kmer_counter_path=self.kmer_counter_path
                )

        context.register("kmer_matrix", output_path)




# -----------------------------
# DR

from kmer_ord.dr.loader import load_matrix
from kmer_ord.dr.preprocess import preprocess_data, reduce_dimensions_with_pca
from kmer_ord.dr.methods import run_dr_methods

class DimensionalityReduction(Operation):
    name = "dimensionality_reduction"
    requires = ["kmer_matrix"]
    produces = ["dr_embeddings"]

    def __init__(
        self,
        methods,
        normalisations=("clr",),
        dims=2,
        pca_dim_red=False,
        keep_pcs=None,
        keep_variance=None,
        scale="auto",
        seed=42,
        screen_params=False,
        max_memory_gb=None
    ):
        self.methods = methods
        self.normalisations = normalisations
        self.dims = dims
        self.pca_dim_red = pca_dim_red
        self.keep_pcs = keep_pcs
        self.keep_variance = keep_variance
        self.scale = scale
        self.seed = seed
        self.screen_params = screen_params
        self.max_memory_gb = max_memory_gb

    def run(self, context):

        matrix_path = context.get("kmer_matrix")
        dr_dir = context.output_dir / "dr"
        dr_dir.mkdir(parents=True, exist_ok=True)

        matrix = load_matrix(matrix_path)

        # Expand normalisations if needed
        if "all" in self.normalisations:
            normalisations = ALL_NORMALISATIONS
        else:
            normalisations = self.normalisations

        merged_outputs = []

        with BenchmarkTimer(label=self.name, input_file=matrix_path):

            # Memory safety check (only once)
            base_mem_gb = matrix.memory_usage(deep=True).sum() / (1024 ** 3)
            est_peak = base_mem_gb * 4.0

            if self.max_memory_gb and est_peak > self.max_memory_gb:
                raise MemoryError(
                    f"Estimated peak {est_peak:.2f} GB exceeds "
                    f"limit {self.max_memory_gb:.2f} GB"
                )

            for norm in normalisations:

                context.logger.info(f"Applying normalisation: {norm}")

                merged_output = (
                    dr_dir /
                    f"{matrix_path.stem}_{norm}_merged_embeddings.tsv"
                )

                # Skip per-normalisation if exists
                if merged_output.exists() and not context.force:
                    context.logger.info(
                        f"Skipping DR for '{norm}', merged file exists."
                    )
                    merged_outputs.append(merged_output)
                    continue

                X = preprocess_data(matrix, norm)

                if self.pca_dim_red:
                    X = reduce_dimensions_with_pca(
                        X,
                        keep_pcs=self.keep_pcs,
                        keep_variance=self.keep_variance
                    )

                # This now merges only across methods
                merged_file = run_dr_methods(
                    X=X,
                    methods=self.methods,
                    dims=self.dims,
                    seed=self.seed,
                    scale=self.scale,
                    screen_params=self.screen_params,
                    output_dir=dr_dir,
                    normalisation=norm,
                    input_name=matrix_path.stem
                )

                merged_outputs.append(merged_file)

        # Register ALL per-normalisation merged outputs
        context.register("dr_embeddings", merged_outputs)

from kmer_ord.io.kmer_stats import process_kmer_file

class KmerMetrics(Operation):
    name = "kmer_metrics"
    requires = ["kmer_matrix"]
    produces = ["kmer_metrics"]

    def __init__(self, chunksize=1000, cpus=1):
        self.chunksize = chunksize
        self.cpus = cpus

    def run(self, context):
        matrix_path = context.get("kmer_matrix")

        output_file = context.artifact_path(name="kmer_metrics", subdir="kmer", suffix=".tsv")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with BenchmarkTimer(label=self.name, input_file=matrix_path):
            if output_file.exists() and not context.force:
                typer.echo(f"Skipping KmerMetrics, output exists: {output_file}")
                context.logger.info(f"Skipping KmerMetrics, output exists: {output_file}")
            else:
                metrics_df = process_kmer_file(
                    input_file=matrix_path,
                    output_file=output_file,
                    chunksize=self.chunksize,
                    cpus=self.cpus
                )

        context.register("kmer_metrics", output_file)