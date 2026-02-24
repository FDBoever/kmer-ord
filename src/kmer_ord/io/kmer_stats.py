import os
import math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from kmer_ord.utils.logging_utils import section, info, warn

def build_dtypes(input_file: str) -> dict:
    """Infer dtypes for chunked reading of k-mer frequency matrix."""
    with open(input_file, "r") as f:
        column_names = f.readline().strip().split("\t")
        num_columns = len(column_names)

    dtypes = {0: "str"}  # index column
    for col in range(1, num_columns - 1):
        dtypes[col] = "uint16"
    return dtypes

def calculate_kmer_metrics_chunk(kmer_df: pd.DataFrame) -> pd.DataFrame:
    """Compute k-mer metrics for a chunk of sequences."""
    numeric = kmer_df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("No numeric k-mer columns found.")

    numeric = numeric.dropna(axis=1, how="all").fillna(0)
    values = numeric.to_numpy(dtype=float)

    nonzero_mask = values != 0
    total_nonzero = nonzero_mask.sum(axis=1)
    row_sums = values.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    probs = values / row_sums_safe[:, None]
    positive = probs > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        shannon_nats = -np.sum(np.where(positive, probs * np.log(probs), 0.0), axis=1)
        shannon_bits = -np.sum(np.where(positive, probs * np.log2(probs), 0.0), axis=1)

    metrics_chunk = pd.DataFrame(
        {
            "total_nonzero_kmers": total_nonzero,
            "num_unique_kmers": total_nonzero,
            "shannon_evenness": shannon_nats,
            "shannon_diversity": shannon_bits,
        },
        index=kmer_df.index,
    )
    return metrics_chunk

def process_kmer_file(
    input_file: str,
    output_file: str = None,
    chunksize: int = 1000,
    cpus: int = 1,
    total_rows: int = None,) -> pd.DataFrame:
    """Process a k-mer matrix to compute metrics, optionally parallelized."""
    section("Calculating k-mer metrics")
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if os.path.exists(output_file):
            os.remove(output_file)

    dtypes = build_dtypes(input_file)
    reader = pd.read_csv(input_file, sep="\t", index_col=0, dtype=dtypes, chunksize=chunksize)
    
    all_chunks = []

    def write_chunk(metrics_chunk: pd.DataFrame, first: bool = False):
        if output_file:
            metrics_chunk.to_csv(
                output_file,
                sep="\t",
                mode="w" if first else "a",
                header=first
            )

    first_chunk = True
    processed_rows = 0
    approx_total_chunks = math.ceil(total_rows / chunksize) if total_rows else None

    if cpus <= 1:
        # Sequential
        for chunk_id, chunk in enumerate(reader, start=1):
            metrics_chunk = calculate_kmer_metrics_chunk(chunk)
            write_chunk(metrics_chunk, first_chunk)
            first_chunk = False
            processed_rows += len(chunk)
            all_chunks.append(metrics_chunk)
            if total_rows:
                pct = processed_rows / total_rows * 100
                #print(f"[INFO] Chunk {chunk_id} processed ({pct:.2f}%)")
    else:
        # Parallel
        in_flight = []
        with ProcessPoolExecutor(max_workers=cpus) as ex:
            for chunk_id, chunk in enumerate(reader, start=1):
                fut = ex.submit(calculate_kmer_metrics_chunk, chunk)
                in_flight.append((fut, chunk_id, len(chunk)))
                if len(in_flight) >= cpus:
                    fut_done, cid, rows = in_flight.pop(0)
                    metrics_chunk = fut_done.result()
                    write_chunk(metrics_chunk, first_chunk)
                    first_chunk = False
                    processed_rows += rows
                    all_chunks.append(metrics_chunk)
                    if total_rows:
                        pct = processed_rows / total_rows * 100
                        #print(f"[INFO] Chunk {cid} processed ({pct:.2f}%)")
            # Drain remaining
            for fut_done, cid, rows in in_flight:
                metrics_chunk = fut_done.result()
                write_chunk(metrics_chunk, first_chunk)
                first_chunk = False
                processed_rows += rows
                all_chunks.append(metrics_chunk)
                if total_rows:
                    pct = processed_rows / total_rows * 100
                    #print(f"[INFO] Chunk {cid} processed ({pct:.2f}%)")
    
    # Combine all chunks into a single DataFrame for pipeline
    combined_metrics = pd.concat(all_chunks)
    return combined_metrics