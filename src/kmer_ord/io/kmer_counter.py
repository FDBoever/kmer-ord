# src/kmer_ord/io/kmer_counter.py
import os
import subprocess
import shutil
import tempfile
import time
import numpy as np
from Bio import SeqIO
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from kmer_ord.utils.benchmark import BenchmarkTimer
from pathlib import Path
from kmer_ord import data
import platform

from kmer_ord.system.env_manager import TOOLS_ENV, run_in_env

def format_size(size_in_bytes):
    return f"{size_in_bytes / (1024*1024):,.2f} MB".replace(",", ".")

def canonical_kmers(k):
    acgt = ['A', 'C', 'G', 'T']
    rev_comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    product_kmers = [''.join(p) for p in product(acgt, repeat=k)]
    canon_set = set()
    for kmer in product_kmers:
        rev = ''.join(rev_comp[base] for base in reversed(kmer))
        canon_set.add(min(kmer, rev))
    return sorted(list(canon_set))


def run_kmer_counter(input_file, output_tsv, kmer_length, num_threads,
                     script_name="kmer-counter"):
    """
    Run kmer-counter from TOOLS_ENV and produce TSV output with per-step benchmarking.
    """

    temp_dir = Path(tempfile.mkdtemp(prefix="kmer_counter_temp_"))
    input_args = f"--input {input_file} --kmer {kmer_length} --threads {num_threads}"

    # --- Run kmer-counter in TOOLS_ENV ---
    with BenchmarkTimer("Kmer_Counter_Run", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        cmd = [
            "kmer-counter",
            "--file", str(input_file),
            "--ids", str(temp_dir / "sequence_headers.txt"),
            "--klength", str(kmer_length),
            "--out", str(temp_dir / "kmer_counts.npy"),
            "--collapse", "1"
        ]
        # run inside the tools environment
        run_in_env(TOOLS_ENV, cmd)

    # --- Load numpy ---
    npy_file = temp_dir / "kmer_counts.npy"
    with BenchmarkTimer("Numpy_Loading", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        kmer_data = np.load(npy_file, mmap_mode='r').astype(np.uint32)
        print(f"-- npy loaded uint32: {kmer_data.shape} {format_size(kmer_data.nbytes)}")

    # --- Extract sequence headers ---
    with BenchmarkTimer("Sequence_Headers_Extraction", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            sequence_headers = list(executor.map(lambda r: r.id, SeqIO.parse(input_file, "fasta")))

    # --- Generate canonical k-mers ---
    with BenchmarkTimer("Canonical_Kmers_Generation", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        kmer_keys = canonical_kmers(kmer_length)

    # --- Compose output TSV ---
    with BenchmarkTimer("TSV_Composition", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        os.makedirs(Path(output_tsv).parent, exist_ok=True)
        with open(output_tsv, "w") as f:
            f.write("Sequence_ID\t" + "\t".join(kmer_keys) + "\n")
            for i in range(len(kmer_data)):
                f.write(sequence_headers[i] + "\t" + "\t".join(map(str, kmer_data[i])) + "\n")

    # --- Cleanup ---
    with BenchmarkTimer("Cleanup", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        shutil.rmtree(temp_dir)

    print(f"-- Output saved: {output_tsv}")
    return output_tsv
