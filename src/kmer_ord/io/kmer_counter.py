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


def get_embedded_kmer_counter_path() -> Path:
    """
    Return the correct embedded kmer-counter binary
    depending on the current operating system.
    """

    base_dir = Path(__file__).parent.parent  # src/kmer_ord
    bin_dir = base_dir / "data" / "bin"

    system = platform.system()

    if system == "Darwin":
        binary_name = "kmer-counter-osx"
    elif system == "Linux":
        binary_name = "kmer-counter-linux"
    else:
        raise OSError(
            f"Unsupported operating system: {system}. "
            "Embedded kmer-counter is only available for macOS and Linux.")

    binary_path = bin_dir / binary_name

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Expected embedded binary '{binary_name}' not found at {binary_path}."
        )

    if not binary_path.is_file():
        raise OSError(f"Embedded binary path is not a file: {binary_path}")

    return binary_path.resolve()


def run_kmer_counter(input_file, output_tsv, kmer_length, num_threads,
                     kmer_counter_path=None, script_name="kmer-counter"):
    """Run kmer-counter and produce TSV output with per-step benchmarking."""

    # Resolve the binary path
    kmer_counter_path = Path(kmer_counter_path) if kmer_counter_path else get_embedded_kmer_counter_path()
    
    if not kmer_counter_path.exists():
        raise FileNotFoundError(f"kmer-counter not found at {kmer_counter_path}")

    input_basename = Path(input_file).stem
    temp_dir = tempfile.mkdtemp(prefix="kmer_counter_temp_")
    input_args = f"--input {input_file} --output {output_tsv} --kmer {kmer_length} --threads {num_threads}"

    # --- Run kmer-counter ---
    with BenchmarkTimer("Kmer_Counter_Run", script_name=script_name,
                        input_file=input_file, input_args=input_args):
        cmd = [
            str(kmer_counter_path),
            "--file", str(input_file),
            "--ids", str(Path(temp_dir) / "sequence_headers.txt"),
            "--klength", str(kmer_length),
            "--out", str(Path(temp_dir) / "kmer_counts.npy"),
            "--collapse", "1"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # --- Load numpy ---
    npy_file = Path(temp_dir) / "kmer_counts.npy"
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
