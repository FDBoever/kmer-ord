from pathlib import Path
from typing import Union
from kmer_ord.utils.logging_utils import section, info, warn


def fastq_to_fasta(input_path: Path, output_path: Path) -> None:
    from Bio import SeqIO
    import gzip
    info("converting fastq to fasta (biopython)...")
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    open_func = gzip.open if input_path.suffix == ".gz" else open
    with open_func(input_path, "rt") as infile, output_path.open("w") as fasta_out:
        for record in SeqIO.parse(infile, "fastq"):
            SeqIO.write(record, fasta_out, "fasta")


def fastq_to_fasta_seqkit(input_path: Path, output_path: Path, threads: int = 1) -> None:
    """Convert FASTQ (plain or gzipped) to FASTA using seqkit.

    Requires seqkit to be installed and available on PATH.
    Faster than the BioPython path and can use multiple threads.
    """
    import subprocess
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info(f"converting fastq to fasta (seqkit, threads={threads})...")
    cmd = ["seqkit", "fq2fa", str(input_path), "-o", str(output_path), "-j", str(threads)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"seqkit fq2fa failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )


def load_fasta_or_convert(
    input_file: Union[str, Path],
    work_dir: Path,
    force: bool = False,
    use_seqkit: bool = True,
    threads: int = 1,
) -> Path:
    """Handle FASTA/FASTQ (plain or gzipped) and return path to a usable FASTA.

    Skips conversion/copy when the output already exists unless force=True.
    FASTQ conversion uses seqkit by default; pass use_seqkit=False for BioPython.
    """
    import gzip
    section("Loading fasta/fastq (or gzipped)")
    input_file = Path(input_file)
    suffix = input_file.suffix.lower()
    stem = input_file.stem

    # If gzipped, check inner suffix
    if suffix == ".gz":
        inner_suffix = Path(stem).suffix.lower()
    else:
        inner_suffix = suffix

    fasta_path = work_dir / f"{stem}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)

    if fasta_path.exists() and not force:
        info(f"Skipping FASTA preparation, output already exists: {fasta_path}")
        return fasta_path

    if inner_suffix in (".fasta", ".fa"):
        if suffix == ".gz":
            import shutil
            with gzip.open(input_file, "rt") as f_in, fasta_path.open("w") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            import shutil
            shutil.copy(input_file, fasta_path)

    elif inner_suffix in (".fastq", ".fq"):
        if use_seqkit:
            fastq_to_fasta_seqkit(input_file, fasta_path, threads=threads)
        else:
            fastq_to_fasta(input_file, fasta_path)
    else:
        raise ValueError(f"Unsupported file type: {input_file}")

    return fasta_path