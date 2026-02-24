import gzip
from pathlib import Path
from Bio import SeqIO
from typing import Union
from kmer_ord.utils.logging_utils import section, info, warn

def fastq_to_fasta(input_path: Path, output_path: Path) -> None:
    info("converting fasta to fastq...")
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    open_func = gzip.open if input_path.suffix == ".gz" else open
    with open_func(input_path, "rt") as infile, output_path.open("w") as fasta_out:
        for record in SeqIO.parse(infile, "fastq"):
            SeqIO.write(record, fasta_out, "fasta")

def load_fasta_or_convert(input_file: Union[str, Path], work_dir: Path) -> Path:
    """
    Handle FASTA/FASTQ (plain or gzipped) and return path to a usable FASTA.
    """
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

    if inner_suffix in (".fasta", ".fa"):
        # Already FASTA: if gzipped, unzip
        if suffix == ".gz":
            import shutil
            with gzip.open(input_file, "rt") as f_in, fasta_path.open("w") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            # just copy to working dir
            import shutil
            shutil.copy(input_file, fasta_path)

    elif inner_suffix in (".fastq", ".fq"):
        # Convert FASTQ → FASTA
        fastq_to_fasta(input_file, fasta_path)
    else:
        raise ValueError(f"Unsupported file type: {input_file}")

    return fasta_path