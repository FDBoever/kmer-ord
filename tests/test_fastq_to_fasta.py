from pathlib import Path
from kmer_ord.io.sequence import fastq_to_fasta


def test_fastq_to_fasta(tmp_path):
    fastq = tmp_path / "test.fastq"
    fasta = tmp_path / "test.fasta"

    fastq.write_text(
        "@seq1\nACTG\n+\n!!!!\n"
    )

    fastq_to_fasta(fastq, fasta)

    assert fasta.exists()
    content = fasta.read_text()
    assert ">seq1" in content