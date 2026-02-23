from pathlib import Path
from kmer_ord.io.summary import calculate_stats

def test_calculate_stats(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">seq1\nACTG\n>seq2\nGGGCCC\n")

    df, overall_file, tsv_file = calculate_stats(fasta, tmp_path)

    # Verify outputs
    assert df.shape[0] == 2
    assert overall_file.exists()
    assert tsv_file.exists()
    content = tsv_file.read_text()
    assert "seq1" in content
    assert "seq2" in content