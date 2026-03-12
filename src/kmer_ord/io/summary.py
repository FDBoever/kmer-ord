# src/kmer_ord/io/summary.py
from pathlib import Path
# from Bio import SeqIO
# import pandas as pd
import statistics
from kmer_ord.workflow.context import Context
from kmer_ord.utils.logging_utils import section, info, warn


def calculate_nx(sequence_lengths, target_percentage=50):
    """
    Calculate Nx statistic.
    Returns (Nx value, largest sequence length).
    """
    sorted_lengths = sorted(sequence_lengths, reverse=True)
    total_length = sum(sorted_lengths)
    target_length = total_length * target_percentage / 100

    cumulative_length = 0
    nx = 0

    for length in sorted_lengths:
        cumulative_length += length
        if cumulative_length >= target_length:
            nx = length
            break

    return nx, sorted_lengths[0]


def calculate_stats(context: Context):
    """
    Calculate per-sequence and overall stats for the canonical FASTA in context.
    Saves files to structured subdirectories.
    
    Returns:
        df : pd.DataFrame
        overall_file : Path
        tsv_file : Path
    """
    section("Calculating per-sequence statistics...")
    from Bio import SeqIO
    import pandas as pd
    fasta_file = context.get("fasta")

    sequences = {r.id: str(r.seq) for r in SeqIO.parse(str(fasta_file), "fasta")}

    results_dict = {"sequence_id": [],
                    "Length": [],
                    "GC_Content": [],
                    "AT_Content": []}

    total_length = 0
    gc_contents = []

    for seq_id, seq in sequences.items():
        length = len(seq)
        gc = (seq.count("G") + seq.count("C")) / length * 100 if length > 0 else 0
        at = (seq.count("A") + seq.count("T")) / length * 100 if length > 0 else 0

        results_dict["sequence_id"].append(seq_id)
        results_dict["Length"].append(length)
        results_dict["GC_Content"].append(gc)
        results_dict["AT_Content"].append(at)

        total_length += length
        gc_contents.append(gc)

    # check array lengths
    array_lengths = set(len(arr) for arr in results_dict.values())
    if len(array_lengths) > 1:
        raise ValueError("All arrays must be of the same length")

    df = pd.DataFrame(results_dict)

    total_seqs = len(sequences)
    n50, n90 = calculate_nx(df["Length"])
    avg_gc = statistics.mean(gc_contents)
    std_gc = statistics.stdev(gc_contents) if len(gc_contents) > 1 else 0

    overall_file = context.artifact_path("overall_stats", subdir="summary", suffix=".txt")
    tsv_file = context.artifact_path("stats_per_sequence", subdir="summary", suffix=".tsv")

    # save overall stats
    with overall_file.open("w") as f:
        f.write(f"Total nr of sequences: {total_seqs}\n")
        f.write(f"Total Length: {total_length} bp\n")
        f.write(f"N50: {n50} bp\n")
        f.write(f"N90: {n90} bp\n")
        f.write(f"Average GC Content: {avg_gc:.2f}%\n")
        f.write(f"GC Content Standard Deviation: {std_gc:.2f}\n")

    # save per-sequence stats
    tsv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_file, sep="\t", index=False)

    # Register files
    context.register("summary_overall", overall_file)
    context.register("summary_per_sequence", tsv_file)

    return df, overall_file, tsv_file