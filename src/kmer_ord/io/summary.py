# src/kmer_ord/io/summary.py
from pathlib import Path
from kmer_ord.utils.logging_utils import section, info, warn


def calculate_nx(sequence_lengths, target_percentage=50):
    """Calculate Nx statistic. Returns (Nx value, largest sequence length)."""
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


def _find_chunk_starts(fasta_path: Path, n_chunks: int) -> list:
    """
    Return byte offsets of record-boundary starts that divide the file into
    at most n_chunks pieces. Each returned position points to a '>' character.
    Always returns at least [0].
    """
    file_size = fasta_path.stat().st_size
    if file_size == 0 or n_chunks <= 1:
        return [0]

    chunk_size = file_size // n_chunks
    starts = [0]

    with open(fasta_path, "rb") as f:
        for i in range(1, n_chunks):
            target = i * chunk_size
            if target >= file_size:
                break
            f.seek(target)
            f.readline()  # finish the current partial line
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if pos not in starts:
                        starts.append(pos)
                    break

    return starts


def _process_fasta_chunk(args: tuple) -> list:
    """
    Worker: reads bytes [start, end) from a FASTA file, parses all complete records,
    and returns a list of (seq_id, length, gc_pct, at_pct) tuples.
    Defined at module level so it is picklable for multiprocessing spawn on macOS.
    """
    from Bio.SeqIO.FastaIO import SimpleFastaParser
    import io

    fasta_path, start, end = args

    with open(fasta_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start) if end is not None else f.read()

    results = []
    for title, seq in SimpleFastaParser(io.StringIO(data.decode("ascii", errors="replace"))):
        seq_id = title.split()[0]
        length = len(seq)
        if length == 0:
            gc = at = 0.0
        else:
            s = seq.upper()
            gc = (s.count("G") + s.count("C")) / length * 100
            at = (s.count("A") + s.count("T")) / length * 100
        results.append((seq_id, length, gc, at))

    return results


def calculate_stats(context):
    """
    Calculate per-sequence and overall stats for the canonical FASTA in context.
    Uses parallel chunked processing when context.threads > 1.
    Saves files to structured subdirectories under context.output_dir.

    Returns (df, overall_file, tsv_file).
    """
    section("Calculating per-sequence statistics...")
    import pandas as pd
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    fasta_file = context.get("fasta")
    threads = getattr(context, "threads", 1)

    chunk_starts = _find_chunk_starts(fasta_file, n_chunks=threads)
    chunk_ends = chunk_starts[1:] + [None]
    tasks = list(zip([fasta_file] * len(chunk_starts), chunk_starts, chunk_ends))

    # Process chunks — skip executor overhead for single-chunk case
    all_records: list = []
    if len(tasks) == 1:
        all_records = _process_fasta_chunk(tasks[0])
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(_process_fasta_chunk, task) for task in tasks]
            for future in futures:  # iterate in submission order to preserve sequence order
                all_records.extend(future.result())

    if not all_records:
        raise RuntimeError(f"No sequences found in {fasta_file}")

    seq_ids, lengths, gc_contents, at_contents = zip(*all_records)
    lengths_arr = np.array(lengths, dtype=np.int64)
    gc_arr = np.array(gc_contents, dtype=np.float64)
    at_arr = np.array(at_contents, dtype=np.float64)

    df = pd.DataFrame({
        "sequence_id": list(seq_ids),
        "Length": lengths_arr,
        "GC_Content": gc_arr,
        "AT_Content": at_arr,
    })

    total_seqs = len(df)
    total_length = int(lengths_arr.sum())
    n50, _ = calculate_nx(lengths, target_percentage=50)
    n90, _ = calculate_nx(lengths, target_percentage=90)
    avg_gc = float(gc_arr.mean())
    std_gc = float(gc_arr.std(ddof=1)) if total_seqs > 1 else 0.0

    overall_file = context.artifact_path("overall_stats", subdir="summary", suffix=".txt")
    tsv_file = context.artifact_path("stats_per_sequence", subdir="summary", suffix=".tsv")

    with overall_file.open("w") as f:
        f.write(f"Total nr of sequences: {total_seqs}\n")
        f.write(f"Total Length: {total_length} bp\n")
        f.write(f"N50: {n50} bp\n")
        f.write(f"N90: {n90} bp\n")
        f.write(f"Average GC Content: {avg_gc:.2f}%\n")
        f.write(f"GC Content Standard Deviation: {std_gc:.2f}\n")

    w = 20
    info(f"{'sequences':<{w}}  {total_seqs:>14,}")
    info(f"{'total length':<{w}}  {total_length:>11,} bp")
    info(f"{'N50':<{w}}  {n50:>11,} bp")
    info(f"{'N90':<{w}}  {n90:>11,} bp")
    info(f"{'average GC':<{w}}  {avg_gc:>13.2f}%")
    info(f"{'GC std dev':<{w}}  {std_gc:>14.2f}")

    tsv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_file, sep="\t", index=False)

    context.register("summary_overall", overall_file)
    context.register("summary_per_sequence", tsv_file)

    return df, overall_file, tsv_file
