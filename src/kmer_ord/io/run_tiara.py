#src/kmer_ord/io/run_tiara.py
from pathlib import Path
import subprocess
import pandas as pd
from kmer_ord.system.env_manager import run_in_env, TIARA_ENV
from kmer_ord.utils.logging_utils import section

def run_tiara(input_file, output_file, threads=1, script_name="tiara"):
    """
    Run Tiara inside its dedicated conda environment.
    """
    section("Running Tiara on reads...")
    input_file = Path(input_file)
    output_file = Path(output_file)

    cmd = ["tiara",
           "-i", str(input_file),
           "-o", str(output_file),
           "-t", str(threads),
           "-v"]

    # Run inside TIARA_ENV
    run_in_env(TIARA_ENV, cmd, check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    # Tiara writes the full FASTA header (e.g. "read_1 length=... origin=...")
    # into its sequence_id column, but every other kmer-ord table truncates to
    # the first whitespace token (io/summary.py). Left unfixed, the join in
    # FeatureMerge never matches and Tiara columns end up all-null in the
    # database. Normalise here, at the source, so every consumer of this file
    # (FeatureMerge, `run-tiara` standalone output, manual inspection) sees
    # IDs consistent with the rest of the pipeline.
    df = pd.read_csv(output_file, sep="\t")
    df["sequence_id"] = df["sequence_id"].astype(str).str.split().str[0]
    df.to_csv(output_file, sep="\t", index=False)

    return output_file