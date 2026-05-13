#src/kmer_ord/io/run_tiara.py
from pathlib import Path
import subprocess
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

    # Tiara writes full FASTA headers as sequence IDs; strip to the token
    # before the first space to match the rest of the pipeline.
    import pandas as pd
    df = pd.read_csv(output_file, sep="\t")
    df.iloc[:, 0] = df.iloc[:, 0].str.split().str[0]
    df.to_csv(output_file, sep="\t", index=False)

    return output_file