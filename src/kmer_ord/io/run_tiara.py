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

    return output_file