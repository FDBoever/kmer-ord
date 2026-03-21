# src/kmer_ord/cli/setup.py
from kmer_ord.utils.logging_utils import section, info, warn
import typer


setup_app = typer.Typer(help="Setup external dependencies for kmerord.")


@setup_app.command("setup", rich_help_panel="Installation")
def setup(
    force: bool = typer.Option(False, "-f","--force", help="Recreate environments even if they already exist.")):
    """
    Setup all required external dependencies:
    - tools environment (infernal, barrnap, rust)
    - tiara environment
    - kmer-counter Rust installation from GitHub
    """
    from kmer_ord.system.env_manager import (TOOLS_ENV, TIARA_ENV, RDNA_ENV,
                                             env_exists, create_tools_env, create_rdna_env,
                                             create_tiara_env, install_rust_tool, install_rdna_miner,
                                             check_tool)

    print("-" * 70)
    section("Starting kmerord setup...")

    # Tools environment
    if env_exists(TOOLS_ENV):
        if force:
            info(f"{TOOLS_ENV} exists. Recreating (--force enabled)...")
            create_tools_env(TOOLS_ENV, recreate=True)
        else:
            info(f"{TOOLS_ENV} already exists. Skipping.")
    else:
        info(f"Creating {TOOLS_ENV}...")
        create_tools_env(TOOLS_ENV)
        info(f"{TOOLS_ENV} created.")

    # Tiara environment
    if env_exists(TIARA_ENV):
        if force:
            info(f"{TIARA_ENV} exists. Recreating (--force enabled)...")
            create_tiara_env(TIARA_ENV, recreate=True)
        else:
            info(f"{TIARA_ENV} already exists. Skipping.")
    else:
        info(f"Creating {TIARA_ENV}...")
        create_tiara_env(TIARA_ENV)
        info(f"{TIARA_ENV} created.")

    # Rust GitHub tool
    section("Installing Rust-based tool...")
    install_rust_tool()
    info("Rust tool installation complete.")

    # rDNA-miner environment
    section("Setting up rDNA-miner...")

    if env_exists(RDNA_ENV):
        if force:
            info(f"{RDNA_ENV} exists. Recreating (--force enabled)...")
            repo_path = create_rdna_env(RDNA_ENV, recreate=True)
            install_rdna_miner(RDNA_ENV, repo_path)
        else:
            info(f"{RDNA_ENV} already exists. Skipping.")
    else:
        info(f"Creating {RDNA_ENV}...")
        repo_path = create_rdna_env(RDNA_ENV)
        install_rdna_miner(RDNA_ENV, repo_path)
        info(f"{RDNA_ENV} created.")

    # Verify installed tools
    section("Verifying external tools...")

    tools_to_check = [
        ("kmer-counter", TOOLS_ENV, ["--help"]),
        ("tiara", TIARA_ENV, ["--help"]),
        ("barrnap", RDNA_ENV, ["--help"]),
        ("cmscan", RDNA_ENV, ["--help"]),
        ("flye", RDNA_ENV, ["--help"]),
        ("minimap2", RDNA_ENV, ["--help"]),
        ("samtools", RDNA_ENV, ["--help"]),
        ("Rscript", RDNA_ENV, ["--version"]),
    ]
    success = True

    success = True
    for tool, env in tools_to_check:
        if not check_tool(tool, env=env):
            success = False

    if success:
        print("-" * 70)
        info("\nAll tools verified successfully")
        info("\nSetup complete. kmerord is ready to use.")

    else:
        print("-" * 70)
        warn("Some tools failed verification. Please check installation or PATH.")
