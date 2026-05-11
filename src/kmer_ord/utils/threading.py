def set_global_threads(threads: int):
    import os

    # Set env vars BEFORE importing numba so it initialises its thread pool
    # at the correct count rather than defaulting to cpu_count().
    for v in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]:
        os.environ[v] = str(threads)

    import numba
    numba.set_num_threads(threads)
