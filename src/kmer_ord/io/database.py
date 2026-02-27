from os import name
from pathlib import Path
import sqlite3
from sqlite3 import OperationalError
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqIO.QualityIO import FastqGeneralIterator


# ---------------------------------------------------------
# INITIALIZE DATABASE
# ---------------------------------------------------------

def initialize_spatialite_db(db_file: Path):
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    try:
        conn.enable_load_extension(True)
        cursor.execute("SELECT load_extension('mod_spatialite');")
        cursor.execute("SELECT InitSpatialMetaData(1);")
    except OperationalError as e:
        conn.close()
        raise RuntimeError(f"Error initializing SpatiaLite: {e}")

    # Speed PRAGMAs
    cursor.execute("PRAGMA journal_mode = MEMORY;")
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA temp_store = MEMORY;")
    cursor.execute("PRAGMA cache_size = -500000;")

    conn.commit()
    return conn


# ---------------------------------------------------------
# FASTA TABLE
# ---------------------------------------------------------

def create_fasta_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fasta (
            header TEXT PRIMARY KEY,
            full_header TEXT,
            sequence TEXT,
            qualities TEXT);
        """)
    conn.commit()


def populate_fasta_table(conn, fasta_file: Path, batch_size=50000):
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO fasta (header, full_header, sequence, qualities)
        VALUES (?, ?, ?, ?);
    """

    conn.execute("BEGIN TRANSACTION;")
    batch = []

    is_fastq = fasta_file.suffix in [".fastq", ".fq"]

    with open(fasta_file) as handle:
        if is_fastq:
            for title, seq, qual in FastqGeneralIterator(handle):
                short_id = title.split()[0]
                batch.append((short_id, title, seq, qual))
                if len(batch) >= batch_size:
                    cursor.executemany(insert_sql, batch)
                    batch.clear()
        else:
            for title, seq in SimpleFastaParser(handle):
                short_id = title.split()[0]
                batch.append((short_id, title, seq, None))
                if len(batch) >= batch_size:
                    cursor.executemany(insert_sql, batch)
                    batch.clear()

    if batch:
        cursor.executemany(insert_sql, batch)

    conn.commit()


# ---------------------------------------------------------
# FEATURES TABLE (auto-merged artifacts)
# ---------------------------------------------------------

def create_features_table(conn, df: pd.DataFrame):
    cursor = conn.cursor()

    create_sql = "CREATE TABLE features (sequence_id TEXT PRIMARY KEY"

    for col in df.columns[1:]:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            sqltype = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sqltype = "REAL"
        else:
            sqltype = "TEXT"
        create_sql += f", {col} {sqltype}"

    create_sql += ");"
    cursor.execute(create_sql)
    conn.commit()


def populate_features_table(conn, df: pd.DataFrame):
    cursor = conn.cursor()

    columns = ", ".join(df.columns)
    placeholders = ", ".join(["?"] * len(df.columns))
    insert_sql = f"INSERT INTO features ({columns}) VALUES ({placeholders});"

    conn.execute("BEGIN TRANSACTION;")
    cursor.executemany(insert_sql, df.to_numpy().tolist())
    conn.commit()


# ---------------------------------------------------------
# COORDINATES TABLE (from DR embeddings)
# ---------------------------------------------------------

def create_coordinates_table(conn, df: pd.DataFrame):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE coordinates (
            header TEXT PRIMARY KEY,
            FOREIGN KEY (header) REFERENCES features (sequence_id)
        );
    """)

    methods = set()

    # detect methods dynamically
    for col in df.columns[1:]:  # skip header column
        base = col.rsplit("_", 1)[0]
        methods.add(base)

    for method in methods:
        # detect dimension count
        dims = 0
        i = 1
        while f"{method}_{i}" in df.columns:
            dims += 1
            i += 1

        if dims not in (2, 3):
            raise ValueError(
                f"{method} must have 2 or 3 dimensions for spatial storage."
            )

        if dims == 2:
            geom_type = "POINT"
            coord_dim = "XY"
        else:
            geom_type = "POINTZ"
            coord_dim = "XYZ"

        cursor.execute(
            f"""
            SELECT AddGeometryColumn(
                'coordinates',
                '{method}',
                0,
                '{geom_type}',
                '{coord_dim}'
            );
            """
        )

    conn.commit()
    return methods


def populate_coordinates_table(conn, df: pd.DataFrame, methods):
    cursor = conn.cursor()
    conn.execute("BEGIN TRANSACTION;")

    for _, row in df.iterrows():
        header = row.iloc[0]

        columns = ["header"]
        values = [header]
        placeholders = ["?"]

        for method in methods:
            dims = []
            i = 1
            while f"{method}_{i}" in df.columns:
                dims.append(row[f"{method}_{i}"])
                i += 1

            if len(dims) == 2:
                wkt = f"POINT({dims[0]} {dims[1]})"
            elif len(dims) == 3:
                wkt = f"POINT Z({dims[0]} {dims[1]} {dims[2]})"
            else:
                raise ValueError(
                    f"{method} must have 2 or 3 dimensions."
                )

            columns.append(method)
            values.append(wkt)
            placeholders.append("ST_GeomFromText(?, 0)")

        insert_sql = f"""
            INSERT INTO coordinates ({','.join(columns)})
            VALUES ({','.join(placeholders)});
        """

        cursor.execute(insert_sql, values)

    conn.commit()


# -----------------------------
# DR EMBEDDINGS TABLES
# -----------------------------
def save_dr_embeddings(conn, df: pd.DataFrame, method_name: str, force: bool = False):
    table_name = f"embedding_{method_name.lower()}"

    # Check if table exists
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    )
    exists = cursor.fetchone() is not None

    if exists and not force:
        print(f"Skipping DR embeddings for '{method_name}', table already exists: {table_name}")
        return table_name

    if exists and force:
        print(f"Overwriting existing embeddings table '{table_name}'")
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    df.to_sql(table_name, conn, index=False)
    return table_name

# -----------------------------
# CLUSTERING TABLE
# -----------------------------
def save_clustering_results(conn: sqlite3.Connection, cluster_files: list[Path], force: bool = False):
    """
    Merge multiple clustering result files into a single flat table and save as 'Clustering'.
    """
    dfs = [pd.read_csv(f, sep="\t") for f in cluster_files]
    merged_df = pd.concat(dfs, axis=1)

    # Keep only one sequence_id column
    if "sequence_id" in merged_df.columns:
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    table_name = "Clustering"
    if force:
        conn.execute(f"DROP TABLE IF EXISTS {table_name};")

    merged_df.to_sql(table_name, conn, index=False)
    return table_name

# -----------------------------
# WRAPPER TO ADD DR + CLUSTERING TO DB
# -----------------------------
def add_dr_and_clusters_to_db(db_path: Path, embedding_files: list[Path], cluster_files: list[Path], force: bool = False):
    """
    Opens (or creates) DB, adds DR embedding tables and a single Clustering table.
    Returns the sqlite3 connection path.
    """
    conn = sqlite3.connect(str(db_path))

    # Save embeddings
    for emb_file in embedding_files:
        df = pd.read_csv(emb_file, sep="\t")
        # method name extraction from filename
        method_name = Path(emb_file).stem.split("_")[-3]
        save_dr_embeddings(conn, df, method_name, force=force)

    # Save clustering
    save_clustering_results(conn, cluster_files, force=force)

    conn.commit()
    conn.close()
    return db_path

# ---------------------------------------------------------
# DATABASE INSPECTION (DEBUGGING / QA)
# ---------------------------------------------------------

def inspect_database(db_file: Path, limit: int = 5):
    """
    Inspect a SpatiaLite/SQLite database.
    Dynamically lists all tables, including:
    - fasta
    - features
    - coordinates
    - embedding_<method>
    - Clustering
    Shows the first `limit` rows for each table.
    """
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Try enabling SpatiaLite extension
    try:
        conn.enable_load_extension(True)
        cursor.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        pass

    # Query all user tables
    cursor.execute("""SELECT name FROM sqlite_master WHERE type='table';""")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print("No tables found in database.")
        conn.close()
        return
    
    SYSTEM_PREFIXES = ("sqlite_",
                       "idx_",
                       "geometry_",
                       "spatial_ref_",
                       "virts_",
                       "views_")

    SYSTEM_TABLES = {"data_licenses",
                     "sql_statements_log",
                     "SpatialIndex",
                     "spatial_ref_sys",
                     "KNN2",
                     "geometry_columns",
                     "spatialite_history",
                     "ElementaryGeometries"}

    tables = [t for t in tables
              if not t.startswith(SYSTEM_PREFIXES)
              and t not in SYSTEM_TABLES]

    for table in tables:
        print(f"\n--- TABLE: {table} ---")

        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()

        if not schema:
            print("Table not found or empty.")
            continue

        try:
            # Special handling for coordinates (geometry columns)
            if table == "coordinates":
                geom_query = """SELECT f_geometry_column FROM geometry_columns WHERE f_table_name = 'coordinates';"""
                geom_cols = pd.read_sql_query(geom_query, conn)

                if geom_cols.empty:
                    df = pd.read_sql_query(f"SELECT * FROM coordinates LIMIT {limit};", conn)
                    print(df)
                    continue

                geom_names = geom_cols["f_geometry_column"].tolist()
                select_parts = ["header"]
                for col in geom_names:
                    select_parts.append(f"ST_X({col}) AS {col}_1")
                    select_parts.append(f"ST_Y({col}) AS {col}_2")
                    select_parts.append(f"ST_Z({col}) AS {col}_3")

                select_sql = f"""SELECT {', '.join(select_parts)} FROM coordinates LIMIT {limit};"""
                df = pd.read_sql_query(select_sql, conn)
                df = df.dropna(axis=1, how="all")
                print(df)

            else:
                # Generic table, just read first rows
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit};", conn)
                print(df)

        except Exception as e:
            print(f"Error reading table {table}: {e}")

    conn.close()