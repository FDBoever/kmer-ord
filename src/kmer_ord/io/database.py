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
            qualities TEXT
        );
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

    for i in range(1, len(df.columns), 2):
        col_x = df.columns[i]
        col_y = df.columns[i + 1]
        base = col_x.rsplit("_", 1)[0]
        methods.add(base)

        cursor.execute(
            f"SELECT AddGeometryColumn('coordinates','{base}',4326,'POINT','XY');"
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
            x = row[f"{method}_1"]
            y = row[f"{method}_2"]

            columns.append(method)
            values.append(f"POINT({x} {y})")
            placeholders.append("ST_GeomFromText(?,4326)")

        insert_sql = f"""
            INSERT INTO coordinates ({','.join(columns)})
            VALUES ({','.join(placeholders)});
        """

        cursor.execute(insert_sql, values)

    conn.commit()



# ---------------------------------------------------------
# DATABASE INSPECTION (DEBUGGING / QA)
# ---------------------------------------------------------

def inspect_database(db_file: Path, limit: int = 5):
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Load SpatiaLite
    try:
        conn.enable_load_extension(True)
        cursor.execute("SELECT load_extension('mod_spatialite');")
    except sqlite3.OperationalError:
        pass

    user_tables = ["fasta", "features", "coordinates"]

    for table in user_tables:
        print(f"\n--- TABLE: {table} ---")

        # Check table exists
        try:
            cursor.execute(f"PRAGMA table_info({table});")
            schema = cursor.fetchall()
        except sqlite3.OperationalError:
            print("Table not found.")
            continue

        if not schema:
            print("Table not found.")
            continue

        try:
            # Special handling for coordinates table
            if table == "coordinates":

                # Get geometry columns via geometry_columns metadata
                geom_query = """
                    SELECT f_geometry_column
                    FROM geometry_columns
                    WHERE f_table_name = 'coordinates';
                """
                geom_cols = pd.read_sql_query(geom_query, conn)

                if geom_cols.empty:
                    # fallback if metadata missing
                    df = pd.read_sql_query(
                        f"SELECT * FROM coordinates LIMIT {limit};", conn
                    )
                    print(df)
                    continue

                geom_names = geom_cols["f_geometry_column"].tolist()

                # Build SELECT with ST_X and ST_Y for each geometry column
                select_parts = ["header"]

                for col in geom_names:
                    select_parts.append(f"ST_X({col}) AS {col}_x")
                    select_parts.append(f"ST_Y({col}) AS {col}_y")

                select_sql = f"""
                    SELECT {', '.join(select_parts)}
                    FROM coordinates
                    LIMIT {limit};
                """

                df = pd.read_sql_query(select_sql, conn)
                print(df)

            else:
                # Normal tables
                df = pd.read_sql_query(
                    f"SELECT * FROM {table} LIMIT {limit};", conn
                )
                print(df)

        except Exception as e:
            print("Error reading rows:", e)

    conn.close()