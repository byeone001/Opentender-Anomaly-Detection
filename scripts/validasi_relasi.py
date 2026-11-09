
import pandas as pd
import sqlite3
import os

DATASET = "dataset"
DB_TEMP = "validasi_sementara.sqlite"
CHUNK_SIZE = 100000

if os.path.exists(DB_TEMP):
    os.remove(DB_TEMP)

db = sqlite3.connect(DB_TEMP)
cursor = db.cursor()

# Tabel sementara untuk pemeriksaan ID
cursor.executescript("""
CREATE TABLE main_ids (
    ocid TEXT,
    main_id TEXT
);

CREATE TABLE award_ids (
    id TEXT,
    main_ocid TEXT,
    main_id TEXT
);

CREATE TABLE supplier_ids (
    id TEXT,
    main_ocid TEXT,
    main_id TEXT,
    awards_id TEXT
);

CREATE INDEX idx_main_ocid ON main_ids(ocid);
CREATE INDEX idx_main_id ON main_ids(main_id);
CREATE INDEX idx_award_id ON award_ids(id);
""")


def load_csv(filename, columns, table):
    path = os.path.join(DATASET, filename)
    total = 0

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        # PENTING: Pastikan urutan kolom sesuai tabel SQLite
        chunk = chunk[columns]

        records = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in chunk.itertuples(index=False, name=None)
        ]

        placeholders = ",".join(["?"] * len(columns))

        cursor.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})",
            records
        )

        total += len(records)
        db.commit()

    print(f"{filename}: {total:,} baris")


print("MEMBACA DATASET...")
load_csv("main.csv", ["ocid", "id"], "main_ids")
load_csv("awards.csv", ["id", "main_ocid", "main_id"], "award_ids")
load_csv(
    "awards_suppliers.csv",
    ["id", "main_ocid", "main_id", "awards_id"],
    "supplier_ids"
)

def check(label, query):
    cursor.execute(query)
    result = cursor.fetchone()[0]
    print(f"{label}: {result:,}")

print("\n=== PEMERIKSAAN DUPLIKASI ===")

check("Duplikasi OCID",
"""
SELECT COUNT(*) - COUNT(DISTINCT ocid)
FROM main_ids WHERE ocid IS NOT NULL
""")

check("Duplikasi main.id",
"""
SELECT COUNT(*) - COUNT(DISTINCT main_id)
FROM main_ids WHERE main_id IS NOT NULL
""")

check("Duplikasi award.id",
"""
SELECT COUNT(*) - COUNT(DISTINCT id)
FROM award_ids WHERE id IS NOT NULL
""")

check("Duplikasi supplier.id",
"""
SELECT COUNT(*) - COUNT(DISTINCT id)
FROM supplier_ids WHERE id IS NOT NULL
""")


print("\n=== PEMERIKSAAN DUPLIKASI KOMBINASI SUPPLIER ===")

check("Duplikasi kombinasi supplier.id + awards_id", """
SELECT COUNT(*) FROM (
    SELECT id, awards_id
    FROM supplier_ids
    GROUP BY id, awards_id
    HAVING COUNT(*) > 1
) AS duplicate_pairs
""")

check("Supplier dengan ID kosong", """
SELECT COUNT(*)
FROM supplier_ids
WHERE id IS NULL OR TRIM(id) = ''
""")


print("\n=== PEMERIKSAAN RELASI ===")

check("Award tanpa tender (OCID)",
"""
SELECT COUNT(*)
FROM award_ids a
WHERE a.main_ocid IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM main_ids m
    WHERE m.ocid = a.main_ocid
)
""")

check("Award tanpa main.id",
"""
SELECT COUNT(*)
FROM award_ids a
WHERE a.main_id IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM main_ids m
    WHERE m.main_id = a.main_id
)
""")

check("Supplier tanpa tender (OCID)",
"""
SELECT COUNT(*)
FROM supplier_ids s
WHERE s.main_ocid IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM main_ids m
    WHERE m.ocid = s.main_ocid
)
""")

check("Supplier tanpa main.id",
"""
SELECT COUNT(*)
FROM supplier_ids s
WHERE s.main_id IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM main_ids m
    WHERE m.main_id = s.main_id
)
""")

check("Supplier tanpa award",
"""
SELECT COUNT(*)
FROM supplier_ids s
WHERE s.awards_id IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM award_ids a
    WHERE a.id = s.awards_id
)
""")

print("\nVALIDASI SELESAI.")
db.close()
