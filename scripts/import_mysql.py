
import os
import pandas as pd
import mysql.connector

from decimal import Decimal
from datetime import datetime

# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CHUNK_SIZE = 5000

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",  # Sesuaikan jika MySQL memiliki password
    "database": "opentender_bigdata"
}

DATA_FILES = [
    ("main.csv", "tenders"),
    ("awards.csv", "awards"),
    ("awards_suppliers.csv", "award_suppliers")
]


# ==========================================
# KONVERSI DATA BERDASARKAN TIPE MYSQL
# ==========================================

def convert_value(value, data_type):

    if pd.isna(value) or str(value).strip() == "":
        return None

    value = str(value).strip()

    if data_type in (
        "tinyint", "smallint", "mediumint",
        "int", "integer", "bigint"
    ):
        return int(float(value))

    if data_type in ("float", "double", "real"):
        return float(value)

    if data_type in ("decimal", "numeric"):
        return Decimal(value)

    if data_type == "date":
        return pd.to_datetime(value, utc=True).date()

    if data_type in ("datetime", "timestamp"):
        dt = pd.to_datetime(value, utc=True)
        return dt.tz_localize(None).to_pydatetime()

    return value


# ==========================================
# AMBIL STRUKTUR KOLOM MYSQL
# ==========================================

def get_table_columns(cursor, table):

    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s
        AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
    """, (DB_CONFIG["database"], table))

    columns = cursor.fetchall()

    if not columns:
        raise Exception(f"Tabel {table} tidak ditemukan.")

    # Abaikan kolom AUTO_INCREMENT dan generated
    result = []

    for column, data_type, extra in columns:

        if "auto_increment" in extra.lower():
            continue

        if "generated" in extra.lower():
            continue

        result.append((column, data_type))

    return result


# ==========================================
# PERIKSA DATABASE MASIH KOSONG
# ==========================================

def check_empty_database(cursor):

    print("\nMEMERIKSA ISI DATABASE...")

    for _, table in DATA_FILES:

        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        total = cursor.fetchone()[0]

        print(f"{table}: {total:,} baris")

        if total != 0:
            raise Exception(
                f"Tabel {table} tidak kosong. "
                "Migrasi dihentikan untuk mencegah duplikasi."
            )


# ==========================================
# PROSES IMPORT CSV
# ==========================================

def import_csv(connection, filename, table):

    path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    cursor = connection.cursor()

    print("\n" + "=" * 55)
    print(f"MIGRASI: {filename} -> {table}")
    print("=" * 55)

    columns = get_table_columns(cursor, table)

    db_column_names = [col[0] for col in columns]

    # Periksa kesesuaian kolom CSV dan MySQL
    csv_columns = list(pd.read_csv(path, nrows=0).columns)

    missing = set(db_column_names) - set(csv_columns)
    extra = set(csv_columns) - set(db_column_names)

    if missing or extra:
        raise Exception(
            f"Kolom tidak sesuai pada {table}.\n"
            f"Kolom yang hilang: {missing}\n"
            f"Kolom tambahan: {extra}"
        )

    column_names = ", ".join(
        f"`{name}`" for name in db_column_names
    )

    placeholders = ", ".join(
        ["%s"] * len(db_column_names)
    )

    sql = (
        f"INSERT INTO `{table}` "
        f"({column_names}) VALUES ({placeholders})"
    )

    total_inserted = 0

    for chunk in pd.read_csv(
        path,
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=True
    ):

        records = []

        for row in chunk[db_column_names].itertuples(
            index=False,
            name=None
        ):

            converted = []

            for value, (_, data_type) in zip(row, columns):
                converted.append(
                    convert_value(value, data_type)
                )

            records.append(tuple(converted))

        try:
            cursor.executemany(sql, records)
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        total_inserted += len(records)

        print(
            f"Berhasil memasukkan: "
            f"{total_inserted:,} baris",
            end="\r"
        )

    cursor.close()

    print(f"\nSELESAI: {total_inserted:,} baris")

    return total_inserted


# ==========================================
# PROGRAM UTAMA
# ==========================================

def main():

    connection = None

    try:

        print("=" * 55)
        print("MIGRASI DATASET OPENTENDER KE MYSQL")
        print("=" * 55)

        connection = mysql.connector.connect(**DB_CONFIG)

        if not connection.is_connected():
            raise Exception("Koneksi MySQL gagal.")

        print("\nKoneksi MySQL berhasil!")

        cursor = connection.cursor()

        # Pastikan seluruh tabel masih kosong
        check_empty_database(cursor)

        cursor.close()

        total_all = 0

        # Urutan import mengikuti relasi foreign key
        for filename, table in DATA_FILES:

            total = import_csv(
                connection,
                filename,
                table
            )

            total_all += total

        # Verifikasi jumlah data
        print("\n" + "=" * 55)
        print("VERIFIKASI HASIL MIGRASI")
        print("=" * 55)

        cursor = connection.cursor()

        for _, table in DATA_FILES:

            cursor.execute(
                f"SELECT COUNT(*) FROM `{table}`"
            )

            total = cursor.fetchone()[0]

            print(f"{table}: {total:,} baris")

        cursor.close()

        print("\n" + "=" * 55)
        print(f"TOTAL DATA BERHASIL DIMIGRASIKAN: {total_all:,}")
        print("MIGRASI SELESAI!")
        print("=" * 55)

    except Exception as error:

        print("\nMIGRASI GAGAL!")
        print("Penyebab:", error)
        print(
            "\nPeriksa pesan error sebelum mencoba kembali. "
            "Sebagian batch mungkin sudah tersimpan."
        )

    finally:

        if connection and connection.is_connected():
            connection.close()
            print("\nKoneksi MySQL ditutup.")


if __name__ == "__main__":
    main()
