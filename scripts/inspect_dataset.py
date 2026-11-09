
import pandas as pd
import os

# Lokasi dataset
DATASET_DIR = "dataset"

files = [
    "main.csv",
    "awards.csv",
    "awards_suppliers.csv"
]

for filename in files:
    path = os.path.join(DATASET_DIR, filename)

    print("\n" + "=" * 70)
    print(f"PEMERIKSAAN DATASET: {filename}")
    print("=" * 70)

    if not os.path.exists(path):
        print("FILE TIDAK DITEMUKAN:", path)
        continue

    # Membaca beberapa baris untuk memeriksa struktur
    df = pd.read_csv(path, nrows=5, low_memory=False)

    print("\nKolom dataset:")
    for column in df.columns:
        print(f"- {column}")

    print("\nJumlah kolom:", len(df.columns))

    print("\nTipe data:")
    print(df.dtypes)

    print("\nContoh 5 baris:")
    print(df.to_string(index=False))

print("\nPEMERIKSAAN STRUKTUR SELESAI.")
