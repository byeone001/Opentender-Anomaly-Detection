
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

FILE_MODEL = "model_isolation_forest_opentender.joblib"
FILE_DATA = "hasil_deteksi_anomali_opentender.csv"

if not os.path.exists(FILE_MODEL):
    print("File model tidak ditemukan.")
    raise SystemExit

if not os.path.exists(FILE_DATA):
    print("File dataset hasil deteksi tidak ditemukan.")
    raise SystemExit

print("Memuat model...")
paket = joblib.load(FILE_MODEL)

print("Membaca 100 baris dataset...")
df = pd.read_csv(FILE_DATA, nrows=100)

print("\nKolom dataset:")
print(df.columns.tolist())

print("\nJumlah baris yang berhasil dibaca:", len(df))

print("\nContoh data:")
print(df.head(3).to_string(index=False))

print("\nKolom prediksi yang tersedia:")
print([
    kolom for kolom in df.columns
    if any(kata in kolom.lower()
           for kata in ["anomali", "prediksi", "label", "skor", "status"])
])
