
import pandas as pd

FILE_DATA = "hasil_deteksi_anomali_opentender.csv"

print("Memeriksa seluruh dataset...")

jumlah_baris = 0
jumlah_prediksi = {}
jumlah_status = {}

for df in pd.read_csv(FILE_DATA, chunksize=100_000):
    jumlah_baris += len(df)

    for kolom, hasil in [
        ("prediksi_anomali", jumlah_prediksi),
        ("status_deteksi", jumlah_status),
    ]:
        hitungan = df[kolom].value_counts(dropna=False)

        for nilai, jumlah in hitungan.items():
            hasil[nilai] = hasil.get(nilai, 0) + int(jumlah)

print("\n=== HASIL PEMERIKSAAN ===")
print("Total baris:", f"{jumlah_baris:,}")

print("\nJumlah berdasarkan prediksi model:")
for label, jumlah in jumlah_prediksi.items():
    print(f"{label}: {jumlah:,}")

print("\nJumlah berdasarkan status deteksi:")
for status, jumlah in jumlah_status.items():
    print(f"{status}: {jumlah:,}")

print("\nPemeriksaan selesai.")
