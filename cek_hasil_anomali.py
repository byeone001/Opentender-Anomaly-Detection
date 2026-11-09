import pandas as pd

file = "hasil_deteksi_anomali_opentender.csv"

df = pd.read_csv(file)

print("Jumlah baris:", len(df))
print("\nNama kolom:")
print(df.columns.tolist())

print("\n5 data pertama:")
print(df.head())

print("\nDistribusi hasil deteksi:")
print(df.iloc[:, -1].value_counts())