
import joblib
import os

nama_file = "model_isolation_forest_opentender.joblib"

if not os.path.exists(nama_file):
    print("ERROR: File model tidak ditemukan.")
else:
    paket = joblib.load(nama_file)

    print("Model berhasil dimuat.")
    print("Ukuran file (MB):", round(
        os.path.getsize(nama_file) / (1024 ** 2), 2
    ))
    print("Isi paket model:", list(paket.keys()))
    print("Tipe model:", type(paket["model"]).__name__)
    print("Jumlah fitur model:", len(paket["fitur_model"]))
    print("Nama fitur:", paket["fitur_model"])

    print("\nMenguji prediksi model...")

try:
    model = paket["model"]
    fitur = paket["fitur_model"]

    import pandas as pd

    # Data uji sementara dengan 7 fitur model
    data_uji = pd.DataFrame(
        [[0.0] * len(fitur)],
        columns=fitur
    )

    hasil = model.predict(data_uji)

    print("Prediksi berhasil.")
    print("Hasil prediksi:", hasil)

except Exception as e:
    print("Prediksi gagal:", repr(e))
