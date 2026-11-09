
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("error", category=InconsistentVersionWarning)

paket = joblib.load("model_isolation_forest_opentender.joblib")
df = pd.read_csv("hasil_deteksi_anomali_opentender.csv", nrows=1000)

# 1. Buat 3 fitur numerik
fitur_numerik = pd.DataFrame(index=df.index)

fitur_numerik["log_min_value"] = np.log1p(
    df["tender_minValue_amount"].clip(lower=0)
)
fitur_numerik["log_award_value"] = np.log1p(
    df["value_amount"].clip(lower=0)
)

fitur_numerik["selisih_award_rel"] = (
    df["tender_minValue_amount"] - df["value_amount"]
) / df["tender_minValue_amount"]

fitur_numerik = fitur_numerik.replace([np.inf, -np.inf], np.nan)

# 2. Imputasi hanya 3 fitur numerik
fitur_imputasi = pd.DataFrame(
    paket["imputer"].transform(fitur_numerik),
    columns=fitur_numerik.columns,
    index=df.index,
)

# 3. Tambahkan 4 fitur indikator
fitur_imputasi["min_value_missing"] = (
    df["tender_minValue_amount"].isna().astype(int)
)

for kategori in ["goods", "services", "works"]:
    fitur_imputasi[
        f"tender_mainProcurementCategory_{kategori}"
    ] = (
        df["tender_mainProcurementCategory"]
        .eq(kategori)
        .astype(int)
    )

# 4. Pastikan urutan fitur sesuai saat model dilatih
fitur_model = fitur_imputasi.reindex(
    columns=paket["fitur_model"]
)

# 5. Prediksi dan bandingkan dengan label tersimpan
prediksi_lokal = paket["model"].predict(fitur_model)
prediksi_asli = df["prediksi_anomali"].to_numpy()

cocok = prediksi_lokal == prediksi_asli

print("Jumlah data diuji:", len(df))
print("Prediksi cocok:", int(cocok.sum()))
print("Prediksi berbeda:", int((~cocok).sum()))
print("Persentase kecocokan:", round(cocok.mean() * 100, 2), "%")

if cocok.all():
    print("HASIL: Semua label pada sampel cocok.")
else:
    print("HASIL: Ada perbedaan; preprocessing perlu diperiksa.")
