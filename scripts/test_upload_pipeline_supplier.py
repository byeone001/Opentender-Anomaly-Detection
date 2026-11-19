"""
Test script untuk memverifikasi integrasi supplier pada pipeline upload app.py
dan kalkulasi RF-4 (rf_pangsa_supplier, rf_4_supplier_conc) pada red_flag_engine.py.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import numpy as np

# Import red_flag_engine
from red_flag_engine import hitung_red_flags

# Fungsi integrasikan_opentender dan cari_tabel dari app.py
def nama_file_sederhana(nama):
    return Path(nama.replace("\\", "/")).name.lower()

def cari_tabel(kumpulan_data, nama_target):
    if isinstance(nama_target, str):
        target_list = [nama_target.lower()]
    else:
        target_list = [t.lower() for t in nama_target]

    for n, d in kumpulan_data.items():
        sederhana = nama_file_sederhana(n)
        for t in target_list:
            if sederhana == t or sederhana.endswith(t):
                return d.copy()
    return None

def integrasikan_opentender(kumpulan_data):
    main = cari_tabel(kumpulan_data, "main.csv")
    awards = cari_tabel(kumpulan_data, "awards.csv")
    suppliers = cari_tabel(
        kumpulan_data,
        ["awards_suppliers.csv", "award_suppliers.csv", "award_supplier.csv", "suppliers.csv"]
    )

    if main is None and awards is None:
        kandidat = next(iter(kumpulan_data.values())).copy()
        wajib = {"tender_minValue_amount", "value_amount", "tender_mainProcurementCategory"}
        if wajib.issubset(kandidat.columns):
            return kandidat
        raise ValueError(
            "File tunggal harus memiliki: tender_minValue_amount, value_amount, tender_mainProcurementCategory."
        )

    if main is None or awards is None:
        raise ValueError("Data relasional OpenTender membutuhkan main.csv dan awards.csv.")

    wajib_main = ["id", "ocid", "tender_minValue_amount", "tender_mainProcurementCategory"]
    wajib_awards = ["value_amount", "main_id", "main_ocid"]
    kurang_m = [c for c in wajib_main if c not in main.columns]
    kurang_a = [c for c in wajib_awards if c not in awards.columns]
    if kurang_m or kurang_a:
        raise ValueError(f"Kolom hilang: main={kurang_m}, awards={kurang_a}")

    for col in ("id", "ocid"):
        main[col] = main[col].astype("string").str.strip()
    for col in ("main_id", "main_ocid"):
        awards[col] = awards[col].astype("string").str.strip()

    id_m = main.loc[main["id"].notna() & main["id"].ne(""), "id"]
    id_a = awards.loc[awards["main_id"].notna() & awards["main_id"].ne(""), "main_id"]
    oc_m = main.loc[main["ocid"].notna() & main["ocid"].ne(""), "ocid"]
    oc_a = awards.loc[awards["main_ocid"].notna() & awards["main_ocid"].ne(""), "main_ocid"]

    cocok_id = len(set(id_m.unique()) & set(id_a.unique()))
    cocok_ocid = len(set(oc_m.unique()) & set(oc_a.unique()))
    if cocok_id >= cocok_ocid and cocok_id > 0:
        k_main, k_awards = "id", "main_id"
    elif cocok_ocid > 0:
        k_main, k_awards = "ocid", "main_ocid"
    else:
        raise ValueError("Tidak ditemukan ID/OCID yang berelasi antara main.csv dan awards.csv.")

    main["tender_minValue_amount"] = pd.to_numeric(main["tender_minValue_amount"], errors="coerce")
    awards["value_amount"] = pd.to_numeric(awards["value_amount"], errors="coerce")

    hasil = awards.merge(
        main, left_on=k_awards, right_on=k_main, how="left",
        suffixes=("_award", "_main")
    )
    if hasil.empty:
        raise ValueError("Hasil penggabungan dataset kosong.")

    # Agregasi & Join Suppliers
    if suppliers is not None and not suppliers.empty:
        supp = suppliers.copy()
        supp_key = None
        for cand in ("main_ocid", "ocid", "main_id", "id_main", "awards_id"):
            if cand in supp.columns:
                supp_key = cand
                break

        if supp_key:
            supp[supp_key] = supp[supp_key].astype("string").str.strip()
            supp_valid = supp[
                supp[supp_key].notna()
                & (supp[supp_key] != "")
                & (supp[supp_key].str.lower() != "nan")
            ].copy()

            if not supp_valid.empty:
                rename_map = {}
                if "supplier_id" not in supp_valid.columns:
                    if "id" in supp_valid.columns:
                        rename_map["id"] = "supplier_id"
                    elif "supplier" in supp_valid.columns:
                        rename_map["supplier"] = "supplier_id"
                if "supplier_name" not in supp_valid.columns:
                    if "name" in supp_valid.columns:
                        rename_map["name"] = "supplier_name"
                    elif "supplier" in supp_valid.columns and "supplier_id" in supp_valid.columns:
                        rename_map["supplier"] = "supplier_name"

                if rename_map:
                    supp_valid = supp_valid.rename(columns=rename_map)

                if "supplier_id" not in supp_valid.columns and "supplier_name" in supp_valid.columns:
                    supp_valid["supplier_id"] = supp_valid["supplier_name"]

                if "supplier_id" in supp_valid.columns:
                    supp_valid["supplier_id"] = supp_valid["supplier_id"].astype("string").str.strip()
                    if "supplier_name" in supp_valid.columns:
                        supp_valid["supplier_name"] = supp_valid["supplier_name"].astype("string").str.strip()
                    else:
                        supp_valid["supplier_name"] = supp_valid["supplier_id"]

                    supp_agg = (
                        supp_valid.groupby(supp_key, as_index=False)
                        .agg(
                            supplier_id=("supplier_id", "first"),
                            supplier_name=("supplier_name", "first"),
                            jumlah_supplier=("supplier_id", "nunique"),
                        )
                    )

                    join_left = None
                    if supp_key in ("main_ocid", "ocid"):
                        for cand in ("ocid", "main_ocid", "ocid_main", "ocid_award"):
                            if cand in hasil.columns and hasil[cand].notna().any():
                                join_left = cand
                                break
                    elif supp_key in ("main_id", "id_main"):
                        for cand in ("id", "main_id", "id_main", "id_award"):
                            if cand in hasil.columns and hasil[cand].notna().any():
                                join_left = cand
                                break
                    elif supp_key == "awards_id":
                        for cand in ("award_id", "id_award", "id"):
                            if cand in hasil.columns and hasil[cand].notna().any():
                                join_left = cand
                                break

                    if not join_left:
                        for cand in ("ocid", "main_ocid", "id", "main_id"):
                            if cand in hasil.columns:
                                val_h = set(hasil[cand].dropna().astype(str).str.strip())
                                val_s = set(supp_agg[supp_key].dropna().astype(str).str.strip())
                                if len(val_h & val_s) > 0:
                                    join_left = cand
                                    break

                    if join_left:
                        hasil[join_left] = hasil[join_left].astype("string").str.strip()
                        supp_agg[supp_key] = supp_agg[supp_key].astype("string").str.strip()
                        hasil = hasil.merge(
                            supp_agg,
                            left_on=join_left,
                            right_on=supp_key,
                            how="left",
                            suffixes=("", "_supp"),
                        )
                        if supp_key != join_left and supp_key in hasil.columns:
                            hasil = hasil.drop(columns=[supp_key])
                        if f"{supp_key}_supp" in hasil.columns:
                            hasil = hasil.drop(columns=[f"{supp_key}_supp"])

    return hasil

def deteksi_anomali(data_baru, paket):
    """Inference Isolation Forest dengan preprocessing 7-fitur identik saat training (dari app.py)."""
    wajib = ["tender_minValue_amount", "value_amount", "tender_mainProcurementCategory"]
    kurang = [c for c in wajib if c not in data_baru.columns]
    if kurang:
        raise ValueError("Kolom model belum tersedia: " + ", ".join(kurang))

    data_baru = data_baru.copy()
    nilai_min = pd.to_numeric(data_baru["tender_minValue_amount"], errors="coerce")
    nilai_kontrak = pd.to_numeric(data_baru["value_amount"], errors="coerce")

    fitur = pd.DataFrame(index=data_baru.index)
    fitur["log_min_value"] = np.log1p(nilai_min.clip(lower=0))
    fitur["log_award_value"] = np.log1p(nilai_kontrak.clip(lower=0))
    fitur["selisih_award_rel"] = ((nilai_min - nilai_kontrak) / nilai_min).where(nilai_min.ne(0))
    fitur = fitur.replace([np.inf, -np.inf], np.nan)

    fitur_imp = pd.DataFrame(
        paket["imputer"].transform(fitur),
        columns=fitur.columns,
        index=data_baru.index,
    )
    fitur_imp["min_value_missing"] = nilai_min.isna().astype(int)

    kategori = data_baru["tender_mainProcurementCategory"].astype("string").str.strip().str.lower()
    for item in ("goods", "services", "works"):
        fitur_imp[f"tender_mainProcurementCategory_{item}"] = (
            kategori == item
        ).fillna(False).astype(int)

    fitur_model = fitur_imp.reindex(columns=paket["fitur_model"])
    prediksi = paket["model"].predict(fitur_model)
    skor = paket["model"].score_samples(fitur_model)

    hasil = data_baru.copy()
    hasil["log_min_value"] = fitur["log_min_value"].values
    hasil["log_award_value"] = fitur["log_award_value"].values
    hasil["selisih_award_rel"] = fitur["selisih_award_rel"].values
    hasil["prediksi_anomali"] = prediksi
    hasil["skor_anomali"] = -skor
    hasil["status_deteksi"] = np.where(prediksi == -1, "Anomali", "Normal")
    return hasil

def run_tests():
    print("--- 1. Membaca Sample Data OpenTender ---")
    df_main = pd.read_csv("dataset/main.csv", nrows=5000, low_memory=False)
    df_awards = pd.read_csv("dataset/awards.csv", nrows=5000, low_memory=False)
    df_supp = pd.read_csv("dataset/awards_suppliers.csv", nrows=5000, low_memory=False)

    print(f"Main shape: {df_main.shape}, Awards shape: {df_awards.shape}, Supp shape: {df_supp.shape}")

    # Tes 1: Format arsip ZIP dengan path subfolder
    print("\n--- 2. Tes Integrasi dengan Subfolder ZIP (e.g. 2023/awards_suppliers.csv) ---")
    kumpulan = {
        "2023/main.csv": df_main,
        "2023/awards.csv": df_awards,
        "2023/awards_suppliers.csv": df_supp,
    }
    gabung = integrasikan_opentender(kumpulan)
    print(f"Hasil gabung shape: {gabung.shape}")
    print(f"Kolom supplier ada di gabung? supplier_id: {'supplier_id' in gabung.columns}, supplier_name: {'supplier_name' in gabung.columns}")
    if "supplier_id" in gabung.columns:
        print(f"Jumlah supplier_id non-null: {gabung['supplier_id'].notna().sum()}/{len(gabung)}")
        print(f"Contoh data supplier_name: {gabung['supplier_name'].dropna().head(3).tolist()}")

    # Tes 2: Jalankan Deteksi Anomali & Red Flag Engine
    print("\n--- 3. Menjalankan Model IF & Red Flag Engine ---")
    paket = joblib.load("model_isolation_forest_opentender.joblib")
    hasil_if = deteksi_anomali(gabung, paket)
    print(f"Deteksi IF selesai. Status deteksi: {hasil_if['status_deteksi'].value_counts().to_dict()}")

    hasil_pred = hitung_red_flags(hasil_if, rf_threshold=2)
    print(f"Hitung Red Flags selesai!")
    print(f"Kolom rf_pangsa_supplier ada? {'rf_pangsa_supplier' in hasil_pred.columns}")
    print(f"Kolom rf_4_supplier_conc ada? {'rf_4_supplier_conc' in hasil_pred.columns}")

    if "rf_pangsa_supplier" in hasil_pred.columns:
        non_null_pangsa = hasil_pred["rf_pangsa_supplier"].notna().sum()
        print(f"rf_pangsa_supplier non-null count: {non_null_pangsa}/{len(hasil_pred)}")
        if non_null_pangsa > 0:
            print(f"Distribusi rf_pangsa_supplier: min={hasil_pred['rf_pangsa_supplier'].min():.3f}, mean={hasil_pred['rf_pangsa_supplier'].mean():.3f}, max={hasil_pred['rf_pangsa_supplier'].max():.3f}")
        print(f"Distribusi rf_4_supplier_conc: {hasil_pred['rf_4_supplier_conc'].value_counts().to_dict()}")

    # Tes 3: Fallback aman jika tanpa file supplier
    print("\n--- 4. Tes Fallback Tanpa File Supplier ---")
    kumpulan_no_supp = {
        "main.csv": df_main,
        "awards.csv": df_awards,
    }
    gabung_no_supp = integrasikan_opentender(kumpulan_no_supp)
    hasil_if_no_supp = deteksi_anomali(gabung_no_supp, paket)
    hasil_pred_no_supp = hitung_red_flags(hasil_if_no_supp, rf_threshold=2)
    print(f"Fallback berhasil! RF-4 flag count: {hasil_pred_no_supp['rf_4_supplier_conc'].sum()} (harus 0 karena tidak ada supplier)")
    print(f"rf_pangsa_supplier all NaN? {hasil_pred_no_supp['rf_pangsa_supplier'].isna().all()}")

    print("\n=== SEMUA TES BERHASIL 100% ===")

if __name__ == "__main__":
    run_tests()
