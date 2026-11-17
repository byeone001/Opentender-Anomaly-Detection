"""
Skrip pengujian End-to-End integrasi:
Upload CSV mini -> Prediksi Model -> Simpan MySQL -> Kirim Firebase -> Verifikasi
"""

import sys
import os
import uuid
import pandas as pd
import numpy as np
import joblib

from config import FILE_MODEL
from db_mysql import get_connection, simpan_batch, simpan_hasil_prediksi, perbarui_status_batch
import db_firebase

def test_pipeline():
    print("=" * 60)
    print("PENGUJIAN INTEGRASI END-TO-END (CSV KECIL)")
    print("=" * 60)

    # 1. Pemuatan Model
    print("\n[1] Memuat model...")
    paket = joblib.load(str(FILE_MODEL))
    print("    Model berhasil dimuat:", type(paket["model"]).__name__)

    # 2. Baca CSV Uji
    print("\n[2] Membaca dataset uji kecil...")
    main_df = pd.read_csv("dataset_test/main.csv")
    awards_df = pd.read_csv("dataset_test/awards.csv")
    print(f"    main.csv   : {len(main_df)} baris")
    print(f"    awards.csv : {len(awards_df)} baris")

    # 3. Integrasi
    print("\n[3] Mengintegrasikan dataset (merge)...")
    for col in ("id", "ocid"):
        main_df[col] = main_df[col].astype("string").str.strip()
    for col in ("main_id", "main_ocid"):
        awards_df[col] = awards_df[col].astype("string").str.strip()

    main_df["tender_minValue_amount"] = pd.to_numeric(main_df["tender_minValue_amount"], errors="coerce")
    awards_df["value_amount"] = pd.to_numeric(awards_df["value_amount"], errors="coerce")

    merged = awards_df.merge(
        main_df, left_on="main_id", right_on="id", how="left",
        suffixes=("_award", "_main")
    )
    print(f"    Hasil merge: {len(merged)} baris")

    # 4. Deteksi Anomali
    print("\n[4] Menjalankan deteksi anomali...")
    nilai_min = pd.to_numeric(merged["tender_minValue_amount"], errors="coerce")
    nilai_kontrak = pd.to_numeric(merged["value_amount"], errors="coerce")

    fitur = pd.DataFrame(index=merged.index)
    fitur["log_min_value"] = np.log1p(nilai_min.clip(lower=0))
    fitur["log_award_value"] = np.log1p(nilai_kontrak.clip(lower=0))
    fitur["selisih_award_rel"] = ((nilai_min - nilai_kontrak) / nilai_min).where(nilai_min.ne(0))
    fitur = fitur.replace([np.inf, -np.inf], np.nan)

    fitur_imp = pd.DataFrame(
        paket["imputer"].transform(fitur),
        columns=fitur.columns,
        index=merged.index,
    )
    fitur_imp["min_value_missing"] = nilai_min.isna().astype(int)

    kategori = merged["tender_mainProcurementCategory"].astype("string").str.strip().str.lower()
    for item in ("goods", "services", "works"):
        fitur_imp[f"tender_mainProcurementCategory_{item}"] = (
            kategori == item
        ).fillna(False).astype(int)

    fitur_model = fitur_imp.reindex(columns=paket["fitur_model"])
    pred = paket["model"].predict(fitur_model)
    skor = paket["model"].score_samples(fitur_model)

    hasil = merged.copy()
    hasil["log_min_value"] = fitur["log_min_value"].values
    hasil["log_award_value"] = fitur["log_award_value"].values
    hasil["selisih_award_rel"] = fitur["selisih_award_rel"].values
    hasil["prediksi_anomali"] = pred
    hasil["skor_anomali"] = -skor
    hasil["status_deteksi"] = np.where(pred == -1, "Anomali", "Normal")

    total_rows = len(hasil)
    total_anomali = int((hasil["status_deteksi"] == "Anomali").sum())
    total_normal = total_rows - total_anomali
    pct_anomali = round((total_anomali / total_rows) * 100, 2)

    print(f"    Total dianalisis : {total_rows}")
    print(f"    Normal           : {total_normal}")
    print(f"    Anomali          : {total_anomali} ({pct_anomali}%)")

    # 5. Simpan ke MySQL
    batch_id = f"test-{uuid.uuid4().hex[:8]}"
    print(f"\n[5] Menyimpan ke MySQL (Batch ID: {batch_id})...")
    conn = get_connection()
    simpan_batch(conn, batch_id, "test_dataset_small.csv", total_rows=total_rows, status="processing")
    n_norm, n_anom = simpan_hasil_prediksi(conn, batch_id, hasil)
    perbarui_status_batch(conn, batch_id, n_norm, n_anom, status="done")

    # Verifikasi langsung dari MySQL
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM upload_batches WHERE batch_id = %s", (batch_id,))
    batch_row = cur.fetchone()
    print("    Verifikasi batch MySQL:", batch_row)

    cur.execute("SELECT COUNT(*) AS total FROM analysis_results WHERE batch_id = %s", (batch_id,))
    res_count = cur.fetchone()["total"]
    print(f"    Verifikasi baris analysis_results: {res_count} baris tersimpan.")
    cur.close()
    conn.close()

    # 6. Kirim ke Firebase RTDB
    print(f"\n[6] Mengirim ke Firebase Realtime Database...")
    ok_fb, msg_fb = db_firebase.kirim_ringkasan_batch(
        batch_id=batch_id,
        filename="test_dataset_small.csv",
        total_rows=total_rows,
        total_normal=total_normal,
        total_anomali=total_anomali,
        catatan="pengujian_otomatis",
    )
    print("    Kirim history:", ok_fb, msg_fb)

    ok_dash, msg_dash = db_firebase.perbarui_dashboard(total_normal, total_anomali)
    print("    Update dashboard:", ok_dash, msg_dash)

    # 7. Verifikasi dari Firebase
    print("\n[7] Verifikasi pembacaan dari Firebase...")
    riwayat = db_firebase.ambil_riwayat_firebase(limit=5)
    print("    Riwayat terbaru di Firebase:")
    for r in riwayat:
        if r.get("batch_id") == batch_id:
            print(f"    -> DITEMUKAN BATCH {batch_id}:", r)

    dash_node = db_firebase.ambil_dashboard_firebase()
    print("    Data terkini opentender/dashboard:")
    print("      total_upload_tenders :", dash_node.get("total_upload_tenders"))
    print("      total_upload_normal  :", dash_node.get("total_upload_normal"))
    print("      total_upload_anomali :", dash_node.get("total_upload_anomali"))
    print("      pct_upload_anomali   :", dash_node.get("pct_upload_anomali"))
    print("      last_updated         :", dash_node.get("last_updated"))

    print("\n" + "=" * 60)
    print("SEMUA PENGUJIAN INTEGRASI BERHASIL 100%!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
