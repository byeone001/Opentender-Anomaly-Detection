"""
Pipeline Utama Tahap 1 & Tahap 2:
1. Agregasi dataset relasional MySQL menjadi 1 baris per tender (OCID)
2. Retraining Isolation Forest (7 fitur) & penyimpanan model .joblib
3. Evaluasi pembanding Rule-Based Red Flag (RF-1 s.d. RF-4)
4. Eksekusi 5 skenario eksperimen lengkap + benchmark runtime & memori
"""

import os
import shutil
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from config import FILE_HASIL_CSV, FILE_MODEL, MYSQL_CONFIG
from db_mysql import get_connection
from red_flag_engine import hitung_red_flags

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(RESULTS_DIR, "laporan_eksperimen_ocid.txt")
CASE_STUDY_FILE = os.path.join(RESULTS_DIR, "studi_kasus_anomali.csv")


def get_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def log(f, text: str = ""):
    print(text)
    f.write(text + "\n")
    f.flush()


def backup_if_exists(path):
    if path.exists():
        bak = path.with_name(path.stem + ".backup" + path.suffix)
        if not bak.exists():
            shutil.copy2(path, bak)
            return str(bak)
    return None


def main():
    t_start_all = time.perf_counter()
    ram_awal = get_ram_mb()

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        log(f, "=" * 82)
        log(f, "LAPORAN EKSPERIMEN KOMPARATIF LEVEL OCID: ISOLATION FOREST VS RULE-BASED RED FLAG")
        log(f, f"Waktu Eksekusi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f, f"RAM Awal Proses: {ram_awal:.2f} MB")
        log(f, "=" * 82)

        # ------------------------------------------------------------------
        # TAHAP 1: PEMUATAN & INTEGRASI RELASIONAL KE LEVEL OCID
        # ------------------------------------------------------------------
        log(f, "\n[TAHAP 1] PEMUATAN & AGREGASI DATA MYSQL KE TINGKAT OCID")
        t0 = time.perf_counter()
        conn = get_connection()

        df_tenders = pd.read_sql(
            """
            SELECT
                id, ocid, buyer_id, buyer_name,
                tender_mainProcurementCategory, tender_title,
                tender_datePublished, tender_status,
                tender_minValue_amount, tender_procuringEntity_name
            FROM tenders
            """,
            conn,
        )
        df_awards = pd.read_sql(
            """
            SELECT id AS award_id, date AS award_date, value_amount, main_ocid
            FROM awards
            """,
            conn,
        )
        df_suppliers = pd.read_sql(
            """
            SELECT id AS supplier_id, name AS supplier_name, main_ocid
            FROM award_suppliers
            """,
            conn,
        )
        conn.close()
        t_load = time.perf_counter() - t0
        ram_after_load = get_ram_mb()

        n_raw_tenders = len(df_tenders)
        n_raw_awards = len(df_awards)
        n_raw_suppliers = len(df_suppliers)

        log(f, f"  - Total baris tabel tenders         : {n_raw_tenders:,}")
        log(f, f"  - Total baris tabel awards          : {n_raw_awards:,}")
        log(f, f"  - Total baris tabel award_suppliers : {n_raw_suppliers:,}")
        log(f, f"  - Waktu baca MySQL                  : {t_load:.2f} detik (RAM: {ram_after_load:.2f} MB)")

        # Pembersihan kunci OCID & penyaringan award valid
        df_tenders["ocid"] = df_tenders["ocid"].astype("string").str.strip()
        df_awards["main_ocid"] = df_awards["main_ocid"].astype("string").str.strip()
        df_awards["value_amount"] = pd.to_numeric(df_awards["value_amount"], errors="coerce")
        df_awards["award_date"] = pd.to_datetime(df_awards["award_date"], errors="coerce", utc=True)

        awards_valid = df_awards[
            df_awards["main_ocid"].notna()
            & (df_awards["main_ocid"] != "")
            & df_awards["value_amount"].notna()
            & (df_awards["value_amount"] > 0)
        ].copy()

        # Agregasi 1 baris per main_ocid pada awards
        awards_agg = (
            awards_valid.groupby("main_ocid", as_index=False)
            .agg(
                value_amount=("value_amount", "sum"),
                jumlah_award=("award_id", "count"),
                award_date=("award_date", "min"),
            )
        )

        # Agregasi 1 penyedia utama per main_ocid
        df_suppliers["main_ocid"] = df_suppliers["main_ocid"].astype("string").str.strip()
        supp_valid = df_suppliers[
            df_suppliers["main_ocid"].notna() & (df_suppliers["main_ocid"] != "")
        ].copy()
        supp_agg = (
            supp_valid.groupby("main_ocid", as_index=False)
            .agg(
                supplier_id=("supplier_id", "first"),
                supplier_name=("supplier_name", "first"),
                jumlah_supplier=("supplier_id", "nunique"),
            )
        )

        # Inner join tenders dengan awards_agg -> menjamin tepat 1 baris per OCID yang memiliki kontrak
        df_ocid = df_tenders.merge(
            awards_agg, left_on="ocid", right_on="main_ocid", how="inner"
        ).merge(
            supp_agg, left_on="ocid", right_on="main_ocid", how="left", suffixes=("", "_supp")
        )
        df_ocid = df_ocid.drop(columns=[c for c in ["main_ocid", "main_ocid_supp"] if c in df_ocid.columns])

        # Audit missing value & kualitas data pada populasi OCID final
        n_ocid_final = len(df_ocid)
        min_val_raw = pd.to_numeric(df_ocid["tender_minValue_amount"], errors="coerce")
        n_min_na = int(min_val_raw.isna().sum())
        n_min_zero_neg = int((min_val_raw <= 0).sum())
        n_tgl_pub_na = int(pd.to_datetime(df_ocid["tender_datePublished"], errors="coerce").isna().sum())

        log(f, "\n[EKSPERIMEN 1] RINGKASAN PENYUSUTAN & KUALITAS DATA LEVEL OCID")
        log(f, f"  1. Populasi awal tenders (main.csv)          : {n_raw_tenders:,} OCID")
        log(f, f"  2. Baris awards mentah                       : {n_raw_awards:,} baris")
        log(f, f"  3. Baris awards valid (value_amount > 0)     : {len(awards_valid):,} baris")
        log(f, f"  4. OCID unik setelah agregasi awards         : {len(awards_agg):,} OCID")
        log(f, f"  5. Dataset analisis final (1 baris / OCID)   : {n_ocid_final:,} OCID")
        log(f, f"  - Missing tender_minValue_amount (NaN)       : {n_min_na:,} ({n_min_na/n_ocid_final*100:.2f}%)")
        log(f, f"  - Nilai tender_minValue_amount <= 0          : {n_min_zero_neg:,} ({n_min_zero_neg/n_ocid_final*100:.2f}%)")
        log(f, f"  - Tanggal publikasi kosong/tidak valid       : {n_tgl_pub_na:,} ({n_tgl_pub_na/n_ocid_final*100:.2f}%)")
        log(f, "\n  Distribusi Kategori Pengadaan pada Populasi OCID:")
        for kat, cnt in df_ocid["tender_mainProcurementCategory"].fillna("missing").value_counts().items():
            log(f, f"    * {kat}: {cnt:,} ({cnt/n_ocid_final*100:.2f}%)")

        # ------------------------------------------------------------------
        # TAHAP 2: FEATURE ENGINEERING & RETRAINING ISOLATION FOREST
        # ------------------------------------------------------------------
        log(f, "\n[TAHAP 2] FEATURE ENGINEERING & RETRAINING ISOLATION FOREST (7 FITUR)")
        t1 = time.perf_counter()

        # Perlakukan <= 0 pada nilai pagu sebagai NaN agar konsisten dengan min_value_missing
        nilai_min = min_val_raw.where(min_val_raw > 0, np.nan)
        nilai_kontrak = pd.to_numeric(df_ocid["value_amount"], errors="coerce")

        fitur = pd.DataFrame(index=df_ocid.index)
        fitur["log_min_value"] = np.log1p(nilai_min)
        fitur["log_award_value"] = np.log1p(nilai_kontrak)
        fitur["selisih_award_rel"] = (nilai_min - nilai_kontrak) / nilai_min
        fitur = fitur.replace([np.inf, -np.inf], np.nan)

        imputer = SimpleImputer(strategy="median")
        fitur_imp = pd.DataFrame(
            imputer.fit_transform(fitur),
            columns=["log_min_value", "log_award_value", "selisih_award_rel"],
            index=df_ocid.index,
        )
        fitur_imp["min_value_missing"] = nilai_min.isna().astype(int)

        kategori_norm = (
            df_ocid["tender_mainProcurementCategory"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        for item in ("goods", "services", "works"):
            fitur_imp[f"tender_mainProcurementCategory_{item}"] = (
                (kategori_norm == item).fillna(False).astype(int)
            )

        fitur_model_cols = [
            "log_min_value",
            "log_award_value",
            "selisih_award_rel",
            "min_value_missing",
            "tender_mainProcurementCategory_goods",
            "tender_mainProcurementCategory_services",
            "tender_mainProcurementCategory_works",
        ]
        X_train = fitur_imp[fitur_model_cols]

        model_if = IsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.01,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )
        pred_if = model_if.fit_predict(X_train)
        skor_if = -model_if.score_samples(X_train)
        t_train = time.perf_counter() - t1
        ram_after_train = get_ram_mb()

        df_ocid["log_min_value"] = fitur["log_min_value"].values
        df_ocid["log_award_value"] = fitur["log_award_value"].values
        df_ocid["selisih_award_rel"] = fitur["selisih_award_rel"].values
        df_ocid["min_value_missing"] = fitur_imp["min_value_missing"].values
        df_ocid["prediksi_anomali"] = pred_if
        df_ocid["skor_anomali"] = skor_if
        df_ocid["status_deteksi"] = np.where(pred_if == -1, "Anomali", "Normal")

        # Simpan artefak model terbaru (backup lama dulu)
        bak_model = backup_if_exists(FILE_MODEL)
        paket_baru = {
            "model": model_if,
            "imputer": imputer,
            "fitur_numerik": ["log_min_value", "log_award_value", "selisih_award_rel", "min_value_missing"],
            "fitur_model": fitur_model_cols,
            "kategori": ["services", "works", "goods"],
            "contamination": 0.01,
            "random_state": 42,
        }
        joblib.dump(paket_baru, FILE_MODEL)

        log(f, f"  - Backup model lama  : {bak_model or 'Tidak diperlukan (sudah ada)'}")
        log(f, f"  - Model baru disimpan: {FILE_MODEL}")
        log(f, f"  - Median Imputer     : {dict(zip(['log_min_value', 'log_award_value', 'selisih_award_rel'], np.round(imputer.statistics_, 6)))}")
        log(f, f"  - Offset IsolationF  : {model_if.offset_:.6f}")
        log(f, f"  - Waktu Feature+Train: {t_train:.2f} detik (RAM: {ram_after_train:.2f} MB)")

        # ------------------------------------------------------------------
        # EKSPERIMEN 2: KARAKTERISTIK NORMAL VS ANOMALI (ISOLATION FOREST)
        # ------------------------------------------------------------------
        log(f, "\n[EKSPERIMEN 2] KARAKTERISTIK HASIL ISOLATION FOREST (LEVEL OCID)")
        n_anom_if = int((df_ocid["prediksi_anomali"] == -1).sum())
        n_norm_if = n_ocid_final - n_anom_if
        log(f, f"  - Total Normal (1)   : {n_norm_if:,} ({n_norm_if/n_ocid_final*100:.2f}%)")
        log(f, f"  - Total Anomali (-1) : {n_anom_if:,} ({n_anom_if/n_ocid_final*100:.2f}%)")

        for grp_name, grp_df in [("Normal (1)", df_ocid[df_ocid["prediksi_anomali"] == 1]),
                                 ("Anomali (-1)", df_ocid[df_ocid["prediksi_anomali"] == -1])]:
            log(f, f"\n  Statistik Kelompok {grp_name} (n={len(grp_df):,}):")
            for col in ["tender_minValue_amount", "value_amount", "selisih_award_rel", "skor_anomali"]:
                s = pd.to_numeric(grp_df[col], errors="coerce").dropna()
                if len(s) > 0:
                    log(
                        f,
                        f"    * {col:24s} | Median={s.median():,.4f} | Q1={s.quantile(0.25):,.4f} | "
                        f"Q3={s.quantile(0.75):,.4f} | Max={s.max():,.4f}"
                    )
            miss_pct = grp_df["min_value_missing"].mean() * 100
            log(f, f"    * Proporsi min_value_missing = 1 : {miss_pct:.2f}%")

        log(f, "\n  Proporsi Anomali Isolation Forest per Kategori Pengadaan:")
        ct_kat = pd.crosstab(
            df_ocid["tender_mainProcurementCategory"].fillna("missing"),
            df_ocid["status_deteksi"],
        )
        for idx_kat, row in ct_kat.iterrows():
            tot_k = row.sum()
            an_k = row.get("Anomali", 0)
            log(f, f"    * {idx_kat:12s}: Anomali={an_k:,} / {tot_k:,} ({an_k/tot_k*100:.2f}%)")

        # ------------------------------------------------------------------
        # TAHAP 3 & EKSPERIMEN 3: RULE-BASED RED FLAG & UJI SENSITIVITAS
        # ------------------------------------------------------------------
        log(f, "\n[EKSPERIMEN 3] EVALUASI RULE-BASED RED FLAG & SENSITIVITAS PARAMETER")
        t2 = time.perf_counter()
        df_ocid = hitung_red_flags(df_ocid, rf_threshold=2)
        t_rf = time.perf_counter() - t2

        log(f, f"  - Waktu komputasi 4 indikator Red Flag: {t_rf:.2f} detik")
        for rf_col, label_rf in [
            ("rf_1_price_dev", "RF-1 Deviasi Harga (<1% pagu / >pagu)"),
            ("rf_2_short_duration", "RF-2 Durasi Singkat (<7 hari)"),
            ("rf_3_q4_spending", "RF-3 Pengadaan Kuartal IV (Okt-Des)"),
            ("rf_4_supplier_conc", "RF-4 Konsentrasi Penyedia (>=20%)"),
        ]:
            cnt_rf = int(df_ocid[rf_col].sum())
            log(f, f"    * {label_rf:42s}: {cnt_rf:,} ({cnt_rf/n_ocid_final*100:.2f}%)")

        log(f, "\n  Distribusi Skor Komposit Red Flag (rf_score 0..4):")
        for sc, cnt in df_ocid["rf_score"].value_counts().sort_index().items():
            log(f, f"    * rf_score = {sc}: {cnt:,} ({cnt/n_ocid_final*100:.2f}%)")

        # Sensitivitas ambang RF_Score (>=1, >=2, >=3) terhadap Isolation Forest (contamination=0.01)
        if_set = set(df_ocid.index[df_ocid["prediksi_anomali"] == -1])
        log(f, "\n  Uji Sensitivitas Ambang Red Flag (vs Isolation Forest 1%):")
        for thr in (1, 2, 3):
            rf_set_thr = set(df_ocid.index[df_ocid["rf_score"] >= thr])
            inter = len(if_set & rf_set_thr)
            union = len(if_set | rf_set_thr)
            jacc = inter / union if union > 0 else 0.0
            ovlp = inter / min(len(if_set), len(rf_set_thr)) if min(len(if_set), len(rf_set_thr)) > 0 else 0.0
            log(
                f,
                f"    * Ambang rf_score >= {thr}: Flagged={len(rf_set_thr):,} ({len(rf_set_thr)/n_ocid_final*100:.2f}%) | "
                f"Irisan={inter:,} | Jaccard={jacc:.4f} | OverlapCoeff={ovlp:.4f}"
            )

        # Sensitivitas parameter contamination Isolation Forest (0.005, 0.01, 0.02, 0.05)
        rf_set_main = set(df_ocid.index[df_ocid["rf_flag"] == 1])
        log(f, "\n  Uji Sensitivitas Parameter Contamination Isolation Forest (vs rf_score >= 2):")
        for cont in (0.005, 0.01, 0.02, 0.05):
            q_thr = np.quantile(df_ocid["skor_anomali"], 1.0 - cont)
            if_set_c = set(df_ocid.index[df_ocid["skor_anomali"] >= q_thr])
            inter_c = len(if_set_c & rf_set_main)
            union_c = len(if_set_c | rf_set_main)
            jacc_c = inter_c / union_c if union_c > 0 else 0.0
            log(
                f,
                f"    * contamination={cont:.3f} -> Anomali={len(if_set_c):,} | "
                f"Irisan dengan RF>=2: {inter_c:,} | Jaccard={jacc_c:.4f}"
            )

        # ------------------------------------------------------------------
        # EKSPERIMEN 4: MATRIKS KONTINGENSI 2x2, JACCARD, SPEARMAN & KASUS
        # ------------------------------------------------------------------
        log(f, "\n[EKSPERIMEN 4] ANALISIS IRISAN & KESEPAKATAN METODE (IF 1% VS RF >= 2)")
        both = int(((df_ocid["prediksi_anomali"] == -1) & (df_ocid["rf_flag"] == 1)).sum())
        if_only = int(((df_ocid["prediksi_anomali"] == -1) & (df_ocid["rf_flag"] == 0)).sum())
        rf_only = int(((df_ocid["prediksi_anomali"] == 1) & (df_ocid["rf_flag"] == 1)).sum())
        neither = int(((df_ocid["prediksi_anomali"] == 1) & (df_ocid["rf_flag"] == 0)).sum())

        union_main = both + if_only + rf_only
        jaccard_main = both / union_main if union_main > 0 else 0.0
        overlap_main = both / min(both + if_only, both + rf_only) if min(both + if_only, both + rf_only) > 0 else 0.0
        rho, p_val = spearmanr(df_ocid["skor_anomali"], df_ocid["rf_score"])

        log(f, "  Matriks Kontingensi 2x2:")
        log(f, f"    - Keduanya Menandai (IF=-1 & RF>=2)     : {both:,} ({both/n_ocid_final*100:.3f}%)")
        log(f, f"    - Hanya Isolation Forest (IF=-1 & RF<2) : {if_only:,} ({if_only/n_ocid_final*100:.3f}%)")
        log(f, f"    - Hanya Red Flag (IF=1 & RF>=2)         : {rf_only:,} ({rf_only/n_ocid_final*100:.3f}%)")
        log(f, f"    - Tidak Ditandai Keduanya               : {neither:,} ({neither/n_ocid_final*100:.3f}%)")
        log(f, "  Metrik Kesepakatan Non-Ground-Truth:")
        log(f, f"    - Jaccard Similarity Index              : {jaccard_main:.4f}")
        log(f, f"    - Overlap Coefficient                   : {overlap_main:.4f}")
        log(f, f"    - Korelasi Peringkat Spearman (rho)     : {rho:.4f} (p-value={p_val:.4e})")

        # Ekspor sampel studi kasus untuk analisis kualitatif artikel
        df_ocid["kelompok_komparasi"] = np.select(
            [
                (df_ocid["prediksi_anomali"] == -1) & (df_ocid["rf_flag"] == 1),
                (df_ocid["prediksi_anomali"] == -1) & (df_ocid["rf_flag"] == 0),
                (df_ocid["prediksi_anomali"] == 1) & (df_ocid["rf_flag"] == 1),
            ],
            ["Irisan (IF + RF)", "Hanya Isolation Forest", "Hanya Red Flag"],
            default="Normal Keduanya",
        )

        kasus_irisan = df_ocid[df_ocid["kelompok_komparasi"] == "Irisan (IF + RF)"].nlargest(10, "skor_anomali")
        kasus_if = df_ocid[df_ocid["kelompok_komparasi"] == "Hanya Isolation Forest"].nlargest(10, "skor_anomali")
        kasus_rf = df_ocid[df_ocid["kelompok_komparasi"] == "Hanya Red Flag"].nlargest(10, ["rf_score", "value_amount"])
        df_kasus = pd.concat([kasus_irisan, kasus_if, kasus_rf], ignore_index=True)
        df_kasus.to_csv(CASE_STUDY_FILE, index=False, encoding="utf-8-sig")
        log(f, f"  - 30 sampel studi kasus diekspor ke     : {CASE_STUDY_FILE}")

        # ------------------------------------------------------------------
        # EKSPERIMEN 5: BENCHMARK RUNTIME INFERENCE & PENYIMPANAN CSV HISTORIS
        # ------------------------------------------------------------------
        log(f, "\n[EKSPERIMEN 5] BENCHMARK WAKTU INFERENCE SKALA BATCH")
        for batch_n in (10_000, 100_000, n_ocid_final):
            sub_X = X_train.iloc[:batch_n]
            sub_df = df_ocid.iloc[:batch_n]
            tb0 = time.perf_counter()
            _ = model_if.predict(sub_X)
            _ = model_if.score_samples(sub_X)
            t_inf_if = time.perf_counter() - tb0

            tb1 = time.perf_counter()
            _ = hitung_red_flags(sub_df)
            t_inf_rf = time.perf_counter() - tb1
            log(
                f,
                f"  - Ukuran Batch {batch_n:>9,} OCID | Inference IF={t_inf_if:.3f}s | "
                f"RedFlag={t_inf_rf:.3f}s | Total={t_inf_if+t_inf_rf:.3f}s"
            )

        # Simpan CSV historis terbaru (tetap mempertahankan kolom kompatibel app.py)
        bak_csv = backup_if_exists(FILE_HASIL_CSV)
        cols_export = [
            "id", "ocid", "tender_datePublished", "tender_mainProcurementCategory",
            "tender_minValue_amount", "value_amount", "skor_anomali",
            "prediksi_anomali", "status_deteksi",
            "rf_1_price_dev", "rf_2_short_duration", "rf_3_q4_spending",
            "rf_4_supplier_conc", "rf_score", "rf_flag", "kelompok_komparasi",
        ]
        df_ocid[cols_export].to_csv(FILE_HASIL_CSV, index=False, encoding="utf-8-sig")

        total_elapsed = time.perf_counter() - t_start_all
        log(f, f"\n  - Backup CSV lama   : {bak_csv or 'Tidak diperlukan (sudah ada)'}")
        log(f, f"  - CSV baru disimpan : {FILE_HASIL_CSV} ({len(df_ocid):,} baris)")
        log(f, f"  - Total Waktu Pipeline: {total_elapsed:.2f} detik | Puncak RAM: {get_ram_mb():.2f} MB")
        log(f, "=" * 82)


if __name__ == "__main__":
    main()