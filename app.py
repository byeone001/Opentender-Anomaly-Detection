"""
=============================================================================
 OpenTender — Anomaly Detection System
 Final UI/UX Refinement & Enterprise Analytics Dashboard
 
 Backend: Python, Scikit-Learn (Isolation Forest), MySQL InnoDB, Firebase RTDB
 Frontend: Streamlit + Altair + Custom Enterprise SaaS CSS
=============================================================================
"""

import io
import tarfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="OpenTender — Anomaly Detection System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 2. PATH FILE & MODEL
# =============================================================================
FILE_DATA = Path("hasil_deteksi_anomali_opentender.csv")
FILE_MODEL = Path("model_isolation_forest_opentender.joblib")

# =============================================================================
# 3. IMPORT MODUL INTEGRASI (MySQL & Firebase)
# =============================================================================
try:
    from db_mysql import (
        get_connection as mysql_conn,
        test_connection as mysql_test,
        simpan_batch,
        perbarui_status_batch,
        simpan_hasil_prediksi,
        ambil_riwayat_upload,
        ambil_detail_batch,
        ambil_hasil_analisis,
    )
    MYSQL_AVAILABLE = True
except Exception as _e:
    MYSQL_AVAILABLE = False
    _MYSQL_ERR = str(_e)

try:
    from db_firebase import (
        init_firebase,
        test_connection as firebase_test,
        kirim_ringkasan_batch,
        perbarui_dashboard as firebase_update_dashboard,
        ambil_dashboard_firebase,
        ambil_riwayat_firebase,
        ambil_statistik_firebase,
    )
    FIREBASE_AVAILABLE = True
except Exception as _e:
    FIREBASE_AVAILABLE = False
    _FIREBASE_ERR = str(_e)

try:
    from red_flag_engine import hitung_red_flags
    RED_FLAG_AVAILABLE = True
except Exception as _e:
    RED_FLAG_AVAILABLE = False
    _RED_FLAG_ERR = str(_e)


# =============================================================================
# 4. CUSTOM CSS: REFINED ENTERPRISE DESIGN SYSTEM
# =============================================================================
CUSTOM_CSS = """
<style>
/* ---- FONT & BASE LAYOUT ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0F172A;
}

.stApp {
    background-color: #F8FAFC;
}

/* Kurangi padding atas halaman Streamlit agar lebih proporsional */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px;
}

/* ---- SIDEBAR REFINEMENT ---- */
section[data-testid="stSidebar"] {
    width: 275px !important;
    min-width: 275px !important;
    max-width: 275px !important;
    background-color: #0F172A !important;
    border-right: 1px solid #1E293B !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

/* Sembunyikan lingkaran radio di sidebar secara total dan elegan */
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroupIndicator"],
section[data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] [data-testid="stRadio"] svg {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:last-child {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}

/* Navigasi menu elegan */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 3px !important;
    transition: all 0.15s ease !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    color: #CBD5E1 !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    border: 1px solid transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #CBD5E1 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover p {
    color: #FFFFFF !important;
}

/* Active Highlight Menu */
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.35) 0%, rgba(37, 99, 235, 0.08) 100%) !important;
    border-left: 3px solid #38BDF8 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Brand Box Sidebar */
.brand-box {
    padding: 8px 6px 16px 6px;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 16px;
}
.brand-title {
    font-size: 1.18rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-badge {
    font-size: 0.65rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0EA5E9, #2563EB);
    color: #FFFFFF !important;
    padding: 2px 6px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.brand-sub {
    font-size: 0.74rem;
    color: #94A3B8 !important;
    margin-top: 4px;
    line-height: 1.35;
}

/* Status Integrasi Box di Sidebar */
.status-pill-box {
    background: #1E293B;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 18px;
    border: 1px solid #334155;
}
.status-pill-title {
    font-size: 0.70rem;
    font-weight: 700;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 5px;
}
.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    padding: 3.5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.status-row:last-child {
    border-bottom: none;
}
.status-name {
    color: #CBD5E1 !important;
}
.status-val {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.74rem;
    font-weight: 600;
}
.dot-green {
    height: 7px;
    width: 7px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 5px #10B981;
}
.dot-red {
    height: 7px;
    width: 7px;
    background-color: #EF4444;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 5px #EF4444;
}

/* ---- PAGE HEADER & BREADCRUMBS ---- */
.page-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
}
.page-breadcrumb {
    font-size: 0.74rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.page-title {
    font-size: 1.50rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.02em;
}
.page-desc {
    font-size: 0.88rem;
    color: #475569;
    margin-top: 4px;
    margin-bottom: 8px;
}
.header-meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    font-size: 0.78rem;
    color: #64748B;
    border-top: 1px solid #F1F5F9;
    padding-top: 8px;
    margin-top: 8px;
}
.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #F1F5F9;
    padding: 3px 9px;
    border-radius: 6px;
    font-weight: 500;
    color: #334155;
}

/* ---- UNIFIED CONTENT CARD (ST.CONTAINER BORDER WRAPPER) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 16px 20px 14px 20px !important;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04) !important;
    margin-bottom: 16px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 10px -2px rgba(15, 23, 42, 0.06) !important;
}

/* Chart Title inside Unified Card */
.chart-header {
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
}
.chart-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.01em;
    margin: 0;
}
.chart-subtitle {
    font-size: 0.76rem;
    color: #64748B;
    margin-top: 2px;
}

/* ---- REFINED KPI CARDS ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}
@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 500px) {
    .kpi-grid { grid-template-columns: 1fr; }
}

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
    position: relative;
    overflow: hidden;
    min-height: 122px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px -2px rgba(15, 23, 42, 0.08);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}
.kpi-blue::before { background-color: #2563EB; }
.kpi-green::before { background-color: #10B981; }
.kpi-red::before { background-color: #EF4444; }
.kpi-cyan::before { background-color: #06B6D4; }

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.kpi-icon-badge {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.badge-bg-blue { background: #EFF6FF; }
.badge-bg-green { background: #ECFDF5; }
.badge-bg-red { background: #FEF2F2; }
.badge-bg-cyan { background: #ECFEFF; }

.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 2px 0;
}
.kpi-caption {
    font-size: 0.74rem;
    color: #64748B;
}

/* ---- PIPELINE STEPS (UPLOAD PAGE) ---- */
.step-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
}
@media (max-width: 768px) {
    .step-container { grid-template-columns: repeat(2, 1fr); }
}
.step-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
}
.step-num {
    font-size: 0.70rem;
    font-weight: 700;
    background: #2563EB;
    color: #FFFFFF;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 4px;
}
.step-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #0F172A;
}
.step-desc {
    font-size: 0.73rem;
    color: #64748B;
    margin-top: 2px;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    padding: 0 16px;
    font-weight: 600;
    font-size: 0.86rem;
    color: #64748B;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}

/* ---- COMPARATIVE & RED FLAG BADGES ---- */
.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-irisan {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}
.badge-if-only {
    background-color: #EDE9FE;
    color: #5B21B6;
    border: 1px solid #C4B5FD;
}
.badge-rf-only {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}
.badge-normal-both {
    background-color: #D1FAE5;
    color: #065F46;
    border: 1px solid #6EE7B7;
}

/* Red Flag Indicator Pills */
.rf-pill-active {
    background-color: #FEF2F2;
    color: #B91C1C;
    border: 1px solid #F87171;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.rf-pill-inactive {
    background-color: #F8FAFC;
    color: #94A3B8;
    border: 1px solid #E2E8F0;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Agreement KPI Box */
.agreement-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    height: 100%;
}
.agreement-val {
    font-size: 1.80rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}
.agreement-lbl {
    font-size: 0.78rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 6px;
}
.agreement-sub {
    font-size: 0.73rem;
    color: #64748B;
    margin-top: 4px;
    line-height: 1.35;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# 5. CACHED DATA & MODEL LOADERS
# =============================================================================
@st.cache_resource
def get_model(path: Path):
    return joblib.load(path)


@st.cache_data(ttl=30)
def check_mysql_health():
    if not MYSQL_AVAILABLE:
        return False, "Modul MySQL tidak ditemukan"
    try:
        ok = mysql_test()
        return ok, "Connected (127.0.0.1:3306)" if ok else "MySQL unreachable"
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30)
def check_firebase_health():
    if not FIREBASE_AVAILABLE:
        return False, "Modul Firebase tidak ditemukan"
    try:
        return firebase_test()
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30)
def get_cached_dashboard_firebase():
    if not FIREBASE_AVAILABLE:
        return None
    return ambil_dashboard_firebase()


@st.cache_data(ttl=60)
def get_cached_statistics_firebase():
    if not FIREBASE_AVAILABLE:
        return None
    return ambil_statistik_firebase()


@st.cache_data
def load_local_csv_sample(file_path: Path, nrows: Optional[int] = 50000):
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    rf_cols = [
        "rf_1_price_dev", "rf_2_short_duration", "rf_3_q4_spending",
        "rf_4_supplier_conc", "rf_score", "rf_flag"
    ]
    for c in rf_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "kelompok_komparasi" in df.columns:
        df["kelompok_komparasi"] = df["kelompok_komparasi"].fillna("Normal Keduanya").astype(str)
    return df


# =============================================================================
# 6. ML PREPROCESSING & INFERENCE (100% PRESERVED)
# =============================================================================
def baca_csv_bytes(data_bytes):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data_bytes), encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("Encoding CSV tidak dikenali.")


def baca_dataset_upload(file_upload):
    """Membaca CSV/XLSX langsung atau dalam ZIP/TAR/RAR/7Z."""
    nama = file_upload.name.lower()
    isi = file_upload.getvalue()
    hasil = {}

    def tambah(nama_internal, data_bytes):
        n = nama_internal.lower()
        if n.endswith(".csv"):
            hasil[nama_internal] = baca_csv_bytes(data_bytes)
        elif n.endswith(".xlsx"):
            hasil[nama_internal] = pd.read_excel(io.BytesIO(data_bytes))

    if nama.endswith(".csv"):
        hasil[Path(nama).name] = baca_csv_bytes(isi)
        return hasil
    if nama.endswith(".xlsx"):
        hasil[Path(nama).name] = pd.read_excel(io.BytesIO(isi))
        return hasil

    if zipfile.is_zipfile(io.BytesIO(isi)):
        with zipfile.ZipFile(io.BytesIO(isi)) as arsip:
            for item in arsip.infolist():
                if item.is_dir() or "__MACOSX/" in item.filename:
                    continue
                if Path(item.filename).name.startswith("._"):
                    continue
                if item.filename.lower().endswith((".csv", ".xlsx")):
                    tambah(item.filename, arsip.read(item))
        if hasil:
            return hasil

    if tarfile.is_tarfile(io.BytesIO(isi)):
        with tarfile.open(fileobj=io.BytesIO(isi), mode="r:*") as arsip:
            for item in arsip.getmembers():
                if not item.isfile():
                    continue
                if item.name.lower().endswith((".csv", ".xlsx")):
                    berkas = arsip.extractfile(item)
                    if berkas:
                        tambah(item.name, berkas.read())
        if hasil:
            return hasil

    if nama.endswith(".rar"):
        try:
            import rarfile
            with rarfile.RarFile(io.BytesIO(isi)) as arsip:
                for item in arsip.infolist():
                    if not item.is_dir() and item.filename.lower().endswith((".csv", ".xlsx")):
                        with arsip.open(item) as berkas:
                            tambah(item.filename, berkas.read())
            if hasil:
                return hasil
        except Exception as exc:
            raise ValueError(f"Gagal membaca file RAR: {exc}")

    if nama.endswith(".7z"):
        try:
            import py7zr
            with py7zr.SevenZipFile(io.BytesIO(isi), mode="r") as arsip:
                nama_file = arsip.getnames()
                data_file = arsip.read(nama_file)
                for n, obj in data_file.items():
                    if n.lower().endswith((".csv", ".xlsx")):
                        tambah(n, obj.read())
            if hasil:
                return hasil
        except Exception as exc:
            raise ValueError(f"Gagal membaca file 7Z: {exc}")

    raise ValueError("Tidak ditemukan berkas CSV atau format tidak didukung.")


def nama_file_sederhana(nama):
    return Path(nama.replace("\\", "/")).name.lower()


def cari_tabel(kumpulan_data, nama_target):
    """
    Mencari DataFrame dari kumpulan_data berdasarkan nama target atau daftar nama target.
    Mendukung string tunggal (misal 'main.csv') atau list/tuple (misal ['awards_suppliers.csv', 'award_suppliers.csv']).
    Mencocokkan baik nama file sederhana maupun akhiran path (misal '2023/awards_suppliers.csv').
    """
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
    """Integrasi relasional main.csv + awards.csv + awards_suppliers.csv atau format file tunggal."""
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

    # ------------------------------------------------------------------
    # Normalisasi, Agregasi, dan Penggabungan Data Penyedia (Suppliers)
    # ------------------------------------------------------------------
    if suppliers is not None and not suppliers.empty:
        supp = suppliers.copy()

        # 1. Identifikasi kolom kunci OCID pada tabel penyedia
        supp_key = None
        for cand in ("main_ocid", "ocid", "main_id", "id_main", "awards_id"):
            if cand in supp.columns:
                supp_key = cand
                break

        if supp_key:
            supp[supp_key] = supp[supp_key].astype("string").str.strip()
            # Saring baris yang tidak kosong
            supp_valid = supp[
                supp[supp_key].notna()
                & (supp[supp_key] != "")
                & (supp[supp_key].str.lower() != "nan")
            ].copy()

            if not supp_valid.empty:
                # 2. Pemetaan kolom bawaan OpenTender (id -> supplier_id, name -> supplier_name)
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

                    # 3. Lakukan agregasi groupby 1 baris per OCID unik
                    # Sesuai logika run_ocid_pipeline.py:
                    # supplier_id=("supplier_id", "first"), supplier_name=("supplier_name", "first"), jumlah_supplier=("supplier_id", "nunique")
                    supp_agg = (
                        supp_valid.groupby(supp_key, as_index=False)
                        .agg(
                            supplier_id=("supplier_id", "first"),
                            supplier_name=("supplier_name", "first"),
                            jumlah_supplier=("supplier_id", "nunique"),
                        )
                    )

                    # 4. Penggabungan (Left Join) ke hasil
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

                    # Fallback jika belum cocok
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
    """Inference Isolation Forest dengan preprocessing 7-fitur identik saat training."""
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


# =============================================================================
# 7. UI HELPER: HEADER & KPI CARDS
# =============================================================================
def render_header(title: str, subtitle: str, breadcrumb: str, meta_items: Optional[list] = None):
    meta_html = ""
    if meta_items:
        chips = "".join([f'<span class="meta-chip">{m}</span>' for m in meta_items])
        meta_html = f'<div class="header-meta-row">{chips}</div>'

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-breadcrumb">{breadcrumb}</div>
            <h1 class="page-title">{title}</h1>
            <p class="page-desc">{subtitle}</p>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(total: int, normal: int, anomali: int, pct: float, label_prefix: str = ""):
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-blue">
                <div class="kpi-header">
                    <span class="kpi-label">{label_prefix}Total Tender</span>
                    <div class="kpi-icon-badge badge-bg-blue">📁</div>
                </div>
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-caption">Populasi tender dianalisis</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-header">
                    <span class="kpi-label">{label_prefix}Tender Normal</span>
                    <div class="kpi-icon-badge badge-bg-green">✅</div>
                </div>
                <div class="kpi-value">{normal:,}</div>
                <div class="kpi-caption">Pola selisih & nilai wajar</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-header">
                    <span class="kpi-label">{label_prefix}Terindikasi Anomali</span>
                    <div class="kpi-icon-badge badge-bg-red">⚠️</div>
                </div>
                <div class="kpi-value">{anomali:,}</div>
                <div class="kpi-caption">Pola penawaran tidak biasa</div>
            </div>
            <div class="kpi-card kpi-cyan">
                <div class="kpi-header">
                    <span class="kpi-label">{label_prefix}Rasio Anomali</span>
                    <div class="kpi-icon-badge badge-bg-cyan">📊</div>
                </div>
                <div class="kpi-value">{pct:.2f}%</div>
                <div class="kpi-caption">Proporsi anomali terisolasi</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparative_kpi_cards(total_ocid: int, anom_if: int, flag_rf: int, irisan: int):
    pct_if = (anom_if / total_ocid * 100) if total_ocid > 0 else 0.0
    pct_rf = (flag_rf / total_ocid * 100) if total_ocid > 0 else 0.0
    pct_irisan = (irisan / total_ocid * 100) if total_ocid > 0 else 0.0
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-blue">
                <div class="kpi-header">
                    <span class="kpi-label">Populasi OCID</span>
                    <div class="kpi-icon-badge badge-bg-blue">🏛️</div>
                </div>
                <div class="kpi-value">{total_ocid:,}</div>
                <div class="kpi-caption">Unit analisis 1 baris/tender</div>
            </div>
            <div class="kpi-card kpi-cyan">
                <div class="kpi-header">
                    <span class="kpi-label">Anomali Isolation Forest</span>
                    <div class="kpi-icon-badge badge-bg-cyan">🤖</div>
                </div>
                <div class="kpi-value">{anom_if:,} <span style="font-size: 0.92rem; font-weight: 600; color: #0284C7;">({pct_if:.2f}%)</span></div>
                <div class="kpi-caption">Unsupervised ML (Contam 1%)</div>
            </div>
            <div class="kpi-card" style="border-left: 4px solid #F59E0B;">
                <div class="kpi-header">
                    <span class="kpi-label">Flagged Red Flag (RF ≥ 2)</span>
                    <div class="kpi-icon-badge" style="background: rgba(245, 158, 11, 0.15); color: #D97706;">🚩</div>
                </div>
                <div class="kpi-value">{flag_rf:,} <span style="font-size: 0.92rem; font-weight: 600; color: #D97706;">({pct_rf:.2f}%)</span></div>
                <div class="kpi-caption">Rule-Based Red Flag Forensik</div>
            </div>
            <div class="kpi-card kpi-red">
                <div class="kpi-header">
                    <span class="kpi-label">Irisan Keduanya (Konsensus)</span>
                    <div class="kpi-icon-badge badge-bg-red">🚨</div>
                </div>
                <div class="kpi-value">{irisan:,} <span style="font-size: 0.92rem; font-weight: 600; color: #DC2626;">({pct_irisan:.3f}%)</span></div>
                <div class="kpi-caption">Prioritas Utama Audit Forensik</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agreement_metrics(jaccard: float, overlap: float, spearman_rho: float = 0.0261):
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 14px 0;">
            <div class="agreement-card">
                <div class="agreement-val" style="color: #2563EB;">{jaccard:.4f}</div>
                <div class="agreement-lbl">Jaccard Similarity Index</div>
                <div class="agreement-sub">|IF ∩ RF| / |IF ∪ RF| (Irisan thdp Gabungan)</div>
            </div>
            <div class="agreement-card">
                <div class="agreement-val" style="color: #0EA5E9;">{overlap:.4f}</div>
                <div class="agreement-lbl">Overlap Coefficient</div>
                <div class="agreement-sub">|IF ∩ RF| / min(|IF|, |RF|) (Proporsi thdp IF)</div>
            </div>
            <div class="agreement-card">
                <div class="agreement-val" style="color: #8B5CF6;">{spearman_rho:.4f}</div>
                <div class="agreement-lbl">Korelasi Spearman (ρ)</div>
                <div class="agreement-sub">Korelasi Peringkat Skor (p-value &lt; 0.001)</div>
            </div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px; font-size: 0.82rem; color: #475569; line-height: 1.55;">
            <b>💡 Interpretasi Audit Non-Ground-Truth:</b> 
            Pada pengadaan publik tanpa label kebenaran mutlak (<i>ground-truth</i>), indeks kesepakatan Jaccard ({jaccard:.4f}) dan Overlap ({overlap:.4f}) menunjukkan bahwa <b>Isolation Forest</b> dan <b>Rule-Based Red Flag</b> bersifat <b>ortogonal dan saling melengkapi (komplementer)</b>.
            Isolation Forest mendeteksi outlier deviasi ekstrem pada distribusi harga (pagu vs penawaran), sedangkan Rule-Based Red Flag mengidentifikasi anomali prosedural dan administratif (waktu tender kilat, belanja akhir tahun Q4, dan konsentrasi vendor). 
            Paket pada <b>Irisan (IF + RF)</b> merupakan paket berisiko paling kritis yang disepakati kedua metode sebagai <b>prioritas investigasi audit forensik</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. SIDEBAR: REFINED BRAND, NAVIGATION & STATUS
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">
                <span>🔎 OpenTender</span>
                <span class="brand-badge">ML AI</span>
            </div>
            <div class="brand-sub">Anomaly Detection & Procurement Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "NAVIGASI",
        [
            "📊 Dashboard Overview",
            "🔎 Hasil Deteksi Anomali",
            "🚀 Upload & Analisis Data",
            "📋 Riwayat Analisis",
            "📈 Statistik Tender",
            "ℹ️ Informasi Sistem",
        ],
        label_visibility="collapsed",
    )

    # Health Checks
    ok_mysql, msg_mysql = check_mysql_health()
    ok_fb, msg_fb = check_firebase_health()

    dot_mysql = "dot-green" if ok_mysql else "dot-red"
    dot_fb = "dot-green" if ok_fb else "dot-red"

    st.markdown(
        f"""
        <div class="status-pill-box">
            <div class="status-pill-title">⚙️ STATUS INTEGRASI SISTEM</div>
            <div class="status-row">
                <span class="status-name">MySQL InnoDB</span>
                <span class="status-val"><span class="{dot_mysql}"></span> {'Online' if ok_mysql else 'Offline'}</span>
            </div>
            <div class="status-row">
                <span class="status-name">Firebase RTDB</span>
                <span class="status-val"><span class="{dot_fb}"></span> {'Online' if ok_fb else 'Offline'}</span>
            </div>
            <div class="status-row">
                <span class="status-name">ML Model Engine</span>
                <span class="status-val"><span class="dot-green"></span> Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size: 0.70rem; color: #64748B; text-align: center; margin-top: 16px;'>"
        "OpenTender Analytics v2.5 &bull; 2026<br>Infrastruktur Big Data & Machine Learning"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# HALAMAN 1: DASHBOARD OVERVIEW
# =============================================================================
if menu == "📊 Dashboard Overview":
    fb_dashboard = get_cached_dashboard_firebase()
    fb_stats = get_cached_statistics_firebase()

    # Tentukan nilai metrik
    if fb_dashboard:
        if fb_stats and "anomaly_statistics" in fb_stats:
            a_stat = fb_stats["anomaly_statistics"]
            total_analisis = a_stat.get("total_data_dianalisis", 1085146)
            normal_data = a_stat.get("normal_data", 1074296)
            total_anom = a_stat.get("total_anomalies", 10850)
            pct_anom = a_stat.get("anomaly_percentage", 1.0)
        else:
            total_analisis = 1085146
            normal_data = 1074296
            total_anom = 10850
            pct_anom = 1.0

        up_total = int(fb_dashboard.get("total_upload_tenders", 0) or 0)
        up_normal = int(fb_dashboard.get("total_upload_normal", 0) or 0)
        up_anom = int(fb_dashboard.get("total_upload_anomali", 0) or 0)
        up_pct = float(fb_dashboard.get("pct_upload_anomali", 0.0) or 0.0)
        last_updated = fb_dashboard.get("last_updated", "2026-11-17")
    else:
        total_analisis = 1085146
        normal_data = 1074296
        total_anom = 10850
        pct_anom = 1.0
        up_total = up_normal = up_anom = 0
        up_pct = 0.0
        last_updated = "Lokal"

    meta_info = [
        f"🕒 Diperbarui: {last_updated}",
        "🔄 Sumber: Firebase Realtime Database",
        "🤖 Model: Isolation Forest (Contamination 1%)",
        "⚡ Status: Sistem Normal",
    ]

    render_header(
        title="OpenTender Analytics Dashboard",
        subtitle="Monitoring dan deteksi anomali data pengadaan barang dan jasa pemerintah Indonesia berbasis Isolation Forest.",
        breadcrumb="OPENTENDER / OVERVIEW",
        meta_items=meta_info,
    )

    tab_hist, tab_komparasi, tab_upload = st.tabs([
        "🏛️ Populasi Data Historis",
        "⚖️ Analisis Komparatif (IF vs Red Flag)",
        "📤 Akumulasi Upload Baru",
    ])

    with tab_hist:
        render_kpi_cards(
            total=total_analisis,
            normal=normal_data,
            anomali=total_anom,
            pct=pct_anom,
            label_prefix="Historis ",
        )

        col_g1, col_g2 = st.columns([1, 1], gap="medium")

        # 1. Donut Chart - DI DALAM SATU CARD BORDER
        with col_g1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Proporsi Deteksi Anomali</div>
                        <div class="chart-subtitle">Rasio Tender Normal vs Terindikasi Anomali</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_pie = pd.DataFrame({
                    "Status": ["Normal", "Anomali"],
                    "Jumlah": [normal_data, total_anom],
                })
                donut = alt.Chart(df_pie).mark_arc(innerRadius=62, cornerRadius=4).encode(
                    theta=alt.Theta(field="Jumlah", type="quantitative"),
                    color=alt.Color(
                        field="Status",
                        type="nominal",
                        scale=alt.Scale(domain=["Normal", "Anomali"], range=["#10B981", "#EF4444"]),
                        legend=alt.Legend(title="Status Deteksi", orient="bottom"),
                    ),
                    tooltip=["Status", alt.Tooltip("Jumlah:Q", format=",d")],
                ).properties(height=250)
                st.altair_chart(donut, use_container_width=True)

        # 2. Bar Chart Kategori - DI DALAM SATU CARD BORDER
        with col_g2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Distribusi Kategori Pengadaan</div>
                        <div class="chart-subtitle">Volume tender per kategori pengadaan barang/jasa</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if fb_stats and "tenders_by_category" in fb_stats:
                    df_cat = pd.DataFrame(fb_stats["tenders_by_category"]).rename(
                        columns={"kategori": "Kategori", "total": "Jumlah"}
                    )
                    bar_cat = alt.Chart(df_cat).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                        x=alt.X("Kategori:N", title=None, sort="-y"),
                        y=alt.Y("Jumlah:Q", title="Total Tender"),
                        color=alt.Color(
                            "Kategori:N",
                            scale=alt.Scale(range=["#2563EB", "#06B6D4", "#8B5CF6", "#F59E0B"]),
                            legend=None,
                        ),
                        tooltip=["Kategori", alt.Tooltip("Jumlah:Q", format=",d")],
                    ).properties(height=250)
                    st.altair_chart(bar_cat, use_container_width=True)
                else:
                    st.info("Data kategori pengadaan tidak ditemukan di Firebase.")

        # 3. Line Chart Tren per Tahun - DI DALAM SATU CARD BORDER
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">Tren Publikasi Tender per Tahun</div>
                    <div class="chart-subtitle">Kronologi aktivitas pengadaan barang/jasa dari dataset historis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if fb_stats and "tenders_by_year" in fb_stats:
                df_year = pd.DataFrame(fb_stats["tenders_by_year"]).rename(
                    columns={"tahun": "Tahun", "total": "Jumlah"}
                )
                df_year = df_year.dropna(subset=["Tahun"])
                df_year = df_year[df_year["Tahun"].astype(str).str.len() == 4]
                chart_year = alt.Chart(df_year).mark_line(point=True, strokeWidth=3, color="#2563EB").encode(
                    x=alt.X("Tahun:O", title="Tahun Publikasi"),
                    y=alt.Y("Jumlah:Q", title="Volume Tender"),
                    tooltip=["Tahun", alt.Tooltip("Jumlah:Q", format=",d")],
                ).properties(height=260)
                st.altair_chart(chart_year, use_container_width=True)
            else:
                st.info("Data tren tahunan tidak tersedia.")

    with tab_komparasi:
        render_comparative_kpi_cards(
            total_ocid=1086425,
            anom_if=10862,
            flag_rf=50658,
            irisan=878,
        )

        render_agreement_metrics(
            jaccard=0.0145,
            overlap=0.0808,
            spearman_rho=0.0261,
        )

        col_km1, col_km2 = st.columns([1.1, 1], gap="medium")

        # 1. Matriks Kontingensi 2x2
        with col_km1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Matriks Kontingensi 2×2 Metode Deteksi</div>
                        <div class="chart-subtitle">Klasifikasi 4 kuadran komparasi Isolation Forest (1%) vs Red Flag (RF ≥ 2)</div>
                    </div>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.84rem;">
                        <thead>
                            <tr style="background: #F1F5F9; border-bottom: 2px solid #CBD5E1;">
                                <th style="padding: 10px; text-align: left; color: #334155;">Kelompok Komparasi</th>
                                <th style="padding: 10px; text-align: right; color: #334155;">Jumlah Tender</th>
                                <th style="padding: 10px; text-align: right; color: #334155;">Proporsi</th>
                                <th style="padding: 10px; text-align: center; color: #334155;">Prioritas Audit</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #FEF2F2;">
                                <td style="padding: 9px 10px; font-weight: 600; color: #991B1B;">🚨 Irisan (IF + RF)</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 700; color: #991B1B;">878</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 600; color: #991B1B;">0.081%</td>
                                <td style="padding: 9px 10px; text-align: center;"><span class="badge-tag badge-irisan">Prioritas Kritis</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #FAF5FF;">
                                <td style="padding: 9px 10px; font-weight: 600; color: #5B21B6;">🤖 Hanya Isolation Forest</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 700;">9,984</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 600;">0.919%</td>
                                <td style="padding: 9px 10px; text-align: center;"><span class="badge-tag badge-if-only">Outlier Harga</span></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #E2E8F0; background: #FFFBEB;">
                                <td style="padding: 9px 10px; font-weight: 600; color: #92400E;">🚩 Hanya Red Flag</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 700;">49,780</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 600;">4.582%</td>
                                <td style="padding: 9px 10px; text-align: center;"><span class="badge-tag badge-rf-only">Anomali Prosedur</span></td>
                            </tr>
                            <tr style="background: #F0FDF4;">
                                <td style="padding: 9px 10px; font-weight: 600; color: #065F46;">✅ Normal Keduanya</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 700;">1,025,783</td>
                                <td style="padding: 9px 10px; text-align: right; font-weight: 600;">94.418%</td>
                                <td style="padding: 9px 10px; text-align: center;"><span class="badge-tag badge-normal-both">Wajar / Rendah</span></td>
                            </tr>
                        </tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

        # 2. Donut Chart 4 Kelompok Komparasi
        with col_km2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Distribusi 4 Kelompok Komparasi</div>
                        <div class="chart-subtitle">Proporsi keterkaitan hasil klasifikasi Machine Learning vs Rule-Based</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_komp_dist = pd.DataFrame({
                    "Kelompok": [
                        "Irisan (IF + RF)",
                        "Hanya Isolation Forest",
                        "Hanya Red Flag",
                        "Normal Keduanya",
                    ],
                    "Jumlah": [878, 9984, 49780, 1025783],
                })
                donut_komp = alt.Chart(df_komp_dist).mark_arc(innerRadius=62, cornerRadius=4).encode(
                    theta=alt.Theta(field="Jumlah", type="quantitative"),
                    color=alt.Color(
                        field="Kelompok",
                        type="nominal",
                        scale=alt.Scale(
                            domain=[
                                "Irisan (IF + RF)",
                                "Hanya Isolation Forest",
                                "Hanya Red Flag",
                                "Normal Keduanya",
                            ],
                            range=["#DC2626", "#8B5CF6", "#F59E0B", "#10B981"],
                        ),
                        legend=alt.Legend(title="Kelompok Komparasi", orient="bottom"),
                    ),
                    tooltip=["Kelompok", alt.Tooltip("Jumlah:Q", format=",d")],
                ).properties(height=250)
                st.altair_chart(donut_komp, use_container_width=True)

        col_kr1, col_kr2 = st.columns([1, 1], gap="medium")

        # 3. Bar Chart 4 Indikator Red Flag
        with col_kr1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Kemunculan 4 Indikator Red Flag</div>
                        <div class="chart-subtitle">Frekuensi indikator teridentifikasi pada seluruh populasi OCID</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_rf_ind = pd.DataFrame([
                    {"Indikator": "RF-1 Deviasi Harga", "Deskripsi": "Selisih <1% pagu / >pagu", "Jumlah": 269152, "Pct": 24.77},
                    {"Indikator": "RF-3 Belanja Q4", "Deskripsi": "Publikasi Oktober - Desember", "Jumlah": 149302, "Pct": 13.74},
                    {"Indikator": "RF-2 Durasi Singkat", "Deskripsi": "Pengumuman ke penetapan <7 hari", "Jumlah": 28597, "Pct": 2.63},
                    {"Indikator": "RF-4 Konsentrasi Vendor", "Deskripsi": "Kemenangan pembeli >=20%", "Jumlah": 82, "Pct": 0.01},
                ])
                bar_rf = alt.Chart(df_rf_ind).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    y=alt.Y("Indikator:N", sort="-x", title=None),
                    x=alt.X("Jumlah:Q", title="Jumlah Tender Teridentifikasi"),
                    color=alt.Color(
                        "Indikator:N",
                        scale=alt.Scale(range=["#EF4444", "#F59E0B", "#3B82F6", "#8B5CF6"]),
                        legend=None,
                    ),
                    tooltip=[
                        "Indikator",
                        "Deskripsi",
                        alt.Tooltip("Jumlah:Q", format=",d"),
                        alt.Tooltip("Pct:Q", format=".2f", title="Persentase (%)"),
                    ],
                ).properties(height=240)
                st.altair_chart(bar_rf, use_container_width=True)

        # 4. Bar Chart Distribusi Skor Komposit Red Flag (0..4)
        with col_kr2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Distribusi Skor Komposit Red Flag (0–4)</div>
                        <div class="chart-subtitle">Jumlah indikator yang terpicu secara bersamaan per tender</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                df_rf_scores = pd.DataFrame([
                    {"Skor": "Skor 0 (Bersih)", "Jumlah": 692320, "Pct": 63.72, "Kategori": "Normal"},
                    {"Skor": "Skor 1 (Waspada)", "Jumlah": 343447, "Pct": 31.61, "Kategori": "Normal"},
                    {"Skor": "Skor 2 (Flagged)", "Jumlah": 48288, "Pct": 4.44, "Kategori": "Red Flag"},
                    {"Skor": "Skor 3 (Kritis)", "Jumlah": 2370, "Pct": 0.22, "Kategori": "Red Flag"},
                    {"Skor": "Skor 4 (Ekstrem)", "Jumlah": 0, "Pct": 0.00, "Kategori": "Red Flag"},
                ])
                bar_sc = alt.Chart(df_rf_scores).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Skor:N", title=None, sort=None),
                    y=alt.Y("Jumlah:Q", title="Total Tender"),
                    color=alt.Color(
                        "Kategori:N",
                        scale=alt.Scale(domain=["Normal", "Red Flag"], range=["#2563EB", "#DC2626"]),
                        legend=alt.Legend(title="Status Ambang (≥2)", orient="bottom"),
                    ),
                    tooltip=[
                        "Skor",
                        alt.Tooltip("Jumlah:Q", format=",d"),
                        alt.Tooltip("Pct:Q", format=".2f", title="Persentase (%)"),
                    ],
                ).properties(height=240)
                st.altair_chart(bar_sc, use_container_width=True)

    with tab_upload:
        if up_total > 0:
            render_kpi_cards(
                total=up_total,
                normal=up_normal,
                anomali=up_anom,
                pct=up_pct,
                label_prefix="Upload ",
            )
            st.caption(f"🕒 Terakhir diperbarui di Firebase: `{last_updated}`")
        else:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="text-align: center; padding: 36px 20px;">
                        <div style="font-size: 2.4rem; margin-bottom: 10px;">📤</div>
                        <h3 style="font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;">Belum Ada Analisis Upload Baru</h3>
                        <p style="color: #64748B; font-size: 0.88rem; max-width: 500px; margin: 0 auto;">
                            Data yang Anda unggah melalui menu 'Upload & Analisis Data' akan diakumulasikan di sini secara real-time.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# HALAMAN 2: HASIL DETEKSI ANOMALI
# =============================================================================
elif menu == "🔎 Hasil Deteksi Anomali":
    render_header(
        title="Hasil Deteksi Anomali & Evaluasi Forensik",
        subtitle="Eksplorasi mendalam data tender berdasarkan isolasi Machine Learning dan 4 indikator Rule-Based Red Flag forensik.",
        breadcrumb="OPENTENDER / DETEKSI ANOMALI",
        meta_items=[
            "Metode: IF (Contam 1%) + Red Flag (RF ≥ 2)",
            "Target: 7 Fitur Multivariat & 4 Aturan Forensik",
            "Unit: Level OCID (1 Baris/Tender)",
        ],
    )

    if not FILE_DATA.exists():
        st.error(f"File data historis tidak ditemukan di {FILE_DATA}.")
        st.stop()

    with st.spinner("Memuat dataset deteksi anomali komparatif..."):
        df_deteksi = load_local_csv_sample(FILE_DATA, nrows=50000)

    # Filter Toolbar di dalam card border
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-header">
                <div class="chart-title">Filter & Pencarian Tender Multivariat</div>
                <div class="chart-subtitle">Saring dataset berdasarkan kategori, status model, kelompok komparasi, dan indikator Red Flag</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        f1, f2, f3 = st.columns([1.2, 1.2, 1.6], gap="small")

        kat_options = sorted([str(x) for x in df_deteksi["tender_mainProcurementCategory"].dropna().unique()])
        with f1:
            sel_kat = st.multiselect("Kategori Pengadaan", kat_options, default=kat_options)

        with f2:
            sel_status = st.multiselect("Status Deteksi (IF)", ["Normal", "Anomali"], default=["Normal", "Anomali"])

        with f3:
            komp_options = ["Semua", "Irisan (IF + RF)", "Hanya Isolation Forest", "Hanya Red Flag", "Normal Keduanya"]
            sel_komp = st.multiselect("Kelompok Komparasi", komp_options, default=["Semua"])

        f4, f5, f6 = st.columns([1.2, 1.2, 1.6], gap="small")

        df_deteksi["tahun"] = pd.to_datetime(df_deteksi["tender_datePublished"], errors="coerce", utc=True).dt.year
        tahun_options = sorted([int(t) for t in df_deteksi["tahun"].dropna().unique()])
        with f4:
            sel_tahun = st.multiselect("Tahun Publikasi", tahun_options, default=tahun_options)

        with f5:
            rf_score_opts = ["Semua", 0, 1, 2, 3, 4]
            sel_rf_score = st.multiselect("Skor Red Flag (rf_score)", rf_score_opts, default=["Semua"])

        with f6:
            cari_id = st.text_input("Cari ID / OCID", placeholder="Ketik ID atau OCID tender...")

    filtered = df_deteksi.copy()
    if sel_kat:
        filtered = filtered[filtered["tender_mainProcurementCategory"].astype(str).isin(sel_kat)]
    if sel_status:
        filtered = filtered[filtered["status_deteksi"].isin(sel_status)]
    if sel_komp and "Semua" not in sel_komp:
        filtered = filtered[filtered["kelompok_komparasi"].isin(sel_komp)]
    if sel_tahun:
        filtered = filtered[filtered["tahun"].isin(sel_tahun)]
    if sel_rf_score and "Semua" not in sel_rf_score:
        filtered = filtered[filtered["rf_score"].isin(sel_rf_score)]
    if cari_id.strip():
        m1 = filtered["id"].astype(str).str.contains(cari_id.strip(), case=False, na=False)
        m2 = filtered["ocid"].astype(str).str.contains(cari_id.strip(), case=False, na=False) if "ocid" in filtered.columns else False
        filtered = filtered[m1 | m2]

    total_f = len(filtered)
    normal_f = int((filtered["status_deteksi"] == "Normal").sum())
    anomali_f = int((filtered["status_deteksi"] == "Anomali").sum())
    pct_f = (anomali_f / total_f * 100) if total_f > 0 else 0.0
    irisan_f = int((filtered["kelompok_komparasi"] == "Irisan (IF + RF)").sum()) if "kelompok_komparasi" in filtered.columns else 0
    rf_flag_f = int((filtered["rf_flag"] == 1).sum()) if "rf_flag" in filtered.columns else 0

    render_kpi_cards(total_f, normal_f, anomali_f, pct_f, "Filter ")

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="chart-header">
                <div class="chart-title">Daftar Paket Tender Terfilter ({total_f:,} data)</div>
                <div class="chart-subtitle">Tabel hasil evaluasi nilai pagu, penawaran, skor anomali IF, dan 4 indikator Red Flag</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tampil_cols = [
            c for c in [
                "id", "tender_datePublished", "tender_mainProcurementCategory",
                "tender_minValue_amount", "value_amount", "skor_anomali", "status_deteksi",
                "rf_score", "kelompok_komparasi",
                "rf_1_price_dev", "rf_2_short_duration", "rf_3_q4_spending", "rf_4_supplier_conc"
            ] if c in filtered.columns
        ]

        st.dataframe(
            filtered[tampil_cols].head(500),
            use_container_width=True,
            hide_index=True,
            column_config={
                "tender_minValue_amount": st.column_config.NumberColumn("Nilai Pagu (IDR)", format="Rp %d"),
                "value_amount": st.column_config.NumberColumn("Nilai Kontrak (IDR)", format="Rp %d"),
                "skor_anomali": st.column_config.NumberColumn("Skor IF", format="%.4f"),
                "status_deteksi": st.column_config.TextColumn("Status IF"),
                "rf_score": st.column_config.NumberColumn("Skor RF", format="%d"),
                "kelompok_komparasi": st.column_config.TextColumn("Kelompok Komparasi"),
                "rf_1_price_dev": st.column_config.CheckboxColumn("RF-1 Harga"),
                "rf_2_short_duration": st.column_config.CheckboxColumn("RF-2 Waktu"),
                "rf_3_q4_spending": st.column_config.CheckboxColumn("RF-3 Q4"),
                "rf_4_supplier_conc": st.column_config.CheckboxColumn("RF-4 Vendor"),
            },
        )

        st.download_button(
            label="⬇️ Unduh Data Terfilter Lengkap (CSV)",
            data=filtered[tampil_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_deteksi_anomali_filter.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Panel Inspeksi Detail Paket Pengadaan
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-header">
                <div class="chart-title">🔍 Detail Inspeksi Forensik Paket Pengadaan</div>
                <div class="chart-subtitle">Pilih salah satu tender dari hasil filter untuk meninjau status aktivasi 4 indikator Red Flag dan skor anomali</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not filtered.empty:
            list_ids = filtered["id"].head(200).tolist()
            pilihan_id = st.selectbox(
                "Pilih Tender untuk Inspeksi:",
                options=list_ids,
                format_func=lambda x: f"ID: {x} | {filtered.loc[filtered['id'] == x, 'kelompok_komparasi'].values[0] if 'kelompok_komparasi' in filtered.columns else ''}"
            )
            if pilihan_id:
                row_sel = filtered[filtered["id"] == pilihan_id].iloc[0]
                rf1 = int(row_sel.get("rf_1_price_dev", 0))
                rf2 = int(row_sel.get("rf_2_short_duration", 0))
                rf3 = int(row_sel.get("rf_3_q4_spending", 0))
                rf4 = int(row_sel.get("rf_4_supplier_conc", 0))
                rf_sc = int(row_sel.get("rf_score", 0))
                st_if = str(row_sel.get("status_deteksi", "Normal"))
                klp = str(row_sel.get("kelompok_komparasi", "Normal Keduanya"))
                min_v = float(row_sel.get("tender_minValue_amount", 0) or 0)
                awd_v = float(row_sel.get("value_amount", 0) or 0)
                skor_if_val = float(row_sel.get("skor_anomali", 0) or 0)

                # Badge CSS
                if klp == "Irisan (IF + RF)":
                    badge_klp = '<span class="badge-tag badge-irisan">🚨 IRISAN KONSENSUS (PRIORITAS KRITIS)</span>'
                elif klp == "Hanya Isolation Forest":
                    badge_klp = '<span class="badge-tag badge-if-only">🤖 HANYA ISOLATION FOREST (OUTLIER HARGA)</span>'
                elif klp == "Hanya Red Flag":
                    badge_klp = '<span class="badge-tag badge-rf-only">🚩 HANYA RED FLAG (ANOMALI PROSEDUR)</span>'
                else:
                    badge_klp = '<span class="badge-tag badge-normal-both">✅ NORMAL KEDUANYA</span>'

                pill_rf1 = '<span class="rf-pill-active">🚩 RF-1 AKTIF: Deviasi Harga (&lt;1% pagu / &gt;pagu)</span>' if rf1 else '<span class="rf-pill-inactive">RF-1 Normal: Nilai Wajar</span>'
                pill_rf2 = '<span class="rf-pill-active">⏱️ RF-2 AKTIF: Durasi Tender Singkat (&lt;7 hari)</span>' if rf2 else '<span class="rf-pill-inactive">RF-2 Normal: Durasi Standar</span>'
                pill_rf3 = '<span class="rf-pill-active">📅 RF-3 AKTIF: Pengadaan Kuartal IV (Okt-Des)</span>' if rf3 else '<span class="rf-pill-inactive">RF-3 Normal: Belanja Q1-Q3</span>'
                pill_rf4 = '<span class="rf-pill-active">🏢 RF-4 AKTIF: Konsentrasi Vendor (≥20%)</span>' if rf4 else '<span class="rf-pill-inactive">RF-4 Normal: Persaingan Terbuka</span>'

                selisih_rp = min_v - awd_v
                selisih_pct = (selisih_rp / min_v * 100) if min_v > 0 else 0.0

                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                            <div>
                                <span style="font-size: 0.76rem; font-weight: 700; color: #64748B; text-transform: uppercase;">ID Paket Tender:</span>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">{pilihan_id}</div>
                            </div>
                            <div>{badge_klp}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
                            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">NILAI PAGU</div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">Rp {min_v:,.0f}</div>
                            </div>
                            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">NILAI KONTRAK</div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">Rp {awd_v:,.0f}</div>
                            </div>
                            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">SKOR ANOMALI IF</div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: {'#EF4444' if st_if == 'Anomali' else '#10B981'};">{skor_if_val:.4f} ({st_if})</div>
                            </div>
                            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px;">
                                <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">SKOR RED FLAG</div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: {'#EF4444' if rf_sc >= 2 else '#2563EB'};">{rf_sc} / 4 Indikator</div>
                            </div>
                        </div>
                        <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 8px;">Status Aktivasi 4 Indikator Red Flag:</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            {pill_rf1}
                            {pill_rf2}
                            {pill_rf3}
                            {pill_rf4}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Tidak ada paket tender yang cocok dengan kriteria filter saat ini.")


# =============================================================================
# HALAMAN 3: UPLOAD & ANALISIS DATA
# =============================================================================
elif menu == "🚀 Upload & Analisis Data":
    render_header(
        title="Upload & Analyze Tender Data",
        subtitle="Unggah dataset pengadaan untuk melakukan validasi relasi dan deteksi anomali secara terdistribusi.",
        breadcrumb="OPENTENDER / PIPELINE ANALISIS",
        meta_items=["Format: CSV / ZIP / TAR", "Integritas: MySQL ACID", "Sinkronisasi: Firebase RTDB"],
    )

    st.markdown(
        """
        <div class="step-container" style="grid-template-columns: repeat(5, 1fr);">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-title">Pilih File Dataset</div>
                <div class="step-desc">CSV, XLSX, ZIP, TAR.GZ</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-title">Validasi Relasi</div>
                <div class="step-desc">main, awards, suppliers</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-title">Isolation Forest ML</div>
                <div class="step-desc">Scoring 7-fitur model</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-title">Rule Red Flag</div>
                <div class="step-desc">4 Indikator Forensik</div>
            </div>
            <div class="step-item">
                <div class="step-num">5</div>
                <div class="step-title">Simpan & Sinkron</div>
                <div class="step-desc">MySQL & Firebase RTDB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not FILE_MODEL.exists():
        st.error(f"Model machine learning tidak ditemukan di {FILE_MODEL}.")
        st.stop()

    paket_model = get_model(FILE_MODEL)

    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-header">
                <div class="chart-title">Unggah Berkas Dataset OpenTender</div>
                <div class="chart-subtitle">Untuk relasi lengkap, satukan main.csv, awards.csv, dan awards_suppliers.csv dalam satu arsip ZIP/TAR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        file_up = st.file_uploader(
            "Pilih file dataset",
            type=["csv", "xlsx", "zip", "tar", "gz", "tgz", "bz2", "7z", "rar"],
            label_visibility="collapsed",
        )

    if file_up is not None:
        try:
            with st.spinner("Memeriksa struktur arsip..."):
                kumpulan = baca_dataset_upload(file_up)

            st.success(f"Arsip terbaca: {len(kumpulan)} file data ditemukan.")

            with st.container(border=True):
                st.markdown(
                    """
                    <div class="chart-header">
                        <div class="chart-title">Struktur Berkas yang Diunggah</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                info_files = pd.DataFrame([
                    {"Nama File": k, "Jumlah Baris": f"{len(v):,}", "Jumlah Kolom": len(v.columns)}
                    for k, v in kumpulan.items()
                ])
                st.dataframe(info_files, use_container_width=True, hide_index=True)

            if st.button("🚀 Mulai Analisis & Deteksi Anomali", type="primary", use_container_width=True):
                batch_id = str(uuid.uuid4())
                filename = file_up.name

                st.info(f"**Batch ID Terdaftar:** `{batch_id}`")

                prog = st.progress(10, text="Langkah 1/5: Mengintegrasikan tabel relasi...")
                data_gabung = integrasikan_opentender(kumpulan)

                prog.progress(35, text="Langkah 2/5: Menjalankan model Isolation Forest (7 fitur)...")
                hasil_if = deteksi_anomali(data_gabung, paket_model)

                prog.progress(55, text="Langkah 3/5: Mengevaluasi 4 indikator Rule-Based Red Flag...")
                hasil_pred = hitung_red_flags(hasil_if, rf_threshold=2)

                # Bentuk kolom kelompok_komparasi
                hasil_pred["kelompok_komparasi"] = np.select(
                    [
                        (hasil_pred["prediksi_anomali"] == -1) & (hasil_pred["rf_flag"] == 1),
                        (hasil_pred["prediksi_anomali"] == -1) & (hasil_pred["rf_flag"] == 0),
                        (hasil_pred["prediksi_anomali"] == 1) & (hasil_pred["rf_flag"] == 1),
                    ],
                    ["Irisan (IF + RF)", "Hanya Isolation Forest", "Hanya Red Flag"],
                    default="Normal Keduanya",
                )

                total_n = len(hasil_pred)
                total_anom = int((hasil_pred["status_deteksi"] == "Anomali").sum())
                total_norm = total_n - total_anom
                pct_anom = (total_anom / total_n * 100) if total_n > 0 else 0.0
                total_rf = int((hasil_pred["rf_flag"] == 1).sum())
                total_irisan = int((hasil_pred["kelompok_komparasi"] == "Irisan (IF + RF)").sum())

                prog.progress(75, text="Langkah 4/5: Menyimpan hasil ke database MySQL...")
                if MYSQL_AVAILABLE and ok_mysql:
                    try:
                        conn_m = mysql_conn()
                        simpan_batch(conn_m, batch_id, filename, total_rows=total_n, status="processing")
                        simpan_hasil_prediksi(conn_m, batch_id, hasil_pred)
                        perbarui_status_batch(conn_m, batch_id, total_norm, total_anom, status="done")
                        conn_m.close()
                    except Exception as err_m:
                        st.error(f"Gagal simpan ke MySQL: {err_m}")

                prog.progress(90, text="Langkah 5/5: Sinkronisasi ringkasan ke Firebase RTDB...")
                if FIREBASE_AVAILABLE and ok_fb:
                    try:
                        kirim_ringkasan_batch(
                            batch_id=batch_id,
                            filename=filename,
                            total_rows=total_n,
                            total_normal=total_norm,
                            total_anomali=total_anom,
                            catatan="web_upload_pipeline",
                        )
                        firebase_update_dashboard(total_norm, total_anom)
                    except Exception as err_f:
                        st.warning(f"Gagal kirim ke Firebase: {err_f}")

                prog.progress(100, text="Selesai!")

                # Kartu Ringkasan Dual Inference
                render_comparative_kpi_cards(
                    total_ocid=total_n,
                    anom_if=total_anom,
                    flag_rf=total_rf,
                    irisan=total_irisan,
                )

                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="chart-header">
                            <div class="chart-title">Pratinjau Hasil Klasifikasi Komparatif (IF vs Red Flag)</div>
                            <div class="chart-subtitle">Tabel evaluasi dual-inference batch terbaru beserta 4 indikator forensik</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    cols_preview = [
                        c for c in [
                            "id", "tender_datePublished", "tender_mainProcurementCategory",
                            "supplier_name", "rf_pangsa_supplier",
                            "tender_minValue_amount", "value_amount", "skor_anomali", "status_deteksi",
                            "rf_score", "kelompok_komparasi",
                            "rf_1_price_dev", "rf_2_short_duration", "rf_3_q4_spending", "rf_4_supplier_conc"
                        ] if c in hasil_pred.columns
                    ]
                    st.dataframe(
                        hasil_pred[cols_preview].head(250),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "supplier_name": st.column_config.TextColumn("Penyedia"),
                            "rf_pangsa_supplier": st.column_config.NumberColumn("Pangsa Vendor", format="%.2f"),
                            "tender_minValue_amount": st.column_config.NumberColumn("Nilai Pagu (IDR)", format="Rp %d"),
                            "value_amount": st.column_config.NumberColumn("Nilai Kontrak (IDR)", format="Rp %d"),
                            "skor_anomali": st.column_config.NumberColumn("Skor IF", format="%.4f"),
                            "status_deteksi": st.column_config.TextColumn("Status IF"),
                            "rf_score": st.column_config.NumberColumn("Skor RF", format="%d"),
                            "kelompok_komparasi": st.column_config.TextColumn("Kelompok Komparasi"),
                            "rf_1_price_dev": st.column_config.CheckboxColumn("RF-1"),
                            "rf_2_short_duration": st.column_config.CheckboxColumn("RF-2"),
                            "rf_3_q4_spending": st.column_config.CheckboxColumn("RF-3"),
                            "rf_4_supplier_conc": st.column_config.CheckboxColumn("RF-4"),
                        },
                    )

                    st.download_button(
                        label="⬇️ Unduh Hasil Analisis Lengkap (CSV Komparatif)",
                        data=hasil_pred.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"hasil_anomali_komparatif_batch_{batch_id[:8]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"Terjadi kesalahan pemrosesan dataset: {exc}")


# =============================================================================
# HALAMAN 4: RIWAYAT ANALISIS
# =============================================================================
elif menu == "📋 Riwayat Analisis":
    render_header(
        title="Riwayat Sesi Analisis",
        subtitle="Log riwayat proses upload dataset dan rekonsiliasi data MySQL serta Firebase RTDB.",
        breadcrumb="OPENTENDER / RIWAYAT BATCH",
        meta_items=["Penyimpanan: Hybrid SQL & NoSQL", "Tracking: UUID Batch ID", "Audit Trail: Lengkap"],
    )

    tab_r_fb, tab_r_mysql = st.tabs(["🔥 Log Firebase RTDB", "🗄️ Log MySQL InnoDB"])

    with tab_r_fb:
        riwayat_fb = ambil_riwayat_firebase(limit=50) if FIREBASE_AVAILABLE else []
        if riwayat_fb:
            df_rfb = pd.DataFrame(riwayat_fb)
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="chart-header">
                        <div class="chart-title">Daftar Sesi Analisis Tersimpan di Firebase ({len(df_rfb)} batch)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cols_rfb = [
                    c for c in [
                        "batch_id", "filename", "total_rows", "total_normal",
                        "total_anomali", "pct_anomali", "status", "timestamp"
                    ] if c in df_rfb.columns
                ]
                st.dataframe(df_rfb[cols_rfb], use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat batch yang tercatat di Firebase.")

    with tab_r_mysql:
        if MYSQL_AVAILABLE and ok_mysql:
            try:
                conn_m = mysql_conn()
                riwayat_my = ambil_riwayat_upload(conn_m, limit=50)
                conn_m.close()

                if riwayat_my:
                    df_rmy = pd.DataFrame(riwayat_my)
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div class="chart-header">
                                <div class="chart-title">Tabel Sesi MySQL ({len(df_rmy)} batch)</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_rmy, use_container_width=True, hide_index=True)

                    with st.container(border=True):
                        st.markdown(
                            """
                            <div class="chart-header">
                                <div class="chart-title">🔍 Tinjau Detail Hasil per Batch ID</div>
                                <div class="chart-subtitle">Pilih ID sesi untuk melihat baris tender yang diisolasi</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        batch_pilihan = st.selectbox("Pilih Batch ID", df_rmy["batch_id"].tolist())
                        if batch_pilihan:
                            conn_drill = mysql_conn()
                            detail_rows = ambil_hasil_analisis(conn_drill, batch_id=batch_pilihan, limit=200)
                            conn_drill.close()

                            if detail_rows:
                                st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
                            else:
                                st.caption("Tidak ada detail baris tersimpan untuk batch ini.")
                else:
                    st.info("Tabel upload_batches di MySQL masih kosong.")
            except Exception as e:
                st.error(f"Gagal membaca log MySQL: {e}")
        else:
            st.warning("MySQL offline. Riwayat MySQL tidak dapat dimuat.")


# =============================================================================
# HALAMAN 5: STATISTIK TENDER
# =============================================================================
elif menu == "📈 Statistik Tender":
    render_header(
        title="Statistik & Intelijen Pengadaan",
        subtitle="Analisis agregat data pengadaan barang dan jasa publik Indonesia dari Firebase Realtime Database.",
        breadcrumb="OPENTENDER / STATISTIK",
        meta_items=["Sumber: Firebase RTDB", "Populasi: 1.303.374 Tenders", "3.476.240 Entitas Terdaftar"],
    )

    fb_stats = get_cached_statistics_firebase()

    if not fb_stats:
        st.warning("Data statistik tidak tersedia di Firebase Realtime Database.")
        st.stop()

    c_s1, c_s2 = st.columns([1, 1], gap="medium")

    # 1. Top Buyers
    with c_s1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">Top 10 Instansi Pembeli (Buyers)</div>
                    <div class="chart-subtitle">Lembaga dengan frekuensi pengadaan terbesar</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "top_buyers" in fb_stats:
                df_b = pd.DataFrame(fb_stats["top_buyers"])
                df_b = df_b[df_b["nama"] != "Tidak diketahui"].head(8)
                chart_b = alt.Chart(df_b).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    y=alt.Y("nama:N", sort="-x", title=None),
                    x=alt.X("total_tender:Q", title="Total Tender"),
                    color=alt.value("#2563EB"),
                    tooltip=["nama", alt.Tooltip("total_tender:Q", format=",d")],
                ).properties(height=260)
                st.altair_chart(chart_b, use_container_width=True)

    # 2. Top Suppliers
    with c_s2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">Top 10 Vendor Rekanan (Suppliers)</div>
                    <div class="chart-subtitle">Pemenang tender pengadaan dengan keterlibatan tertinggi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "top_suppliers" in fb_stats:
                df_s = pd.DataFrame(fb_stats["top_suppliers"])
                df_s = df_s[df_s["nama"] != "Tidak diketahui"].head(8)
                chart_s = alt.Chart(df_s).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    y=alt.Y("nama:N", sort="-x", title=None),
                    x=alt.X("total_keterlibatan:Q", title="Total Keterlibatan"),
                    color=alt.value("#10B981"),
                    tooltip=["nama", alt.Tooltip("total_keterlibatan:Q", format=",d")],
                ).properties(height=260)
                st.altair_chart(chart_s, use_container_width=True)

    c_s3, c_s4 = st.columns([1, 1], gap="medium")

    # 3. Status Tender
    with c_s3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">Status Penyelesaian Tender</div>
                    <div class="chart-subtitle">Perbandingan tender selesai vs tahap perencanaan</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "tenders_by_status" in fb_stats:
                df_st = pd.DataFrame(fb_stats["tenders_by_status"])
                chart_st = alt.Chart(df_st).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("total:Q"),
                    color=alt.Color("status:N", scale=alt.Scale(range=["#0EA5E9", "#F59E0B"])),
                    tooltip=["status", alt.Tooltip("total:Q", format=",d")],
                ).properties(height=240)
                st.altair_chart(chart_st, use_container_width=True)

    # 4. Satuan Kerja LPSE Teratas
    with c_s4:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">Unit Layanan Pengadaan (LPSE) Teratas</div>
                    <div class="chart-subtitle">Volume tender per satuan kerja pengadaan</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if "top_procuring_entities" in fb_stats:
                df_pe = pd.DataFrame(fb_stats["top_procuring_entities"])
                df_pe = df_pe[df_pe["nama"] != "Tidak diketahui"].head(6)
                chart_pe = alt.Chart(df_pe).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                    y=alt.Y("nama:N", sort="-x", title=None),
                    x=alt.X("total_tender:Q", title="Total Tender"),
                    color=alt.value("#8B5CF6"),
                    tooltip=["nama", alt.Tooltip("total_tender:Q", format=",d")],
                ).properties(height=240)
                st.altair_chart(chart_pe, use_container_width=True)


# =============================================================================
# HALAMAN 6: INFORMASI SISTEM
# =============================================================================
elif menu == "ℹ️ Informasi Sistem":
    render_header(
        title="Spesifikasi & Arsitektur Sistem",
        subtitle="Dokumentasi teknis algoritma Machine Learning, skema database, dan infrastruktur Big Data.",
        breadcrumb="OPENTENDER / ARSITEKTUR & SISTEM",
        meta_items=["Framework: Streamlit Enterprise", "Algoritma: Scikit-Learn IsolationForest", "Database: Hybrid MySQL & Firebase"],
    )

    c_i1, c_i2 = st.columns([1, 1], gap="medium")

    with c_i1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">🤖 Model Machine Learning: Isolation Forest</div>
                    <div class="chart-subtitle">Algoritma unsupervised outlier isolation untuk data multivariat</div>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #334155; margin-top: 10px;">
                    Model mengisolasi anomali dengan membuat partisi acak pada fitur. Titik data yang anomali
                    memerlukan jalur pohon keputusan (decision path) yang jauh lebih pendek dibandingkan titik normal.
                    <br><br>
                    <b>Fitur Model (7 Fitur Terlatih):</b>
                    <ol style="padding-left: 20px; margin-top: 6px;">
                        <li><code>log_min_value</code>: Transformasi logaritmik nilai pagu tender</li>
                        <li><code>log_award_value</code>: Transformasi logaritmik nilai kontrak pemenang</li>
                        <li><code>selisih_award_rel</code>: Rasio selisih relatif antara pagu dan penawaran</li>
                        <li><code>min_value_missing</code>: Indikator ketiadaan data nilai pagu</li>
                        <li><code>tender_mainProcurementCategory_goods</code>: One-hot barang</li>
                        <li><code>tender_mainProcurementCategory_services</code>: One-hot jasa</li>
                        <li><code>tender_mainProcurementCategory_works</code>: One-hot konstruksi</li>
                    </ol>
                    <b>Parameter Pelatihan:</b>
                    <ul>
                        <li>Contamination: <code>0.01</code> (1.0% anomali)</li>
                        <li>Estimators: <code>100 trees</code></li>
                        <li>Imputer: <code>SimpleImputer(strategy='median')</code></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_i2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">🗄️ Infrastruktur & Arsitektur Penyimpanan</div>
                    <div class="chart-subtitle">Pola hybrid SQL (MySQL) & NoSQL (Firebase Realtime Database)</div>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #334155; margin-top: 10px;">
                    <b>1. MySQL InnoDB (Relational SQL):</b>
                    <ul>
                        <li>Tabel historis: <code>tenders</code>, <code>awards</code>, <code>award_suppliers</code> (3.4M baris)</li>
                        <li>Tabel analisis baru: <code>upload_batches</code>, <code>analysis_results</code></li>
                        <li>Peran: Integritas relasional foreign key, ACID compliance, chunk storage</li>
                    </ul>
                    <b>2. Firebase Realtime Database (NoSQL):</b>
                    <ul>
                        <li>Node: <code>opentender/dashboard</code> (KPI metrik global & upload)</li>
                        <li>Node: <code>opentender/statistics</code> (Agregasi tahun, pembeli, vendor)</li>
                        <li>Node: <code>opentender/analysis_history</code> (Log per batch upload)</li>
                        <li>Peran: Sumber data cepat dashboard, sinkronisasi cloud real-time</li>
                    </ul>
                    <b>3. Streamlit & Altair Frontend Engine:</b>
                    <ul>
                        <li>Responsive grid layout dengan card wrapping terpadu</li>
                        <li>Grafik interaktif Altair dengan tooltip presisi</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    c_i3, c_i4 = st.columns([1, 1], gap="medium")

    with c_i3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">🚩 Rule-Based Red Flag Forensik</div>
                    <div class="chart-subtitle">Deteksi anomali berbasis aturan domain audit pengadaan publik</div>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #334155; margin-top: 10px;">
                    Modul ini mengevaluasi 4 indikator forensik terverifikasi pada dataset tingkat OCID:
                    <ol style="padding-left: 20px; margin-top: 6px;">
                        <li><b>RF-1 (Deviasi Harga)</b>: Rasio selisih harga negatif (&gt;pagu) atau sangat mepet pagu (&lt;1% dari pagu).</li>
                        <li><b>RF-2 (Durasi Singkat)</b>: Jarak tanggal pengumuman tender ke penetapan pemenang &lt; 7 hari kalender (tender kilat).</li>
                        <li><b>RF-3 (Belanja Akhir Tahun)</b>: Pengadaan dipublikasikan pada Kuartal IV (Oktober - Desember) yang rentan penyerapan anggaran terburu-buru.</li>
                        <li><b>RF-4 (Konsentrasi Vendor)</b>: Pangsa kemenangan penyedia ≥ 20% pada satu instansi pembeli (minimal 10 paket tender).</li>
                    </ol>
                    <b>Ambang Keputusan Flagged:</b>
                    <ul>
                        <li>Skor Komposit: <code>rf_score = RF1 + RF2 + RF3 + RF4</code> (rentang 0–4)</li>
                        <li>Ambang Deteksi: <code>rf_score ≥ 2</code> (menandakan minimal 2 indikator terpicu bersamaan)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_i4:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-header">
                    <div class="chart-title">⚖️ Metodologi Komparatif Non-Ground-Truth</div>
                    <div class="chart-subtitle">Indeks kesepakatan dan analisis 4 kuadran tanpa label verifikasi mutlak</div>
                </div>
                <div style="font-size: 0.85rem; line-height: 1.6; color: #334155; margin-top: 10px;">
                    Karena data audit publik tidak memiliki ground-truth label kepastian tindak korupsi, kedua metode diukur menggunakan metrik kesepakatan himpunan:
                    <ul style="padding-left: 20px; margin-top: 6px;">
                        <li><b>Jaccard Similarity Index</b>: <code>0.0145</code> (Rasio irisan terhadap gabungan kedua metode).</li>
                        <li><b>Overlap Coefficient</b>: <code>0.0808</code> (Rasio irisan terhadap ukuran model Isolation Forest).</li>
                        <li><b>Korelasi Spearman (ρ)</b>: <code>0.0261</code> (Korelasi non-parametrik antara skor continuous IF dan skor diskrit RF).</li>
                    </ul>
                    <b>4 Kelompok Komparasi:</b>
                    <ol style="padding-left: 20px; margin-top: 6px;">
                        <li><b style="color: #DC2626;">Irisan (IF + RF)</b>: 878 paket (0.081%) &mdash; <i>Prioritas Kritis Konsensus</i>.</li>
                        <li><b style="color: #8B5CF6;">Hanya Isolation Forest</b>: 9,984 paket (0.919%) &mdash; <i>Outlier Nilai Kontrak Ekstrem</i>.</li>
                        <li><b style="color: #F59E0B;">Hanya Red Flag</b>: 49,780 paket (4.582%) &mdash; <i>Pelanggaran Aturan Administratif</i>.</li>
                        <li><b style="color: #10B981;">Normal Keduanya</b>: 1,025,783 paket (94.418%) &mdash; <i>Pola Wajar Sesuai Kebijakan</i>.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
