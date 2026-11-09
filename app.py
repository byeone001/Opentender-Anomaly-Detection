import io
import tarfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OpenTender Anomaly Detection", page_icon="🔎", layout="wide")

FILE_DATA = Path("hasil_deteksi_anomali_opentender.csv")
FILE_MODEL = Path("model_isolation_forest_opentender.joblib")


def baca_csv_bytes(data_bytes):
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data_bytes), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("Encoding CSV tidak dikenali.")


def baca_dataset_upload(file_upload):
    """Membaca CSV/XLSX langsung atau CSV/XLSX dalam ZIP/TAR."""
    nama = file_upload.name.lower()
    isi = file_upload.getvalue()
    hasil = {}

    def tambah_file(nama_internal, data_bytes):
        nama_lower = nama_internal.lower()
        if nama_lower.endswith(".csv"):
            hasil[nama_internal] = baca_csv_bytes(data_bytes)
        elif nama_lower.endswith(".xlsx"):
            hasil[nama_internal] = pd.read_excel(io.BytesIO(data_bytes))

    if nama.endswith(".csv"):
        hasil[Path(nama).name] = baca_csv_bytes(isi)
        return hasil
    if nama.endswith(".xlsx"):
        hasil[Path(nama).name] = pd.read_excel(io.BytesIO(isi))
        return hasil

    # Deteksi ZIP dari isi file, bukan ekstensi.
    if zipfile.is_zipfile(io.BytesIO(isi)):
        with zipfile.ZipFile(io.BytesIO(isi)) as arsip:
            for item in arsip.infolist():
                if item.is_dir() or "__MACOSX/" in item.filename:
                    continue
                if Path(item.filename).name.startswith("._"):
                    continue
                if item.filename.lower().endswith((".csv", ".xlsx")):
                    tambah_file(item.filename, arsip.read(item))
        if hasil:
            return hasil

    # Mendukung TAR, TAR.GZ, TGZ, TAR.BZ2, TAR.XZ, dan lainnya.
    if tarfile.is_tarfile(io.BytesIO(isi)):
        with tarfile.open(fileobj=io.BytesIO(isi), mode="r:*") as arsip:
            for item in arsip.getmembers():
                if not item.isfile():
                    continue
                if item.name.lower().endswith((".csv", ".xlsx")):
                    berkas = arsip.extractfile(item)
                    if berkas is not None:
                        tambah_file(item.name, berkas.read())
        if hasil:
            return hasil

    # RAR membutuhkan library dan backend eksternal yang sesuai.
    if nama.endswith(".rar"):
        try:
            import rarfile
        except ImportError as exc:
            raise ValueError(
                "RAR membutuhkan rarfile. Instal: python -m pip install rarfile"
            ) from exc
        try:
            with rarfile.RarFile(io.BytesIO(isi)) as arsip:
                for item in arsip.infolist():
                    if not item.is_dir() and item.filename.lower().endswith((".csv", ".xlsx")):
                        with arsip.open(item) as berkas:
                            tambah_file(item.filename, berkas.read())
        except Exception as exc:
            raise ValueError(
                "Gagal membaca RAR. Format tertentu membutuhkan unrar/unar."
            ) from exc
        if hasil:
            return hasil

    if nama.endswith(".7z"):
        try:
            import py7zr
        except ImportError as exc:
            raise ValueError(
                "7Z membutuhkan py7zr. Instal: python -m pip install py7zr"
            ) from exc
        try:
            with py7zr.SevenZipFile(io.BytesIO(isi), mode="r") as arsip:
                nama_file = arsip.getnames()
                data_file = arsip.read(nama_file)
                for nama_internal, objek in data_file.items():
                    if nama_internal.lower().endswith((".csv", ".xlsx")):
                        tambah_file(nama_internal, objek.read())
        except Exception as exc:
            raise ValueError(f"Gagal membaca arsip 7Z: {exc}") from exc
        if hasil:
            return hasil

    raise ValueError(
        "Tidak ditemukan CSV/XLSX yang dapat dibaca atau format tidak didukung. "
        "Didukung: CSV, XLSX, ZIP, TAR, TAR.GZ, TGZ, TAR.BZ2, TAR.XZ. "
        "RAR/7Z perlu library tambahan. Untuk arsip bertingkat (misalnya TAR "
        "berisi ZIP), ekstrak lapisan luarnya terlebih dahulu."
    )


def nama_file_sederhana(nama):
    return Path(nama.replace("\\", "/")).name.lower()


def cari_tabel(kumpulan_data, nama_target):
    target = nama_target.lower()
    cocok = [(nama, df) for nama, df in kumpulan_data.items()
             if nama_file_sederhana(nama) == target]
    if not cocok:
        return None
    if len(cocok) > 1:
        st.warning(f"Ditemukan beberapa file {nama_target}; memakai kemunculan pertama.")
    return cocok[0][1].copy()


def integrasikan_opentender(kumpulan_data):
    """Gabungkan main.csv dan awards.csv dengan validasi kunci relasi."""
    main = cari_tabel(kumpulan_data, "main.csv")
    awards = cari_tabel(kumpulan_data, "awards.csv")

    # CSV tunggal yang sudah memiliki kolom model dapat dipakai langsung.
    if main is None and awards is None:
        kandidat = next(iter(kumpulan_data.values())).copy()
        wajib = {"tender_minValue_amount", "value_amount", "tender_mainProcurementCategory"}
        if wajib.issubset(kandidat.columns):
            return kandidat
        raise ValueError(
            "Tidak ditemukan main.csv dan awards.csv. File tunggal harus "
            "memiliki tender_minValue_amount, value_amount, dan "
            "tender_mainProcurementCategory."
        )

    if main is None or awards is None:
        raise ValueError("Dataset OpenTender perlu menyertakan main.csv dan awards.csv.")

    wajib_main = ["id", "ocid", "tender_minValue_amount", "tender_mainProcurementCategory"]
    wajib_awards = ["value_amount", "main_id", "main_ocid"]
    kurang_main = [c for c in wajib_main if c not in main.columns]
    kurang_awards = [c for c in wajib_awards if c not in awards.columns]
    if kurang_main or kurang_awards:
        pesan = []
        if kurang_main:
            pesan.append("Kolom main.csv tidak ada: " + ", ".join(kurang_main))
        if kurang_awards:
            pesan.append("Kolom awards.csv tidak ada: " + ", ".join(kurang_awards))
        raise ValueError(" | ".join(pesan))

    for col in ("id", "ocid"):
        main[col] = main[col].astype("string").str.strip()
    for col in ("main_id", "main_ocid"):
        awards[col] = awards[col].astype("string").str.strip()

    id_main = main.loc[main["id"].notna() & main["id"].ne(""), "id"]
    id_awards = awards.loc[awards["main_id"].notna() & awards["main_id"].ne(""), "main_id"]
    ocid_main = main.loc[main["ocid"].notna() & main["ocid"].ne(""), "ocid"]
    ocid_awards = awards.loc[awards["main_ocid"].notna() & awards["main_ocid"].ne(""), "main_ocid"]

    cocok_id = len(set(id_main.unique()) & set(id_awards.unique()))
    cocok_ocid = len(set(ocid_main.unique()) & set(ocid_awards.unique()))
    if cocok_id >= cocok_ocid and cocok_id > 0:
        kunci_main, kunci_awards = "id", "main_id"
    elif cocok_ocid > 0:
        kunci_main, kunci_awards = "ocid", "main_ocid"
    else:
        raise ValueError("Tidak ada ID/OCID yang cocok antara main.csv dan awards.csv.")

    kunci_valid = main[kunci_main].notna() & main[kunci_main].ne("")
    if main.loc[kunci_valid, kunci_main].duplicated().any():
        raise ValueError(
            f"main.{kunci_main} memiliki nilai duplikat. Integrasi dihentikan "
            "agar data award tidak tergandakan secara keliru."
        )

    main["tender_minValue_amount"] = pd.to_numeric(main["tender_minValue_amount"], errors="coerce")
    awards["value_amount"] = pd.to_numeric(awards["value_amount"], errors="coerce")

    hasil = awards.merge(
        main, left_on=kunci_awards, right_on=kunci_main, how="left",
        suffixes=("_award", "_main"), validate="many_to_one", indicator=True
    )
    cocok = int((hasil["_merge"] == "both").sum())
    tidak_cocok = int((hasil["_merge"] == "left_only").sum())
    st.info(
        f"Relasi: main.{kunci_main} → awards.{kunci_awards}. "
        f"Baris award cocok: {cocok:,}; tidak cocok: {tidak_cocok:,}."
    )
    hasil = hasil.drop(columns=["_merge"])
    if hasil.empty:
        raise ValueError("Hasil integrasi kosong.")
    return hasil


def deteksi_anomali(data_baru, paket):
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

    fitur_imputasi = pd.DataFrame(
        paket["imputer"].transform(fitur),
        columns=fitur.columns,
        index=data_baru.index,
    )
    fitur_imputasi["min_value_missing"] = nilai_min.isna().astype(int)

    kategori = data_baru["tender_mainProcurementCategory"].astype("string").str.strip().str.lower()
    for item in ("goods", "services", "works"):
        fitur_imputasi[f"tender_mainProcurementCategory_{item}"] = (
            kategori == item
        ).fillna(False).astype(int)

    fitur_model = fitur_imputasi.reindex(columns=paket["fitur_model"])
    if fitur_model.isna().any().any():
        raise ValueError("Fitur model masih mengandung nilai kosong setelah preprocessing.")

    prediksi = paket["model"].predict(fitur_model)
    skor = paket["model"].score_samples(fitur_model)
    hasil = data_baru.copy()
    hasil["prediksi_anomali"] = prediksi
    hasil["skor_anomali"] = -skor
    hasil["status_deteksi"] = np.where(prediksi == -1, "Anomali", "Normal")
    return hasil


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, low_memory=False)


# =========================================================
# Navigasi dan halaman
# =========================================================
st.title("🔎 OpenTender Anomaly Detection")
halaman = st.sidebar.radio("Navigasi", ["Dashboard Hasil Deteksi", "Deteksi Dataset Baru"])

if halaman == "Dashboard Hasil Deteksi":
    st.write("Dashboard untuk mengeksplorasi pola tender yang tidak biasa menggunakan Isolation Forest.")

    if not FILE_DATA.exists():
        st.error(f"File dataset tidak ditemukan: {FILE_DATA}")
        st.stop()

    df = load_data(str(FILE_DATA))
    kolom_dashboard = [
        "id", "tender_datePublished", "tender_mainProcurementCategory",
        "tender_minValue_amount", "value_amount", "skor_anomali",
        "prediksi_anomali", "status_deteksi",
    ]
    kurang = [c for c in kolom_dashboard if c not in df.columns]
    if kurang:
        st.error("Kolom dashboard tidak ditemukan: " + ", ".join(kurang))
        st.stop()

    df["tender_datePublished"] = pd.to_datetime(df["tender_datePublished"], errors="coerce", utc=True)
    df["status_deteksi"] = df["status_deteksi"].astype(str)

    total = len(df)
    normal = int((df["status_deteksi"] == "Normal").sum())
    anomali = int((df["status_deteksi"] == "Anomali").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tender", f"{total:,}")
    c2.metric("Tender Normal", f"{normal:,}")
    c3.metric("Terindikasi Anomali", f"{anomali:,}")
    c4.metric("Persentase Anomali", f"{anomali / total * 100:.2f}%" if total else "0%")

    st.divider()
    st.subheader("Filter Data Tender")
    f1, f2, f3 = st.columns(3)
    kategori_opsi = sorted(df["tender_mainProcurementCategory"].dropna().unique().tolist())
    with f1:
        kategori = st.multiselect("Kategori pengadaan", kategori_opsi, default=kategori_opsi)
    with f2:
        status = st.multiselect("Status deteksi", ["Normal", "Anomali"], default=["Normal", "Anomali"])
    with f3:
        tahun_opsi = sorted(df["tender_datePublished"].dt.year.dropna().unique().tolist())
        tahun = st.multiselect("Tahun publikasi", tahun_opsi, default=tahun_opsi)

    hasil_filter = df[
        df["tender_mainProcurementCategory"].isin(kategori)
        & df["status_deteksi"].isin(status)
        & df["tender_datePublished"].dt.year.isin(tahun)
    ].copy()
    st.caption(f"Menampilkan {len(hasil_filter):,} dari {total:,} baris.")
    st.subheader("Distribusi Status Deteksi")
    st.bar_chart(hasil_filter["status_deteksi"].value_counts().reindex(["Normal", "Anomali"], fill_value=0))

    st.subheader("Anomali per Tahun")
    anomali_tahun = hasil_filter[hasil_filter["status_deteksi"] == "Anomali"].copy()
    anomali_tahun["Tahun"] = anomali_tahun["tender_datePublished"].dt.year
    grafik_tahun = anomali_tahun.groupby("Tahun").size().rename("Jumlah Anomali")
    if not grafik_tahun.empty:
        st.line_chart(grafik_tahun)
    else:
        st.info("Tidak ada data anomali untuk filter yang dipilih.")

    st.subheader("Data Tender")
    cari = st.text_input("Cari berdasarkan ID tender", placeholder="Masukkan ID tender...")
    if cari.strip():
        hasil_filter = hasil_filter[hasil_filter["id"].astype(str).str.contains(cari.strip(), case=False, na=False)]
    st.dataframe(hasil_filter[kolom_dashboard], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Unduh hasil filter (CSV)",
        data=hasil_filter[kolom_dashboard].to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_deteksi_opentender.csv",
        mime="text/csv",
    )
    st.caption("Anomali adalah indikasi pola tidak biasa, bukan bukti korupsi atau kecurangan.")

else:
    st.subheader("Deteksi Anomali pada Dataset Baru")
    st.write(
        "Unggah CSV/Excel tunggal atau arsip yang berisi dataset OpenTender. "
        "Untuk data relasional OpenTender, sertakan main.csv dan awards.csv."
    )
    if not FILE_MODEL.exists():
        st.error(f"File model tidak ditemukan: {FILE_MODEL}")
        st.stop()
    paket = load_model(str(FILE_MODEL))

    file_upload = st.file_uploader(
        "Unggah dataset OpenTender",
        type=["csv", "xlsx", "zip", "tar", "gz", "tgz", "bz2", "tbz2", "xz", "txz", "rar", "7z"],
        help="Format arsip bertingkat seperti TAR berisi ZIP perlu diekstrak terlebih dahulu. RAR/7Z memerlukan library tambahan.",
    )

    if file_upload is not None:
        try:
            with st.spinner("Membaca dataset..."):
                kumpulan_data = baca_dataset_upload(file_upload)
            st.success(f"Berhasil membaca {len(kumpulan_data)} file data.")
            ringkasan = pd.DataFrame([
                {"Nama file": nama, "Jumlah baris": len(data), "Jumlah kolom": len(data.columns)}
                for nama, data in kumpulan_data.items()
            ])
            st.dataframe(ringkasan, use_container_width=True, hide_index=True)
            with st.expander("Lihat kolom setiap file"):
                for nama, data in kumpulan_data.items():
                    st.write(f"**{nama}** — {len(data):,} baris")
                    st.write(data.columns.tolist())

            if st.button("Siapkan Dataset dan Mulai Deteksi", type="primary"):
                with st.spinner("Mengintegrasikan data dan menjalankan model..."):
                    data_baru = integrasikan_opentender(kumpulan_data)
                    hasil = deteksi_anomali(data_baru, paket)

                total_baru = len(hasil)
                jumlah_anomali = int((hasil["status_deteksi"] == "Anomali").sum())
                jumlah_normal = total_baru - jumlah_anomali
                st.success("Deteksi selesai!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Baris Dinilai", f"{total_baru:,}")
                c2.metric("Normal", f"{jumlah_normal:,}")
                c3.metric("Terindikasi Anomali", f"{jumlah_anomali:,}")
                st.subheader("Hasil Deteksi")
                st.dataframe(hasil, use_container_width=True, hide_index=True)
                st.download_button(
                    "Unduh Hasil Deteksi (CSV)",
                    data=hasil.to_csv(index=False).encode("utf-8-sig"),
                    file_name="hasil_deteksi_dataset_baru.csv",
                    mime="text/csv",
                )
                st.warning("Anomali menunjukkan pola tidak biasa, bukan bukti korupsi atau kecurangan.")
        except Exception as exc:
            st.error(f"Terjadi kesalahan saat memproses dataset: {exc}")
