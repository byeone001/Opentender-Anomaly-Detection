"""
Modul Rule-Based Red Flag untuk Audit Forensik Pengadaan OpenTender Indonesia.
Unit analisis: 1 baris per tender berdasarkan OCID.

Indikator terverifikasi kolom dataset aktual:
  - RF-1: Deviasi harga kontrak terhadap nilai minimum/pagu (R_harga < 0 atau 0 <= R_harga < 0.01)
  - RF-2: Durasi publikasi hingga penetapan pemenang terlalu singkat (0 <= delta_hari < 7 hari)
  - RF-3: Pengadaan dipublikasikan pada Kuartal IV akhir tahun (Oktober - Desember)
  - RF-4: Konsentrasi kemenangan penyedia >= 20% pada satu instansi pembeli (minimal 10 paket)
"""

import numpy as np
import pandas as pd


def hitung_red_flags(
    df_ocid: pd.DataFrame,
    rf_threshold: int = 2,
    min_paket_buyer: int = 10,
    konsentrasi_threshold: float = 0.20,
) -> pd.DataFrame:
    """
    Menghitung 4 indikator Rule-Based Red Flag pada DataFrame level OCID.
    Mengembalikan salinan DataFrame dengan tambahan kolom metrik & flag.
    """
    df = df_ocid.copy()

    def _get_series(col_names):
        for c in col_names:
            if c in df.columns:
                return df[c]
        return pd.Series(np.nan, index=df.index)

    # ------------------------------------------------------------------
    # RF-1: Deviasi Harga Kontrak Tidak Lazim (Over-ceiling / Near-ceiling)
    # ------------------------------------------------------------------
    min_val = pd.to_numeric(_get_series(["tender_minValue_amount", "minValue_amount"]), errors="coerce")
    awd_val = pd.to_numeric(_get_series(["value_amount", "value"]), errors="coerce")

    valid_harga = (min_val > 0) & (awd_val > 0)
    rasio_selisih = pd.Series(np.nan, index=df.index, dtype="float64")
    rasio_selisih.loc[valid_harga] = (
        (min_val.loc[valid_harga] - awd_val.loc[valid_harga])
        / min_val.loc[valid_harga]
    )

    df["rf_rasio_harga"] = rasio_selisih
    df["rf_1_price_dev"] = (
        valid_harga & ((rasio_selisih < 0) | (rasio_selisih < 0.01))
    ).astype(int)

    # ------------------------------------------------------------------
    # RF-2: Durasi Pengumuman ke Penetapan Pemenang Terlalu Singkat (< 7 hari)
    # ------------------------------------------------------------------
    tgl_pub = pd.to_datetime(_get_series(["tender_datePublished", "datePublished"]), errors="coerce", utc=True)
    tgl_awd = pd.to_datetime(_get_series(["award_date", "date", "date_award"]), errors="coerce", utc=True)

    # Filter tanggal valid (menghindari tanggal korup tahun 1970)
    valid_tgl_pub = tgl_pub.notna() & (tgl_pub.dt.year >= 2000) & (tgl_pub.dt.year <= 2026)
    valid_tgl_awd = tgl_awd.notna() & (tgl_awd.dt.year >= 2000) & (tgl_awd.dt.year <= 2026)

    durasi_hari = (tgl_awd - tgl_pub).dt.total_seconds() / 86400.0
    valid_durasi = valid_tgl_pub & valid_tgl_awd & (durasi_hari >= 0)

    df["rf_durasi_hari"] = durasi_hari.where(valid_durasi)
    df["rf_2_short_duration"] = (
        valid_durasi & (durasi_hari < 7.0)
    ).astype(int)

    # ------------------------------------------------------------------
    # RF-3: Pengadaan Menjelang Akhir Tahun Anggaran (Kuartal IV: Okt-Des)
    # ------------------------------------------------------------------
    bulan_pub = tgl_pub.dt.month.where(valid_tgl_pub)
    df["rf_bulan_publikasi"] = bulan_pub
    df["rf_3_q4_spending"] = (
        valid_tgl_pub & bulan_pub.isin([10, 11, 12])
    ).astype(int)

    # ------------------------------------------------------------------
    # RF-4: Konsentrasi Kemenangan Penyedia pada Satu Pembeli (>= 20%)
    # ------------------------------------------------------------------
    buyer_ser = _get_series(["buyer_id", "buyer", "buyer_name", "tender_procuringEntity_id"])
    supp_ser = _get_series(["supplier_id", "supplier", "supplier_name"])
    if buyer_ser.notna().any() and supp_ser.notna().any():
        buyer = buyer_ser.astype("string").str.strip()
        supp = supp_ser.astype("string").str.strip()
        valid_bs = (
            buyer.notna() & (buyer != "") & (buyer.str.lower() != "nan")
            & supp.notna() & (supp != "") & (supp.str.lower() != "nan")
        )

        df["rf_pangsa_supplier"] = np.nan
        df["rf_4_supplier_conc"] = 0

        if valid_bs.any():
            sub = pd.DataFrame({
                "buyer_id": buyer[valid_bs],
                "supplier_id": supp[valid_bs],
            }, index=df.index[valid_bs])

            total_per_buyer = sub.groupby("buyer_id")["supplier_id"].transform("count")
            menang_pair = sub.groupby(["buyer_id", "supplier_id"])["supplier_id"].transform("count")
            pangsa = menang_pair / total_per_buyer

            df.loc[valid_bs, "rf_pangsa_supplier"] = pangsa.values
            df.loc[valid_bs, "rf_4_supplier_conc"] = (
                (total_per_buyer >= min_paket_buyer) & (pangsa >= konsentrasi_threshold)
            ).astype(int).values
    else:
        df["rf_pangsa_supplier"] = np.nan
        df["rf_4_supplier_conc"] = 0

    # ------------------------------------------------------------------
    # Agregasi Skor Komposit & Status Red Flag
    # ------------------------------------------------------------------
    rf_cols = [
        "rf_1_price_dev",
        "rf_2_short_duration",
        "rf_3_q4_spending",
        "rf_4_supplier_conc",
    ]
    df["rf_score"] = df[rf_cols].sum(axis=1).astype(int)
    df["rf_flag"] = (df["rf_score"] >= rf_threshold).astype(int)
    df["status_red_flag"] = np.where(df["rf_flag"] == 1, "Red Flag", "Normal")

    return df