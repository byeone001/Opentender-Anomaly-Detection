"""
Modul database MySQL untuk proyek OpenTender Anomaly Detection.

Fungsi utama:
    - get_connection()           : membuka koneksi MySQL
    - test_connection()          : menguji koneksi MySQL
    - simpan_batch()             : catat riwayat upload ke upload_batches
    - simpan_hasil_prediksi()    : simpan baris hasil prediksi ke analysis_results
    - perbarui_status_batch()    : tandai batch selesai / gagal
    - ambil_riwayat_upload()     : ambil daftar batch untuk ditampilkan di dashboard
"""

import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from config import MYSQL_CONFIG, MYSQL_BATCH_SIZE


# ============================================================
# KONEKSI
# ============================================================

def get_connection():
    """Membuka dan mengembalikan koneksi MySQL."""
    return mysql.connector.connect(**MYSQL_CONFIG)


def test_connection() -> bool:
    """Menguji koneksi MySQL. Mengembalikan True jika berhasil."""
    try:
        cfg = MYSQL_CONFIG.copy()
        cfg["connection_timeout"] = 2
        conn = mysql.connector.connect(**cfg)
        ok = conn.is_connected()
        conn.close()
        return ok
    except Exception:
        return False


# ============================================================
# UPLOAD BATCHES
# ============================================================

def simpan_batch(
    conn,
    batch_id: str,
    filename: str,
    total_rows: int = 0,
    status: str = "processing",
) -> None:
    """Mencatat entri baru ke tabel upload_batches."""
    sql = """
        INSERT INTO upload_batches
            (batch_id, filename, total_rows, total_normal, total_anomali,
             pct_anomali, status, uploaded_at)
        VALUES (%s, %s, %s, 0, 0, NULL, %s, %s)
        ON DUPLICATE KEY UPDATE
            filename   = VALUES(filename),
            total_rows = VALUES(total_rows),
            status     = VALUES(status)
    """
    cursor = conn.cursor()
    cursor.execute(sql, (batch_id, filename, total_rows, status, datetime.now()))
    conn.commit()
    cursor.close()


def perbarui_status_batch(
    conn,
    batch_id: str,
    total_normal: int,
    total_anomali: int,
    status: str = "done",
    catatan: Optional[str] = None,
) -> None:
    """Memperbarui statistik dan status akhir sebuah batch."""
    total = total_normal + total_anomali
    pct = round(total_anomali / total * 100, 2) if total > 0 else 0.0
    sql = """
        UPDATE upload_batches
        SET total_rows    = %s,
            total_normal  = %s,
            total_anomali = %s,
            pct_anomali   = %s,
            status        = %s,
            completed_at  = %s,
            catatan       = %s
        WHERE batch_id = %s
    """
    cursor = conn.cursor()
    cursor.execute(sql, (
        total, total_normal, total_anomali, pct,
        status, datetime.now(), catatan, batch_id,
    ))
    conn.commit()
    cursor.close()


def ambil_riwayat_upload(conn, limit: int = 20) -> list[dict]:
    """Mengambil daftar riwayat upload terbaru."""
    sql = """
        SELECT batch_id, filename, total_rows, total_normal,
               total_anomali, pct_anomali, status,
               uploaded_at, completed_at
        FROM   upload_batches
        ORDER BY uploaded_at DESC
        LIMIT %s
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, (limit,))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def ambil_detail_batch(conn, batch_id: str) -> Optional[dict]:
    """Mengambil informasi detail satu batch dari upload_batches."""
    sql = "SELECT * FROM upload_batches WHERE batch_id = %s"
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, (batch_id,))
    row = cursor.fetchone()
    cursor.close()
    return row


def ambil_hasil_analisis(conn, batch_id: Optional[str] = None, limit: int = 2000) -> list[dict]:
    """Mengambil baris hasil analisis dari analysis_results."""
    if batch_id:
        sql = """
            SELECT row_id, batch_id, tender_id, tender_ocid, tender_datePublished,
                   tender_mainProcurementCategory, tender_minValue_amount, value_amount,
                   skor_anomali, prediksi_anomali, status_deteksi
            FROM analysis_results
            WHERE batch_id = %s
            ORDER BY row_id ASC
            LIMIT %s
        """
        params = (batch_id, limit)
    else:
        sql = """
            SELECT row_id, batch_id, tender_id, tender_ocid, tender_datePublished,
                   tender_mainProcurementCategory, tender_minValue_amount, value_amount,
                   skor_anomali, prediksi_anomali, status_deteksi
            FROM analysis_results
            ORDER BY row_id DESC
            LIMIT %s
        """
        params = (limit,)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cursor.close()
    return rows


# ============================================================
# HASIL PREDIKSI
# ============================================================

def _safe(val):
    """Konversi nilai numpy/pandas ke tipe Python biasa untuk MySQL."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        return None if (np.isnan(val) or np.isinf(val)) else float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    return None if s == "" or s.lower() == "nan" else s


def simpan_hasil_prediksi(
    conn,
    batch_id: str,
    df_hasil: pd.DataFrame,
    batch_size: int = MYSQL_BATCH_SIZE,
) -> tuple[int, int]:
    """
    Menyimpan DataFrame hasil prediksi ke tabel analysis_results.
    Mengembalikan (total_normal, total_anomali).
    """
    sql = """
        INSERT INTO analysis_results (
            batch_id,
            tender_id, tender_ocid,
            tender_datePublished,
            tender_mainProcurementCategory,
            tender_minValue_amount,
            value_amount,
            log_min_value, log_award_value, selisih_award_rel,
            skor_anomali,
            prediksi_anomali,
            status_deteksi
        ) VALUES (
            %s,
            %s, %s,
            %s,
            %s,
            %s,
            %s,
            %s, %s, %s,
            %s,
            %s,
            %s
        )
    """

    def _get(row, *keys, default=None):
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                return row[k]
        return default

    total_normal  = 0
    total_anomali = 0
    total_saved   = 0

    cursor = conn.cursor()

    for start in range(0, len(df_hasil), batch_size):
        chunk = df_hasil.iloc[start : start + batch_size]
        records = []

        for _, row in chunk.iterrows():
            status = str(row.get("status_deteksi", "")) if "status_deteksi" in row.index else ""
            if status == "Normal":
                total_normal += 1
            elif status == "Anomali":
                total_anomali += 1

            # Parse tanggal
            tgl = _get(row, "tender_datePublished", "date")
            if tgl is not None:
                try:
                    tgl = pd.to_datetime(tgl, utc=True).tz_localize(None).to_pydatetime()
                except Exception:
                    tgl = None

            records.append((
                batch_id,
                _safe(_get(row, "id", "main_id", "id_main")),
                _safe(_get(row, "ocid", "main_ocid", "ocid_main")),
                tgl,
                _safe(_get(row, "tender_mainProcurementCategory")),
                _safe(_get(row, "tender_minValue_amount")),
                _safe(_get(row, "value_amount")),
                _safe(_get(row, "log_min_value")),
                _safe(_get(row, "log_award_value")),
                _safe(_get(row, "selisih_award_rel")),
                _safe(_get(row, "skor_anomali")),
                _safe(_get(row, "prediksi_anomali")),
                status if status else None,
            ))

        try:
            cursor.executemany(sql, records)
            conn.commit()
            total_saved += len(records)
        except Exception:
            conn.rollback()
            cursor.close()
            raise

    cursor.close()
    return total_normal, total_anomali
