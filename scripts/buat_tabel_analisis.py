"""
Skema tabel baru untuk menyimpan riwayat upload dan hasil analisis.

Jalankan sekali:
    python scripts/buat_tabel_analisis.py

Tabel yang dibuat:
    - upload_batches   : riwayat setiap proses upload CSV
    - analysis_results : hasil prediksi per baris tender
"""

import sys
import os

# Pastikan root proyek ada di sys.path sehingga config.py bisa diimpor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import mysql.connector
from config import MYSQL_CONFIG

QUERIES = [
    """
    CREATE TABLE IF NOT EXISTS upload_batches (
        batch_id        VARCHAR(36)  NOT NULL PRIMARY KEY,
        filename        TEXT,
        total_rows      INT          NOT NULL DEFAULT 0,
        total_normal    INT          NOT NULL DEFAULT 0,
        total_anomali   INT          NOT NULL DEFAULT 0,
        pct_anomali     DOUBLE       NULL,
        status          VARCHAR(50)  NOT NULL DEFAULT 'pending',
        uploaded_at     DATETIME     NOT NULL,
        completed_at    DATETIME     NULL,
        catatan         TEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_results (
        row_id                          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
        batch_id                        VARCHAR(36)  NOT NULL,
        tender_id                       VARCHAR(255) NULL,
        tender_ocid                     VARCHAR(255) NULL,
        tender_datePublished            DATETIME     NULL,
        tender_mainProcurementCategory  VARCHAR(100) NULL,
        tender_minValue_amount          DOUBLE       NULL,
        value_amount                    DOUBLE       NULL,
        log_min_value                   DOUBLE       NULL,
        log_award_value                 DOUBLE       NULL,
        selisih_award_rel               DOUBLE       NULL,
        skor_anomali                    DOUBLE       NULL,
        prediksi_anomali                TINYINT      NULL   COMMENT '1=Normal, -1=Anomali',
        status_deteksi                  VARCHAR(20)  NULL   COMMENT 'Normal atau Anomali',
        INDEX idx_ar_batch_id (batch_id),
        INDEX idx_ar_status (status_deteksi),
        CONSTRAINT fk_ar_batch_id FOREIGN KEY (batch_id) 
            REFERENCES upload_batches(batch_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
]

def main():
    print("=" * 55)
    print("MEMBUAT TABEL ANALISIS BARU")
    print("=" * 55)

    conn = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        print(f"Koneksi MySQL berhasil ke database '{MYSQL_CONFIG['database']}'.")

        for i, query in enumerate(QUERIES, 1):
            cursor.execute(query)
            print(f"Query {i} berhasil dieksekusi.")

        conn.commit()

        print("\nTabel yang tersedia:")
        cursor.execute("SHOW TABLES")
        for (tbl,) in cursor.fetchall():
            print(f"  - {tbl}")

        print("\nSelesai membuat tabel analisis.")

    except mysql.connector.Error as err:
        print(f"\nGAGAL: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    main()
