
import pandas as pd
import mysql.connector
import firebase_admin

from pathlib import Path
from firebase_admin import credentials, db
from datetime import datetime


# ==========================================
# 1. KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

KEY_PATH = BASE_DIR / "credentials" / "firebase-key.json"

ANOMALY_FILE = BASE_DIR / "hasil_deteksi_anomali_opentender.csv"

DATABASE_URL = (
    "https://opentender-bigdata-default-rtdb.firebaseio.com/"
)

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "opentender_bigdata"
}


# ==========================================
# 2. KONEKSI MYSQL
# ==========================================

def koneksi_mysql():
    return mysql.connector.connect(**MYSQL_CONFIG)


# ==========================================
# 3. FUNGSI MENJALANKAN QUERY
# ==========================================

def ambil_data(cursor, query):

    cursor.execute(query)

    kolom = [item[0] for item in cursor.description]

    hasil = []

    for row in cursor.fetchall():
        hasil.append(dict(zip(kolom, row)))

    return hasil


# ==========================================
# 4. KONEKSI FIREBASE RTDB
# ==========================================

def koneksi_firebase():

    if not KEY_PATH.exists():
        raise FileNotFoundError(
            f"File Firebase tidak ditemukan: {KEY_PATH}"
        )

    cred = credentials.Certificate(str(KEY_PATH))

    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": DATABASE_URL
        }
    )

    print("Koneksi Firebase Realtime Database berhasil!")


# ==========================================
# 5. MENGHITUNG STATISTIK ANOMALI
# ==========================================

def hitung_statistik_anomali():

    if not ANOMALY_FILE.exists():
        raise FileNotFoundError(
            f"File hasil deteksi tidak ditemukan: {ANOMALY_FILE}"
        )

    total_normal = 0
    total_anomali = 0

    print("Membaca hasil deteksi Isolation Forest...")

    # Membaca CSV secara bertahap agar hemat RAM
    for chunk in pd.read_csv(
        ANOMALY_FILE,
        usecols=["status_deteksi"],
        chunksize=100000
    ):

        jumlah = chunk["status_deteksi"].value_counts()

        total_normal += int(jumlah.get("Normal", 0))
        total_anomali += int(jumlah.get("Anomali", 0))

    total_data = total_normal + total_anomali

    persentase_anomali = (
        round((total_anomali / total_data) * 100, 2)
        if total_data > 0 else 0
    )

    return {
        "total_data_dianalisis": total_data,
        "total_anomalies": total_anomali,
        "normal_data": total_normal,
        "anomaly_percentage": persentase_anomali
    }


# ==========================================
# 6. PROSES PENGOLAHAN DATA MYSQL
# ==========================================

def proses_statistik(cursor):

    print("\nMenghitung total data...")

    total_tenders = ambil_data(
        cursor,
        "SELECT COUNT(*) AS total FROM tenders"
    )[0]["total"]

    total_awards = ambil_data(
        cursor,
        "SELECT COUNT(*) AS total FROM awards"
    )[0]["total"]

    total_suppliers = ambil_data(
        cursor,
        "SELECT COUNT(*) AS total FROM award_suppliers"
    )[0]["total"]


    print("Menghitung statistik tender per tahun...")

    tenders_by_year = ambil_data(
        cursor,
        """
        SELECT
            LEFT(tender_datePublished, 4) AS tahun,
            COUNT(*) AS total
        FROM tenders
        WHERE tender_datePublished IS NOT NULL
          AND LENGTH(tender_datePublished) >= 4
        GROUP BY LEFT(tender_datePublished, 4)
        ORDER BY tahun
        """
    )


    print("Menghitung statistik berdasarkan status...")

    tenders_by_status = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                tender_status,
                'Tidak diketahui'
            ) AS status,
            COUNT(*) AS total
        FROM tenders
        GROUP BY tender_status
        ORDER BY total DESC
        """
    )


    print("Menghitung 10 buyer teratas...")

    top_buyers = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                buyer_name,
                'Tidak diketahui'
            ) AS nama,
            COUNT(*) AS total_tender
        FROM tenders
        GROUP BY buyer_name
        ORDER BY total_tender DESC
        LIMIT 10
        """
    )


    print("Menghitung 10 supplier teratas...")

    top_suppliers = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                name,
                'Tidak diketahui'
            ) AS nama,
            COUNT(*) AS total_keterlibatan
        FROM award_suppliers
        GROUP BY name
        ORDER BY total_keterlibatan DESC
        LIMIT 10
        """
    )


    print("Menghitung tender berdasarkan kategori...")

    tenders_by_category = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                tender_mainProcurementCategory,
                'Tidak diketahui'
            ) AS kategori,
            COUNT(*) AS total
        FROM tenders
        GROUP BY tender_mainProcurementCategory
        ORDER BY total DESC
        """
    )


    print("Menghitung instansi pengadaan teratas...")

    top_procuring_entities = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                tender_procuringEntity_name,
                'Tidak diketahui'
            ) AS nama,
            COUNT(*) AS total_tender
        FROM tenders
        GROUP BY tender_procuringEntity_name
        ORDER BY total_tender DESC
        LIMIT 10
        """
    )


    print("Menghitung awards berdasarkan tahun...")

    awards_by_year = ambil_data(
        cursor,
        """
        SELECT
            LEFT(date, 4) AS tahun,
            COUNT(*) AS total
        FROM awards
        WHERE date IS NOT NULL
          AND LENGTH(date) >= 4
        GROUP BY LEFT(date, 4)
        ORDER BY tahun
        """
    )


    print("Menghitung awards berdasarkan mata uang...")

    awards_by_currency = ambil_data(
        cursor,
        """
        SELECT
            COALESCE(
                value_currency,
                'Tidak diketahui'
            ) AS mata_uang,
            COUNT(*) AS total
        FROM awards
        GROUP BY value_currency
        ORDER BY total DESC
        """
    )


    print("Menghitung keterlibatan supplier per tahun...")

    suppliers_by_year = ambil_data(
        cursor,
        """
        SELECT
            LEFT(t.tender_datePublished, 4) AS tahun,
            COUNT(*) AS total
        FROM award_suppliers s
        JOIN tenders t
            ON s.main_ocid = t.ocid
        WHERE t.tender_datePublished IS NOT NULL
          AND LENGTH(t.tender_datePublished) >= 4
        GROUP BY LEFT(t.tender_datePublished, 4)
        ORDER BY tahun
        """
    )


    # ======================================
    # STATISTIK ISOLATION FOREST
    # ======================================

    print("Menghitung hasil deteksi anomali...")

    anomaly_stats = hitung_statistik_anomali()


    # ======================================
    # MENYUSUN HASIL STATISTIK
    # ======================================

    return {

        "dashboard": {
            "total_tenders": int(total_tenders),
            "total_awards": int(total_awards),
            "total_suppliers": int(total_suppliers),
            "total_anomalies": anomaly_stats["total_anomalies"]
        },

        "statistics": {

            "tenders_by_year": tenders_by_year,

            "tenders_by_status": tenders_by_status,

            "top_buyers": top_buyers,

            "top_suppliers": top_suppliers,

            "tenders_by_category": tenders_by_category,

            "top_procuring_entities": top_procuring_entities,

            "awards_by_year": awards_by_year,

            "awards_by_currency": awards_by_currency,

            "suppliers_by_year": suppliers_by_year,

            "anomaly_statistics": anomaly_stats
        },

        "metadata": {
            "source": "OpenTender Indonesia - MySQL + Isolation Forest",
            "generated_at": datetime.now().astimezone().isoformat()
        }
    }


# ==========================================
# 7. KIRIM DATA KE FIREBASE
# ==========================================

def kirim_ke_firebase(data):

    referensi = db.reference("opentender")

    referensi.update(data)

    print("\nData berhasil dikirim ke Firebase!")


# ==========================================
# 8. PROGRAM UTAMA
# ==========================================

def main():

    koneksi = None
    cursor = None

    try:

        print("====================================")
        print(" OPEN TENDER - MYSQL TO FIREBASE")
        print("====================================")

        koneksi = koneksi_mysql()

        cursor = koneksi.cursor()

        print("Koneksi MySQL berhasil!")

        koneksi_firebase()

        data = proses_statistik(cursor)

        print("\nMengirim data ke Firebase...")

        kirim_ke_firebase(data)

        print("\n====================================")
        print("MIGRASI STATISTIK BERHASIL!")
        print("====================================")

        print(
            "Total Tender:",
            data["dashboard"]["total_tenders"]
        )

        print(
            "Total Awards:",
            data["dashboard"]["total_awards"]
        )

        print(
            "Total Suppliers:",
            data["dashboard"]["total_suppliers"]
        )

        print(
            "Total Anomali:",
            data["dashboard"]["total_anomalies"]
        )

        print(
            "Data Normal:",
            data["statistics"]["anomaly_statistics"]["normal_data"]
        )

        print(
            "Persentase Anomali:",
            str(
                data["statistics"]["anomaly_statistics"][
                    "anomaly_percentage"
                ]
            ) + "%"
        )

        print("\nNode yang diperbarui:")
        print("- opentender/dashboard")
        print("- opentender/statistics")
        print("- opentender/metadata")

        print("\n====================================")

    except Exception as error:

        print("\nPROSES GAGAL!")
        print("Error:", error)

    finally:

        if cursor is not None:
            cursor.close()

        if koneksi is not None and koneksi.is_connected():
            koneksi.close()


if __name__ == "__main__":
    main()
