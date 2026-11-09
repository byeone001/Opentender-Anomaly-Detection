
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="",
    database="opentender_bigdata"
)

cursor = db.cursor()

queries = [
    """
    CREATE TABLE IF NOT EXISTS tenders (
        id VARCHAR(255),
        tag VARCHAR(100),
        date DATETIME NULL,
        ocid VARCHAR(255) PRIMARY KEY,
        language VARCHAR(20),
        initiationType VARCHAR(100),
        buyer_id VARCHAR(100),
        buyer_name TEXT,
        tender_id BIGINT,
        tender_mainProcurementCategory VARCHAR(100),
        tender_title TEXT,
        tender_datePublished DATETIME NULL,
        tender_status VARCHAR(100),
        tender_statusDetails TEXT,
        tender_value_currency VARCHAR(10),
        tender_minValue_amount DOUBLE NULL,
        tender_minValue_currency VARCHAR(10),
        tender_tenderPeriod_startDate DATETIME NULL,
        tender_procuringEntity_id BIGINT,
        tender_procuringEntity_name TEXT,
        tender_additionalProcurementCategories TEXT,
        planning_budget_id DOUBLE NULL,
        planning_budget_description TEXT,
        planning_budget_project TEXT,
        planning_budget_amount_amount DOUBLE NULL,
        planning_budget_amount_currency VARCHAR(10)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS awards (
        id VARCHAR(255) PRIMARY KEY,
        date DATETIME NULL,
        title TEXT,
        value_amount DOUBLE NULL,
        value_currency VARCHAR(10),
        main_ocid VARCHAR(255),
        main_id VARCHAR(255),
        FOREIGN KEY (main_ocid)
            REFERENCES tenders(ocid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS award_suppliers (
        id VARCHAR(255) PRIMARY KEY,
        name TEXT,
        main_ocid VARCHAR(255),
        main_id VARCHAR(255),
        awards_id VARCHAR(255),
        FOREIGN KEY (main_ocid)
            REFERENCES tenders(ocid),
        FOREIGN KEY (awards_id)
            REFERENCES awards(id)
    )
    """
]

try:
    for query in queries:
        cursor.execute(query)

    db.commit()
    print("Semua tabel berhasil dibuat!")

    cursor.execute("SHOW TABLES")
    for table in cursor.fetchall():
        print("-", table[0])

except mysql.connector.Error as error:
    print("Terjadi kesalahan:", error)

finally:
    cursor.close()
    db.close()
