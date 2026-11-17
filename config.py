import os
from pathlib import Path

# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FILE_MODEL       = BASE_DIR / "model_isolation_forest_opentender.joblib"
FILE_HASIL_CSV   = BASE_DIR / "hasil_deteksi_anomali_opentender.csv"
FILE_CREDENTIALS = BASE_DIR / "credentials" / "firebase-key.json"

# ============================================================
# KONFIGURASI MYSQL
# ============================================================

MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST",     "127.0.0.1"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "user":     os.getenv("MYSQL_USER",     "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "opentender_bigdata"),
}

# ============================================================
# KONFIGURASI FIREBASE
# ============================================================

FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://opentender-bigdata-default-rtdb.firebaseio.com/",
)

FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    str(FILE_CREDENTIALS),
)

# ============================================================
# KONSTANTA PEMROSESAN
# ============================================================

MYSQL_BATCH_SIZE    = 2_000
PREDICT_CHUNK_SIZE  = 50_000
