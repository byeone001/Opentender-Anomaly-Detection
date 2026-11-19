import os
import json
import urllib.request
from datetime import datetime

# ==========================================================
# KONFIGURASI (Sesuaikan jika berbeda dengan lokal Anda)
# ==========================================================
OUTPUT_FILE = "hasil_audit_bukti.txt"
JOBLIB_FILENAME = "model_isolation_forest_opentender.joblib"

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",  # Default Laragon biasanya kosong
    "database": "opentender_bigdata",
}

FIREBASE_URL = "https://opentender-bigdata-default-rtdb.firebaseio.com"

IGNORE_DIRS = {
    "venv", ".venv", "env", "__pycache__", ".git",
    "node_modules", ".ipynb_checkpoints", ".idea", ".vscode"
}
ALLOWED_CODE_EXT = {".py", ".sql", ".txt", ".toml", ".yaml", ".yml", ".html", ".js"}
MAX_FILE_SIZE_KB = 500  # Abaikan file teks di atas 500 KB
# ==========================================================


def write_section(f, title):
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"BAGIAN: {title}\n")
    f.write("=" * 80 + "\n\n")


def collect_directory_tree(f, root_dir="."):
    write_section(f, "1. STRUKTUR FOLDER & DAFTAR FILE PROYEK")
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        level = current_root.replace(root_dir, "").count(os.sep)
        indent = "  " * level
        f.write(f"{indent}[DIR] {os.path.basename(current_root) or root_dir}/\n")
        sub_indent = "  " * (level + 1)
        for file in sorted(files):
            filepath = os.path.join(current_root, file)
            try:
                size_kb = os.path.getsize(filepath) / 1024
                f.write(f"{sub_indent}- {file} ({size_kb:.1f} KB)\n")
            except OSError:
                f.write(f"{sub_indent}- {file}\n")


def find_file(filename, root_dir="."):
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        if filename in files:
            return os.path.join(current_root, filename)
    return None


def inspect_joblib_model(f):
    write_section(f, "2. INSPEKSI MODEL JOBLIB")
    model_path = find_file(JOBLIB_FILENAME)
    if not model_path:
        f.write(f"[INFO] File '{JOBLIB_FILENAME}' tidak ditemukan di direktori proyek.\n")
        return

    f.write(f"Lokasi file: {model_path}\n")
    try:
        import joblib
        artifact = joblib.load(model_path)
        f.write(f"Tipe objek utama: {type(artifact)}\n\n")

        if isinstance(artifact, dict):
            f.write(f"Daftar kunci (keys): {list(artifact.keys())}\n\n")
            for k, v in artifact.items():
                f.write(f"--- Key: '{k}' (Tipe: {type(v).__name__}) ---\n")
                if hasattr(v, "get_params"):
                    f.write(f"Parameters: {v.get_params()}\n")
                    for attr in ["n_features_in_", "feature_names_in_", "offset_", "statistics_"]:
                        if hasattr(v, attr):
                            f.write(f"{attr}: {getattr(v, attr)}\n")
                else:
                    f.write(f"Value: {v}\n")
                f.write("\n")
        else:
            if hasattr(artifact, "get_params"):
                f.write(f"Parameters: {artifact.get_params()}\n")
            for attr in ["n_features_in_", "feature_names_in_", "offset_"]:
                if hasattr(artifact, attr):
                    f.write(f"{attr}: {getattr(artifact, attr)}\n")
    except Exception as e:
        f.write(f"[ERROR] Gagal membaca file joblib: {e}\n")


def inspect_mysql_database(f):
    write_section(f, "3. INSPEKSI DATABASE MYSQL (SCHEMA & JUMLAH BARIS)")
    conn = None
    driver_used = None
    try:
        try:
            import mysql.connector
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            driver_used = "mysql.connector"
        except ImportError:
            import pymysql
            conn = pymysql.connect(
                host=MYSQL_CONFIG["host"],
                port=MYSQL_CONFIG["port"],
                user=MYSQL_CONFIG["user"],
                password=MYSQL_CONFIG["password"],
                database=MYSQL_CONFIG["database"],
            )
            driver_used = "pymysql"

        f.write(f"Koneksi berhasil menggunakan driver: {driver_used}\n")
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]
        f.write(f"Daftar tabel di '{MYSQL_CONFIG['database']}': {tables}\n\n")

        for tbl in tables:
            f.write(f"--- TABEL: {tbl} ---\n")
            cursor.execute(f"SELECT COUNT(*) FROM `{tbl}`;")
            count = cursor.fetchone()[0]
            f.write(f"Jumlah baris: {count:,}\n")

            cursor.execute(f"SHOW CREATE TABLE `{tbl}`;")
            create_stmt = cursor.fetchone()[1]
            f.write(f"DDL / Struktur:\n{create_stmt}\n\n")

        cursor.close()
        conn.close()
    except Exception as e:
        f.write(f"[ERROR / INFO] Tidak dapat terhubung ke MySQL ({MYSQL_CONFIG['database']}): {e}\n")
        f.write("Pastikan service MySQL di Laragon aktif jika ingin mengaudit database.\n")


def inspect_firebase_rtdb(f):
    write_section(f, "4. INSPEKSI STRUKTUR FIREBASE REALTIME DATABASE")
    base = FIREBASE_URL.rstrip("/")
    endpoints = [
        ("Shallow Root (/opentender.json?shallow=true)", f"{base}/opentender.json?shallow=true"),
        ("Node Dashboard (/opentender/dashboard.json)", f"{base}/opentender/dashboard.json"),
        ("Node Metadata (/opentender/metadata.json)", f"{base}/opentender/metadata.json"),
        ("Shallow Statistics (/opentender/statistics.json?shallow=true)", f"{base}/opentender/statistics.json?shallow=true"),
    ]
    for label, url in endpoints:
        f.write(f"--- {label} ---\n")
        try:
            req = urllib.request.urlopen(url, timeout=10)
            data = json.loads(req.read().decode("utf-8"))
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n\n")
        except Exception as e:
            f.write(f"[ERROR] Gagal mengakses {url}: {e}\n\n")


def inspect_csv_headers(f, root_dir="."):
    write_section(f, "5. INSPEKSI HEADER & SAMPEL DATASET CSV (5 BARIS PERTAMA)")
    found_csv = False
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in sorted(files):
            if file.endswith(".csv") or file.endswith(".csv.gz"):
                found_csv = True
                filepath = os.path.join(current_root, file)
                f.write(f"--- FILE CSV: {filepath} ---\n")
                try:
                    import pandas as pd
                    df_sample = pd.read_csv(filepath, nrows=5, low_memory=False)
                    f.write(f"Jumlah Kolom: {len(df_sample.columns)}\n")
                    f.write(f"Daftar Kolom: {list(df_sample.columns)}\n")
                    f.write("Tipe Data (dari 5 baris sampel):\n")
                    f.write(df_sample.dtypes.to_string() + "\n\n")
                    f.write("Sampel 3 baris pertama:\n")
                    f.write(df_sample.head(3).to_string() + "\n\n")
                except Exception as e:
                    f.write(f"[ERROR] Gagal membaca sampel CSV: {e}\n\n")
    if not found_csv:
        f.write("[INFO] Tidak ada file .csv / .csv.gz ditemukan di dalam folder proyek ini.\n")


def extract_notebooks_and_code(f, root_dir="."):
    write_section(f, "6. EKSTRAKSI SOURCE CODE (.PY, .SQL, .IPYNB TANPA GAMBAR)")
    self_name = os.path.basename(__file__)

    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in sorted(files):
            if file in {self_name, OUTPUT_FILE}:
                continue
            filepath = os.path.join(current_root, file)
            ext = os.path.splitext(file)[1].lower()

            # Ekstrak Jupyter Notebook (kode + output teks saja, buang gambar base64)
            if ext == ".ipynb":
                f.write("\n" + "-" * 70 + "\n")
                f.write(f"NOTEBOOK: {filepath}\n")
                f.write("-" * 70 + "\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as nb_file:
                        nb = json.load(nb_file)
                    for idx, cell in enumerate(nb.get("cells", []), start=1):
                        ctype = cell.get("cell_type")
                        source = "".join(cell.get("source", []))
                        if not source.strip():
                            continue
                        f.write(f"\n[Cell {idx} - {ctype.upper()}]\n")
                        f.write(source + "\n")

                        if ctype == "code" and cell.get("outputs"):
                            f.write(f"[Output Cell {idx}]\n")
                            for out in cell["outputs"]:
                                if "text" in out:
                                    f.write("".join(out["text"])[:4000] + "\n")
                                elif "data" in out and "text/plain" in out["data"]:
                                    f.write("".join(out["data"]["text/plain"])[:4000] + "\n")
                                elif out.get("output_type") == "error":
                                    f.write(f"ERROR: {out.get('ename')}: {out.get('evalue')}\n")
                except Exception as e:
                    f.write(f"[ERROR] Gagal membaca notebook {filepath}: {e}\n")

            # Ekstrak file kode biasa
            elif ext in ALLOWED_CODE_EXT:
                try:
                    size_kb = os.path.getsize(filepath) / 1024
                    if size_kb > MAX_FILE_SIZE_KB:
                        f.write(f"\n[SKIPPED] {filepath} ({size_kb:.1f} KB melebihi batas {MAX_FILE_SIZE_KB} KB)\n")
                        continue
                    f.write("\n" + "-" * 70 + "\n")
                    f.write(f"FILE: {filepath}\n")
                    f.write("-" * 70 + "\n")
                    with open(filepath, "r", encoding="utf-8", errors="replace") as code_file:
                        f.write(code_file.read() + "\n")
                except Exception as e:
                    f.write(f"[ERROR] Gagal membaca {filepath}: {e}\n")


def main():
    print("Memulai pengumpulan bukti audit proyek...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("LAPORAN BUKTI TEKNIS UNTUK AUDIT PENELITIAN OPENTENDER\n")
        f.write(f"Waktu Ekstraksi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        collect_directory_tree(f)
        inspect_joblib_model(f)
        inspect_mysql_database(f)
        inspect_firebase_rtdb(f)
        inspect_csv_headers(f)
        extract_notebooks_and_code(f)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"Selesai! Hasil disimpan di: {OUTPUT_FILE} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()