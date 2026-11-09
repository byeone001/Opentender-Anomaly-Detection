# 🔎 OpenTender Anomaly Detection

Sistem deteksi anomali pada data pengadaan barang dan jasa publik (OpenTender) menggunakan algoritma **Isolation Forest** dengan antarmuka visualisasi berbasis **Streamlit**, serta integrasi database **MySQL** dan **Firebase (Realtime Database & Firestore)**.

---

## 📌 Fitur Utama

- **Deteksi Anomali Berbasis Machine Learning**: Memanfaatkan model `Isolation Forest` untuk mengidentifikasi tender dengan pola mencurigakan atau nilai yang tidak lazim.
- **Dashboard Interaktif Streamlit**: Antarmuka web untuk eksplorasi data, filter anomali, serta pengunggahan data secara langsung (mendukung format `.csv`, `.xlsx`, `.zip`, `.tar.gz`).
- **Pipeline Data & Sinkronisasi Database**:
  - Validasi konsistensi relasi antar-tabel (*awards*, *suppliers*, *main tenders*).
  - Skrip migrasi & sinkronisasi data ke **MySQL** dan **Firebase RTDB / Firestore**.

---

## 🗂️ Struktur Proyek

```text
opentender-anomaly-detection/
├── app.py                                    # Aplikasi utama dashboard Streamlit
├── model_isolation_forest_opentender.joblib  # Model machine learning pre-trained (~1.4 MB)
├── requirements.txt                          # Daftar dependensi library Python
├── .gitignore                                # Konfigurasi file yang diabaikan Git
├── credentials/
│   ├── firebase-key.example.json             # Template kunci Service Account Firebase
│   └── README.md                             # Panduan penempatan kredensial
├── dataset/
│   └── README.md                             # Informasi penempatan file dataset lokal
├── scripts/
│   ├── buat_tabel_mysql.py                   # Pembuatan skema tabel MySQL
│   ├── hitungdataset.py                      # Menghitung ringkasan & baris data
│   ├── import_mysql.py                       # Pipeline impor data ke MySQL
│   ├── import_rtdb.py                        # Pipeline sinkronisasi data ke Firebase RTDB
│   ├── inspect_dataset.py                    # Inspeksi struktur & kolom dataset
│   ├── koneksi_firebase.py                   # Modul tes koneksi Firebase/Firestore
│   ├── koneksi_mysql.py                      # Modul tes koneksi database MySQL
│   └── validasi_relasi.py                    # Validasi integritas relasi antar-dataset
├── cek_hasil_anomali.py                      # Skrip verifikasi output hasil deteksi
├── cek_model.py                              # Skrip inspeksi parameter model ML
├── cek_status.py                             # Skrip cek status pipeline/data
├── uji_dataset.py                            # Skrip pengujian format dataset
└── uji_konsistensi.py                        # Skrip pengujian konsistensi data
```

---

## 🚀 Panduan Memulai (Quick Start)

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/opentender-anomaly-detection.git
cd opentender-anomaly-detection
```

### 2. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Kredensial Firebase (Opsional jika menggunakan Firebase)

1. Dapatkan file Service Account Key dari Firebase Console (`Project Settings` > `Service Accounts`).
2. Simpan file tersebut di folder `credentials/` dengan nama `firebase-key.json`.
3. Gunakan contoh template pada [`credentials/firebase-key.example.json`](credentials/firebase-key.example.json) sebagai acuan.

> ⚠️ **Catatan Keamanan**: File `firebase-key.json` sudah otomatis diabaikan oleh `.gitignore` sehingga tidak akan terunggah ke GitHub.

### 5. Menjalankan Dashboard Streamlit

```bash
streamlit run app.py
```

Buka peramban (browser) di alamat `http://localhost:8501`.

---

## 📊 Catatan Mengenai Dataset

File dataset mentah berukuran besar (> 100 MB seperti `dataset/main.csv`, `dataset/awards.csv`, `hasil_deteksi_anomali_opentender.csv`, dan `validasi_sementara.sqlite`) **tidak disertakan** di repository ini karena batasan ukuran file di GitHub. 

Jika Anda memiliki file dataset lokal, tempatkan di dalam direktori `dataset/` sesuai dengan skema yang dibutuhkan oleh skrip pemrosesan.

---

## 🛠️ Tech Stack

- **Bahasa**: Python 3.10+
- **Machine Learning**: Scikit-Learn (`IsolationForest`), Joblib
- **Data Processing**: Pandas, NumPy
- **Frontend / Dashboard**: Streamlit
- **Database**: MySQL, Firebase (Firestore & Realtime Database), SQLite

