# 🔎 OpenTender — Hybrid Anomaly Detection & Forensic Procurement Analytics

Sistem intelijen dan audit forensik pengadaan barang/jasa publik (OpenTender Indonesia) menggunakan pendekatan **Dual-Engine Hibrida**: **Unsupervised Machine Learning (Isolation Forest)** dikombinasikan dengan **Rule-Based Red Flag**, terintegrasi penuh dengan arsitektur penyimpanan Big Data **MySQL InnoDB (Relational SQL)** dan **Firebase Realtime Database (NoSQL Cloud)**, serta divisualisasikan melalui **Enterprise Analytics Dashboard (Streamlit)**.

---

## 📌 Fitur Utama Sistem

1. **Dual-Engine Deteksi Anomali Pengadaan:**
   - **Machine Learning (Isolation Forest):** Model multivariat 7-fitur terlatih dengan parameter `contamination=0.01` (1.0% populasi) untuk mengisolasi anomali ekstrem pada distribusi nilai pagu, nilai kontrak, dan selisih relatif.
   - **Rule-Based Red Flag Forensik:** Evaluasi 4 indikator forensik domain pengadaan publik:
     - `RF-1`: Deviasi Harga Kontrak Ekstrem (<1% dari pagu atau melebihi pagu).
     - `RF-2`: Durasi Pengumuman ke Penetapan Pemenang Sangat Singkat (<7 hari kalender / tender kilat).
     - `RF-3`: Pengadaan Menjelang Akhir Tahun Anggaran (Kuartal IV: Oktober–Desember).
     - `RF-4`: Konsentrasi Kemenangan Penyedia pada Satu Instansi Pembeli (≥20% dari total paket).
     - *Ambang Deteksi*: Skor Komposit `rf_score ≥ 2` (terpicu minimal 2 indikator secara bersamaan).

2. **Analisis Komparatif Non-Ground-Truth:**
   - Karena dataset audit pengadaan publik tidak memiliki *ground-truth label* kepastian korupsi, performa diukur melalui metrik kesepakatan himpunan:
     - **Jaccard Similarity Index** (`0.0145`)
     - **Overlap Coefficient** (`0.0808`)
     - **Korelasi Peringkat Spearman** ($\rho = 0.0261$, $p < 0.001$)
   - Klasifikasi 4 Kuadran Komparasi:
     - 🚨 **Irisan (IF + RF)**: *878 paket (0.081%)* &mdash; Konsensus kedua metode, prioritas investigasi audit paling kritis.
     - 🤖 **Hanya Isolation Forest**: *9,984 paket (0.919%)* &mdash; Outlier nilai kontrak dan pagu ekstrem.
     - 🚩 **Hanya Red Flag**: *49,780 paket (4.582%)* &mdash; Anomali prosedural dan administratif.
     - ✅ **Normal Keduanya**: *1,025,783 paket (94.418%)* &mdash; Pola pengadaan wajar sesuai ketentuan.

3. **Pipeline Upload & Inferensi Ganda Real-Time:**
   - Mendukung format `.csv`, `.xlsx`, `.zip`, `.tar.gz` (otomatis mendeteksi relasi `main.csv` dan `awards.csv`).
   - Eksekusi inferensi simultan: Preprocessing 7-fitur $\rightarrow$ Prediksi Isolation Forest $\rightarrow$ Komputasi 4 Red Flag $\rightarrow$ Pembentukan 4 Kelompok Komparasi.
   - Penyimpanan batch otomatis ke tabel MySQL `upload_batches` & `analysis_results` dengan chunking ACID.
   - Sinkronisasi instan ringkasan batch ke node Firebase `opentender/dashboard` dan `opentender/analysis_history`.

4. **Enterprise Analytics Dashboard (Streamlit + Altair):**
   - **Dashboard Overview**: Ringkasan KPI populasi historis (1.086.425 OCID), tab analisis komparatif lengkap (matriks 2×2, grafik donat kelompok, distribusi 4 indikator, distribusi skor red flag), dan tab akumulasi upload baru.
   - **Hasil Deteksi Anomali**: Filter multivariat (kategori, status IF, kelompok komparasi, tahun, skor RF, pencarian ID) dilengkapi **Panel Detail Inspeksi Forensik Paket** (dossier status 4 indikator Red Flag).
   - **Upload & Analisis Data**: Wizard interaktif upload 5 langkah, visualisasi dual KPI, tabel pratinjau komparatif, dan unduhan CSV 16 kolom.
   - **Riwayat Analisis**: Audit trail sesi batch upload dari MySQL InnoDB dan Firebase RTDB.
   - **Statistik Tender**: Agregasi Top Buyers, Top Suppliers, LPSE, dan tren tahunan.
   - **Informasi Sistem**: Dokumentasi teknis arsitektur dual-engine, formula metrik, dan skema database.

---

## 🗂️ Struktur Proyek Terbaru

```text
opentender-anomaly-detection/
├── app.py                                    # Dashboard utama Streamlit Enterprise
├── config.py                                 # Konfigurasi terpusat (MySQL, Firebase, Path)
├── red_flag_engine.py                        # Engine komputasi 4 indikator Rule-Based Red Flag
├── run_ocid_pipeline.py                      # Pipeline relasional MySQL, retraining IF, & eksperimen
├── db_mysql.py                               # Driver MySQL InnoDB (ACID, batch chunking, query)
├── db_firebase.py                            # Driver Firebase Admin SDK (Realtime Database sync)
├── model_isolation_forest_opentender.joblib  # Artefak model ML Isolation Forest terlatih
├── hasil_deteksi_anomali_opentender.csv      # Dataset historis hasil deteksi lengkap 16 kolom (1.08M baris)
├── test_e2e_integration.py                   # Skrip pengujian end-to-end pipeline & integrasi DB
├── requirements.txt                          # Daftar dependensi pustaka Python
├── README.md                                 # Dokumentasi komprehensif proyek
├── .gitignore                                # Konfigurasi pengabaian Git (termasuk backups & dataset)
├── dataset/                                  # Folder dataset mentah lokal
├── dataset_test/                             # Dataset uji mini (main.csv & awards.csv)
├── credentials/                              # Kredensial Firebase (Service Account Key)
│   ├── firebase-key.json                     # Kunci aktif (diabaikan Git)
│   ├── firebase-key.example.json             # Template contoh
│   └── README.md                             # Panduan kredensial
├── results/                                  # Artefak hasil eksperimen penelitian
│   ├── laporan_eksperimen_ocid.txt           # Laporan lengkap 5 skenario eksperimen & benchmark
│   └── studi_kasus_anomali.csv               # 30 sampel studi kasus (Irisan, Hanya IF, Hanya RF)
├── backups/                                  # Folder cadangan artefak (diabaikan Git)
│   ├── hasil_deteksi_anomali_opentender.backup.csv
│   └── model_isolation_forest_opentender.backup.joblib
└── scripts/                                  # Skrip migrasi dan utilitas aktif
    ├── buat_tabel_analisis.py                # Pembuatan tabel upload_batches & analysis_results
    ├── buat_tabel_mysql.py                   # Pembuatan skema tabel historis MySQL
    ├── import_mysql.py                       # Pipeline migrasi dataset relasional ke MySQL
    ├── import_rtdb.py                        # Pipeline sinkronisasi statistik ke Firebase
    ├── koneksi_mysql.py                      # Tes koneksi database MySQL
    ├── koneksi_firebase.py                   # Tes koneksi Firebase
    └── archive/                              # Skrip arsip debugging & analisis sekali pakai
        ├── audit_collector.py
        ├── cek_hasil_anomali.py
        ├── cek_model.py
        ├── cek_status.py
        ├── check_radio_html.py
        ├── get_label_html.py
        ├── inspect_dom.py
        ├── inspect_elements.py
        ├── hasil_audit_bukti.txt
        ├── uji_dataset.py
        └── uji_konsistensi.py
```

---

## 🔬 Hasil Temuan Eksperimen Penelitian (Level OCID)

Berdasarkan eksekusi pipeline pada seluruh populasi pengadaan Indonesia tingkat OCID (**1.086.425 tender unik**):

| Metrik / Indikator | Nilai Populasi | Persentase (%) | Keterangan |
| :--- | :---: | :---: | :--- |
| **Total Populasi OCID** | 1.086.425 | 100.00% | Unit analisis 1 baris per tender |
| **Anomali Isolation Forest** | 10.862 | 1.00% | Parameter contamination = 0.01 |
| **Flagged Red Flag (rf_score ≥ 2)** | 50.658 | 4.66% | Terpicu ≥ 2 dari 4 indikator |
| **🚨 Irisan (IF + RF)** | **878** | **0.081%** | **Konsensus Prioritas Kritis Audit** |
| Hanya Isolation Forest | 9.984 | 0.919% | Outlier deviasi harga dan pagu ekstrem |
| Hanya Red Flag | 49.780 | 4.582% | Anomali administratif / prosedural |
| Normal Keduanya | 1.025.783 | 94.418% | Pola pengadaan wajar |
| **Jaccard Similarity Index** | 0.0145 | - | Menunjukkan sifat komplementer/ortogonal |
| **Overlap Coefficient** | 0.0808 | - | Proporsi irisan terhadap himpunan terkecil |
| **Korelasi Spearman (ρ)** | 0.0261 | - | $p$-value < 0.001 (signifikan secara statistik) |

---

## 🚀 Panduan Menjalankan Sistem

### 1. Prasyarat Sistem
- **Python**: 3.10 – 3.12
- **MySQL Server**: Port 3306 (database: `opentender_bigdata`)
- **Kredensial Firebase**: Simpan file service account di `credentials/firebase-key.json`

### 2. Aktivasi Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Menjalankan Dashboard Streamlit
```powershell
.\.venv\Scripts\streamlit run app.py
```
Akses dashboard pada peramban web di alamat: `http://localhost:8501`.

### 4. Menjalankan Pengujian Integrasi End-to-End
```powershell
.\.venv\Scripts\python test_e2e_integration.py
```
Skrip ini akan menguji pemuatan model, integrasi relasi dataset uji, deteksi anomali, penyimpanan batch ke MySQL, dan sinkronisasi real-time ke Firebase RTDB.

---

## 🛠️ Tech Stack & Spesifikasi

- **Bahasa Pemrograman**: Python 3.12
- **Machine Learning**: Scikit-Learn (`IsolationForest`), Scipy (`spearmanr`), Joblib
- **Data Engine**: Pandas, NumPy
- **Frontend Dashboard**: Streamlit, Altair Interactive Visualizations
- **Relational SQL Database**: MySQL 8.x InnoDB (Foreign Keys, B-Tree Indexes, ACID Transactions)
- **NoSQL Cloud Database**: Google Firebase Realtime Database (Admin SDK)
