"""
Modul Firebase Realtime Database untuk proyek OpenTender Anomaly Detection.

Fungsi utama:
    - init_firebase()            : inisialisasi koneksi Firebase (sekali saja)
    - test_connection()          : uji koneksi Firebase
    - kirim_ringkasan_batch()    : kirim ringkasan hasil upload ke analysis_history
    - perbarui_dashboard()       : perbarui node dashboard dengan akumulasi terbaru
    - ambil_riwayat_firebase()   : baca riwayat analysis_history untuk dashboard
    - ambil_dashboard_firebase() : baca node dashboard

Prinsip keamanan:
    - Tidak menghapus node historis yang sudah ada.
    - Hanya MENAMBAH atau MEMPERBARUI node.
    - Jika Firebase gagal, MySQL tidak akan dipengaruhi (dipanggil setelah MySQL sukses).
"""

import threading
from datetime import datetime, timezone
from typing import Optional

import firebase_admin
from firebase_admin import credentials, db

from config import FIREBASE_CREDENTIALS_PATH, FIREBASE_DATABASE_URL

# Lock agar inisialisasi aman di environment multi-thread (Streamlit)
_init_lock = threading.Lock()
_initialized = False


# ============================================================
# INISIALISASI
# ============================================================

def init_firebase() -> bool:
    """
    Menginisialisasi koneksi Firebase Admin SDK.
    Aman dipanggil berkali-kali — hanya dijalankan sekali.
    Mengembalikan True jika berhasil, False jika gagal.
    """
    global _initialized
    with _init_lock:
        if _initialized:
            return True
        try:
            # Jika sudah ada app yang diinisialisasi sebelumnya
            if firebase_admin._apps:
                _initialized = True
                return True

            cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
            _initialized = True
            return True
        except Exception as e:
            print(f"[Firebase] Inisialisasi gagal: {e}")
            return False


def test_connection() -> tuple[bool, str]:
    """
    Menguji kesiapan koneksi Firebase Admin SDK.
    Mengembalikan (True, pesan) jika siap, (False, pesan error) jika gagal.
    """
    try:
        if not init_firebase():
            return False, "Inisialisasi Firebase gagal."
        if firebase_admin._apps:
            return True, "Koneksi Firebase terhubung."
        return False, "Firebase belum terinisialisasi."
    except Exception as e:
        return False, f"Koneksi Firebase gagal: {e}"


# ============================================================
# HELPER
# ============================================================

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_dict(obj) -> dict:
    """Pastikan semua nilai dalam dict bisa diserialisasi ke JSON Firebase."""
    if not isinstance(obj, dict):
        return {}
    result = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            result[str(k)] = _safe_dict(v)
        elif isinstance(v, list):
            result[str(k)] = [str(x) for x in v]
        elif v is None or v != v:  # None atau NaN
            result[str(k)] = None
        else:
            result[str(k)] = v
    return result


# ============================================================
# KIRIM RINGKASAN BATCH
# ============================================================

def kirim_ringkasan_batch(
    batch_id: str,
    filename: str,
    total_rows: int,
    total_normal: int,
    total_anomali: int,
    catatan: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Menyimpan ringkasan hasil satu batch upload ke:
        opentender/analysis_history/{batch_id}

    Tidak menimpa atau menghapus batch lain.
    """
    try:
        if not init_firebase():
            return False, "Firebase tidak terinisialisasi."

        total = total_normal + total_anomali
        pct   = round(total_anomali / total * 100, 2) if total > 0 else 0.0

        payload = {
            "batch_id":       batch_id,
            "filename":       str(filename),
            "total_rows":     int(total_rows),
            "total_normal":   int(total_normal),
            "total_anomali":  int(total_anomali),
            "pct_anomali":    float(pct),
            "status":         "done",
            "timestamp":      _now_iso(),
        }
        if catatan:
            payload["catatan"] = str(catatan)

        ref = db.reference(f"opentender/analysis_history/{batch_id}")
        ref.set(payload)
        return True, f"Ringkasan batch '{batch_id}' berhasil dikirim."
    except Exception as e:
        return False, f"Gagal kirim ringkasan batch: {e}"


# ============================================================
# PERBARUI DASHBOARD
# ============================================================

def perbarui_dashboard(
    total_normal_baru: int,
    total_anomali_baru: int,
) -> tuple[bool, str]:
    """
    Memperbarui node opentender/dashboard dengan menambahkan
    hasil batch terbaru ke akumulasi yang sudah ada.

    Node yang diperbarui:
        opentender/dashboard/total_upload_tenders
        opentender/dashboard/total_upload_normal
        opentender/dashboard/total_upload_anomali
        opentender/dashboard/pct_upload_anomali
        opentender/dashboard/last_updated

    Node historis (total_tenders, total_awards, dll.) TIDAK disentuh.
    """
    try:
        if not init_firebase():
            return False, "Firebase tidak terinisialisasi."

        ref = db.reference("opentender/dashboard")
        snapshot = ref.get() or {}

        # Akumulasi khusus untuk hasil upload (terpisah dari data historis)
        prev_normal  = int(snapshot.get("total_upload_normal",  0) or 0)
        prev_anomali = int(snapshot.get("total_upload_anomali", 0) or 0)

        new_normal  = prev_normal  + total_normal_baru
        new_anomali = prev_anomali + total_anomali_baru
        new_total   = new_normal + new_anomali
        new_pct     = round(new_anomali / new_total * 100, 2) if new_total > 0 else 0.0

        ref.update({
            "total_upload_tenders": int(new_total),
            "total_upload_normal":  int(new_normal),
            "total_upload_anomali": int(new_anomali),
            "pct_upload_anomali":   float(new_pct),
            "last_updated":         _now_iso(),
        })
        return True, "Node dashboard berhasil diperbarui."
    except Exception as e:
        return False, f"Gagal memperbarui dashboard Firebase: {e}"


# ============================================================
# BACA DATA DARI FIREBASE (UNTUK DASHBOARD)
# ============================================================

def ambil_dashboard_firebase() -> Optional[dict]:
    """
    Membaca node opentender/dashboard dari Firebase.
    Mengembalikan dict atau None jika gagal.
    """
    try:
        if not init_firebase():
            return None
        return db.reference("opentender/dashboard").get()
    except Exception:
        return None


def ambil_riwayat_firebase(limit: int = 20) -> list[dict]:
    """
    Membaca riwayat upload dari opentender/analysis_history.
    Mengembalikan list dict yang diurutkan terbaru dulu.
    """
    try:
        if not init_firebase():
            return []
        snapshot = db.reference("opentender/analysis_history").get()
        if not snapshot:
            return []
        items = list(snapshot.values())
        # Urutkan berdasarkan timestamp terbaru
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[:limit]
    except Exception:
        return []


def ambil_statistik_firebase() -> Optional[dict]:
    """
    Membaca node opentender/statistics dari Firebase.
    """
    try:
        if not init_firebase():
            return None
        return db.reference("opentender/statistics").get()
    except Exception:
        return None
