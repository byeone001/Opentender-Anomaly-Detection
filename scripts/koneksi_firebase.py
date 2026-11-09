
import firebase_admin
from firebase_admin import credentials, firestore

# Lokasi file kredensial Firebase
CREDENTIAL_PATH = "credentials/firebase-key.json"

try:
    # Inisialisasi Firebase
    cred = credentials.Certificate(CREDENTIAL_PATH)
    firebase_admin.initialize_app(cred)

    # Menghubungkan ke Firestore
    db = firestore.client()

    print("================================")
    print("KONEKSI FIREBASE BERHASIL!")
    print("Project ID:", cred.project_id)
    print("Firestore siap digunakan.")
    print("================================")

except Exception as e:
    print("KONEKSI FIREBASE GAGAL!")
    print("Error:", e)
