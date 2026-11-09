
import mysql.connector

try:
    koneksi = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database="opentender_bigdata"
    )

    if koneksi.is_connected():
        print("================================")
        print("KONEKSI MYSQL BERHASIL!")

        cursor = koneksi.cursor()
        cursor.execute("SELECT DATABASE()")

        hasil = cursor.fetchone()
        print("Database:", hasil[0])

        print("MySQL siap digunakan.")
        print("================================")

    cursor.close()
    koneksi.close()

except mysql.connector.Error as error:
    print("KONEKSI MYSQL GAGAL!")
    print("Error:", error)
