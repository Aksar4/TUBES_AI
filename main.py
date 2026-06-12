# ============================================================
# main.py - Program utama, antarmuka pengguna
# Jalankan file ini untuk memulai program
# ============================================================

from knn import DATA, hitung_minmax, prediksi_knn
from evaluasi import cari_k_terbaik

# Hitung parameter normalisasi dan K terbaik saat program dimulai
MIN_VAL, MAX_VAL = hitung_minmax(DATA)
K_TERBAIK, AKURASI = cari_k_terbaik()

def input_angka(pesan):
    """Minta input angka dari pengguna, ulangi jika tidak valid."""
    while True:
        try:
            # Hapus titik pemisah ribuan sebelum konversi (misal: 5.000.000)
            return float(input(f"  {pesan} : ").replace('.', '').replace(',', ''))
        except ValueError:
            print("  [!] Masukkan angka yang valid.\n")

def tampilkan_hasil(prediksi, voting, tetangga, k, akurasi):
    """Tampilkan hasil prediksi ke layar."""
    print()
    print("=" * 40)
    print("         HASIL PREDIKSI")
    print("=" * 40)
    print(f"  Status Kesejahteraan : {prediksi.upper()}")
    print(f"  K yang digunakan     : {k}")
    print(f"  Akurasi Model        : {akurasi}%")
    print()
    print("  Voting K Tetangga:")
    for label, suara in voting.items():
        print(f"    {label:<20}: {suara} suara")
    print()
    print("  Detail Tetangga Terdekat:")
    for i, (jarak, label) in enumerate(tetangga, 1):
        print(f"    {i}. {label:<20} (jarak: {jarak:.4f})")
    print("=" * 40)

def main():
    print("=" * 40)
    print("  KLASIFIKASI KESEJAHTERAAN PEKERJA")
    print("       Algoritma K-Nearest Neighbor")
    print("=" * 40)
    print(f"  Jumlah Data Training : {len(DATA)}")
    print(f"  K Terbaik            : {K_TERBAIK}")
    print(f"  Akurasi              : {AKURASI}%")
    print("=" * 40)

    while True:
        print()
        print("  [1] Prediksi")
        print("  [0] Keluar")
        pilihan = input("  Pilih : ").strip()

        if pilihan == '1':
            print()
            print("  Masukkan Data Pekerja:")
            print("  " + "-" * 35)
            pendapatan  = input_angka("Pendapatan/Bulan (Rp)")
            pengeluaran = input_angka("Pengeluaran/Bulan (Rp)")
            ump         = input_angka("UMP Provinsi (Rp)")
            jam_kerja   = input_angka("Jam Kerja/Minggu")

            # Jalankan KNN dengan input pengguna
            hasil, voting, tetangga = prediksi_knn(
                [pendapatan, pengeluaran, ump, jam_kerja],
                K_TERBAIK, MIN_VAL, MAX_VAL
            )

            tampilkan_hasil(hasil, voting, tetangga, K_TERBAIK, AKURASI)

        elif pilihan == '0':
            print("\n  Program selesai.\n")
            break
        else:
            print("  [!] Pilih 1 atau 0.")

if __name__ == '__main__':
    main()
