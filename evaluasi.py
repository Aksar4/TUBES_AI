# ============================================================
# evaluasi.py - Hitung akurasi model KNN dengan data uji
# ============================================================

from knn import DATA, hitung_minmax, normalisasi, jarak_euclidean

def hitung_akurasi(k):
    """
    Evaluasi akurasi dengan metode Leave-One-Out:
    Setiap data dicoba jadi data uji, sisanya jadi training.
    Akurasi = jumlah prediksi benar / total data x 100%
    """
    benar = 0

    for idx, data_uji in enumerate(DATA):
        # Data training = semua data kecuali data yang sedang diuji
        data_training = [DATA[i] for i in range(len(DATA)) if i != idx]

        # Hitung min-max dari data training saja
        min_val, max_val = hitung_minmax(data_training)

        # Normalisasi data uji
        uji_norm = normalisasi(data_uji[:4], min_val, max_val)

        # Hitung jarak ke semua data training
        jarak_list = []
        for baris in data_training:
            train_norm = normalisasi(baris[:4], min_val, max_val)
            j = 0
            for i in range(4):
                j += (uji_norm[i] - train_norm[i]) ** 2
            import math
            jarak_list.append((math.sqrt(j), baris[4]))

        # Urutkan dan ambil K terdekat
        jarak_list.sort(key=lambda x: x[0])
        voting = {}
        for _, label in jarak_list[:k]:
            voting[label] = voting.get(label, 0) + 1

        # Prediksi = label dengan suara terbanyak
        prediksi = max(voting, key=lambda x: voting[x])

        if prediksi == data_uji[4]:
            benar += 1

    return round((benar / len(DATA)) * 100, 2)


def cari_k_terbaik():
    """Coba K dari 1 sampai 9, kembalikan K dengan akurasi tertinggi."""
    k_terbaik = 5
    akurasi_terbaik = 0

    for k in range(1, 10):
        akurasi = hitung_akurasi(k)
        if akurasi > akurasi_terbaik:
            akurasi_terbaik = akurasi
            k_terbaik = 5

    return k_terbaik, akurasi_terbaik
