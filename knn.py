# ============================================================
# knn.py - Implementasi Algoritma KNN secara manual
# ============================================================

import math

# ---------- Data Training ----------
# Format: [pendapatan, pengeluaran, ump, jam_kerja, label]
DATA = [
    [7500000, 3000000, 3000000, 40, "Sejahtera"],
    [8000000, 3500000, 3200000, 40, "Sejahtera"],
    [9000000, 4000000, 3500000, 45, "Sejahtera"],
    [10000000, 4500000, 3000000, 40, "Sejahtera"],
    [6500000, 2800000, 2800000, 40, "Sejahtera"],
    [7000000, 3200000, 3000000, 42, "Sejahtera"],
    [12000000, 5000000, 3500000, 40, "Sejahtera"],
    [8500000, 3800000, 3200000, 44, "Sejahtera"],
    [6000000, 2600000, 2800000, 40, "Sejahtera"],
    [4500000, 3000000, 2800000, 40, "Cukup Sejahtera"],
    [5000000, 3200000, 3000000, 40, "Cukup Sejahtera"],
    [5500000, 3500000, 3200000, 42, "Cukup Sejahtera"],
    [4800000, 3100000, 2800000, 35, "Cukup Sejahtera"],
    [5200000, 3400000, 3000000, 40, "Cukup Sejahtera"],
    [4200000, 2900000, 2500000, 38, "Cukup Sejahtera"],
    [5800000, 3800000, 3200000, 40, "Cukup Sejahtera"],
    [4600000, 3050000, 2800000, 36, "Cukup Sejahtera"],
    [5100000, 3300000, 3000000, 40, "Cukup Sejahtera"],
    [2000000, 1900000, 2500000, 30, "Kurang Sejahtera"],
    [1800000, 1700000, 2300000, 25, "Kurang Sejahtera"],
    [2500000, 2400000, 2800000, 35, "Kurang Sejahtera"],
    [1500000, 1450000, 2000000, 20, "Kurang Sejahtera"],
    [2200000, 2100000, 2500000, 30, "Kurang Sejahtera"],
    [1700000, 1650000, 2300000, 22, "Kurang Sejahtera"],
    [2800000, 2700000, 3000000, 35, "Kurang Sejahtera"],
    [2100000, 2000000, 2500000, 28, "Kurang Sejahtera"],
    [1900000, 1850000, 2300000, 25, "Kurang Sejahtera"],
]

# ---------- Normalisasi Min-Max ----------
# Ubah nilai ke rentang [0,1] agar fitur tidak saling mendominasi
def hitung_minmax(data):
    """Cari nilai min dan max dari setiap kolom fitur (kolom 0-3)."""
    min_val = [data[0][i] for i in range(4)]
    max_val = [data[0][i] for i in range(4)]
    for baris in data:
        for i in range(4):
            if baris[i] < min_val[i]: min_val[i] = baris[i]
            if baris[i] > max_val[i]: max_val[i] = baris[i]
    return min_val, max_val

def normalisasi(baris, min_val, max_val):
    """Normalisasi satu baris data menggunakan rumus (x - min) / (max - min)."""
    hasil = []
    for i in range(4):
        rentang = max_val[i] - min_val[i]
        nilai = (baris[i] - min_val[i]) / rentang if rentang != 0 else 0
        hasil.append(nilai)
    return hasil

# ---------- Jarak Euclidean ----------
# Rumus: sqrt( sum( (a - b)^2 ) )
def jarak_euclidean(a, b):
    """Hitung jarak antara dua titik data yang sudah dinormalisasi."""
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total)

# ---------- Prediksi KNN ----------
def prediksi_knn(input_data, k, min_val, max_val):
    """
    Langkah KNN:
    1. Normalisasi input
    2. Hitung jarak ke semua data training
    3. Ambil K tetangga terdekat
    4. Voting mayoritas -> hasil prediksi
    """
    # Normalisasi input pengguna
    input_norm = normalisasi(input_data, min_val, max_val)

    # Hitung jarak ke setiap data training
    jarak_list = []
    for baris in DATA:
        train_norm = normalisasi(baris[:4], min_val, max_val)
        j = jarak_euclidean(input_norm, train_norm)
        jarak_list.append((j, baris[4]))  # simpan (jarak, label)

    # Urutkan dari jarak terkecil (manual bubble sort)
    for i in range(len(jarak_list) - 1):
        for j in range(len(jarak_list) - i - 1):
            if jarak_list[j][0] > jarak_list[j+1][0]:
                jarak_list[j], jarak_list[j+1] = jarak_list[j+1], jarak_list[j]

    # Ambil K tetangga terdekat dan lakukan voting
    voting = {}
    for j, label in jarak_list[:k]:
        voting[label] = voting.get(label, 0) + 1

    # Kembalikan label dengan suara terbanyak
    hasil = max(voting, key=lambda x: voting[x])
    return hasil, voting, jarak_list[:k]
