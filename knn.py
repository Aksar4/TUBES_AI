import math
import pandas as pd

# Membaca file CSV menggunakan Pandas
df = pd.read_csv('dataset_pekerja.csv')

# Mengubah dataframe Pandas kembali menjadi List agar bisa diproses algoritma KNN manual
DATA = df.values.tolist()

# ---------- Normalisasi Min-Max ----------
# (Sisa kode di bawahnya biarkan saja, jangan ada yang diubah)
def hitung_minmax(data):
    ...

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
