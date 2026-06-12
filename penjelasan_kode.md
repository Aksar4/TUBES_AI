# Penjelasan Source Code
## Sistem Klasifikasi Kesejahteraan Pekerja
### Algoritma K-Nearest Neighbor (KNN)

---

## Gambaran Umum Struktur Program

Program ini dibagi menjadi **3 file** dengan tugas masing-masing:

| File | Tugas |
|------|-------|
| `knn.py` | Menyimpan data training + seluruh logika algoritma KNN |
| `evaluasi.py` | Mengukur seberapa akurat model kita |
| `main.py` | Antarmuka pengguna — input, output, dan menu |

Ketiga file ini saling terhubung. `main.py` memanggil fungsi dari `knn.py` dan `evaluasi.py`. Jadi cukup jalankan `main.py` untuk menjalankan seluruh program.

---

## FILE 1 — `knn.py`

File ini adalah **inti dari seluruh program**. Di sinilah algoritma KNN diimplementasikan secara penuh dan manual, tanpa menggunakan library seperti scikit-learn.

---

### Bagian 1: Import Library

```python
import math
```

Kita hanya menggunakan satu library, yaitu `math`. Library ini adalah bawaan Python dan digunakan khusus untuk fungsi `math.sqrt()` — yaitu akar kuadrat — yang dibutuhkan dalam perhitungan jarak Euclidean. Tidak ada library machine learning yang digunakan sama sekali.

---

### Bagian 2: Data Training

```python
DATA = [
    [7500000, 3000000, 3000000, 40, "Sejahtera"],
    [5000000, 3200000, 3000000, 40, "Cukup Sejahtera"],
    [2000000, 1900000, 2500000, 30, "Kurang Sejahtera"],
    ...
]
```

`DATA` adalah variabel berbentuk list yang berisi 27 data pekerja. Data ini berperan sebagai **data training**, yaitu data yang sudah diketahui labelnya dan dijadikan acuan oleh algoritma KNN saat melakukan prediksi.

Setiap baris mewakili satu data pekerja dengan format:
- Kolom 0 → Pendapatan per bulan (Rupiah)
- Kolom 1 → Pengeluaran per bulan (Rupiah)
- Kolom 2 → UMP (Upah Minimum Provinsi) (Rupiah)
- Kolom 3 → Jam kerja per minggu
- Kolom 4 → Label kelas: `"Sejahtera"`, `"Cukup Sejahtera"`, atau `"Kurang Sejahtera"`

Data dibagi merata, masing-masing 9 data per kelas, agar model tidak bias ke salah satu kelas.

---

### Bagian 3: Fungsi `hitung_minmax`

```python
def hitung_minmax(data):
    min_val = [data[0][i] for i in range(4)]
    max_val = [data[0][i] for i in range(4)]
    for baris in data:
        for i in range(4):
            if baris[i] < min_val[i]: min_val[i] = baris[i]
            if baris[i] > max_val[i]: max_val[i] = baris[i]
    return min_val, max_val
```

Fungsi ini bertugas **mencari nilai minimum dan maksimum** dari setiap fitur di seluruh data training.

Mengapa ini diperlukan? Karena sebelum menghitung jarak, kita harus melakukan normalisasi terlebih dahulu. Normalisasi membutuhkan nilai min dan max sebagai patokan. Fungsi ini menelusuri seluruh data satu per satu dan membandingkan setiap nilai — sepenuhnya manual, tanpa fungsi `min()` atau `max()` bawaan Python.

---

### Bagian 4: Fungsi `normalisasi`

```python
def normalisasi(baris, min_val, max_val):
    hasil = []
    for i in range(4):
        rentang = max_val[i] - min_val[i]
        nilai = (baris[i] - min_val[i]) / rentang if rentang != 0 else 0
        hasil.append(nilai)
    return hasil
```

Fungsi ini menerapkan **Normalisasi Min-Max** dengan rumus:

```
nilai_baru = (x - min) / (max - min)
```

Hasilnya, semua nilai diubah ke rentang antara 0 sampai 1.

Normalisasi ini sangat penting karena fitur-fitur kita memiliki skala yang sangat berbeda. Pendapatan bisa bernilai jutaan, sementara jam kerja hanya puluhan. Tanpa normalisasi, fitur pendapatan akan mendominasi perhitungan jarak dan membuat fitur lain tidak berpengaruh. Dengan normalisasi, semua fitur punya bobot yang setara.

Terdapat pengecekan `if rentang != 0` untuk menghindari pembagian dengan nol, dalam kasus semua nilai pada satu fitur sama persis.

---

### Bagian 5: Fungsi `jarak_euclidean`

```python
def jarak_euclidean(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return math.sqrt(total)
```

Fungsi ini menghitung **jarak Euclidean** antara dua titik data — yaitu data input dari pengguna dan satu data dari training.

Rumus Euclidean Distance:

```
d = sqrt( (a1-b1)² + (a2-b2)² + (a3-b3)² + (a4-b4)² )
```

Cara kerjanya: hitung selisih nilai tiap fitur, kuadratkan, jumlahkan semua, lalu ambil akar kuadratnya. Semakin kecil nilainya, berarti dua data tersebut semakin mirip satu sama lain.

---

### Bagian 6: Fungsi `prediksi_knn` — Inti Algoritma KNN

```python
def prediksi_knn(input_data, k, min_val, max_val):
    input_norm = normalisasi(input_data, min_val, max_val)

    jarak_list = []
    for baris in DATA:
        train_norm = normalisasi(baris[:4], min_val, max_val)
        j = jarak_euclidean(input_norm, train_norm)
        jarak_list.append((j, baris[4]))

    for i in range(len(jarak_list) - 1):
        for j in range(len(jarak_list) - i - 1):
            if jarak_list[j][0] > jarak_list[j+1][0]:
                jarak_list[j], jarak_list[j+1] = jarak_list[j+1], jarak_list[j]

    voting = {}
    for j, label in jarak_list[:k]:
        voting[label] = voting.get(label, 0) + 1

    hasil = max(voting, key=lambda x: voting[x])
    return hasil, voting, jarak_list[:k]
```

Inilah fungsi utama yang menjalankan algoritma KNN. Prosesnya terdiri dari 4 langkah:

**Langkah 1 — Normalisasi input**
Data yang dimasukkan pengguna dinormalisasi terlebih dahulu menggunakan nilai min-max yang sama dengan data training, agar perbandingannya adil.

**Langkah 2 — Hitung jarak ke semua data training**
Program menghitung jarak Euclidean antara input pengguna dan setiap data di `DATA`. Hasilnya disimpan dalam `jarak_list` sebagai pasangan `(jarak, label)`.

**Langkah 3 — Urutkan menggunakan Bubble Sort**
`jarak_list` diurutkan dari jarak terkecil ke terbesar menggunakan Bubble Sort yang diimplementasikan secara manual. Ini dilakukan dengan dua perulangan bersarang yang membandingkan dan menukar posisi elemen secara berulang.

**Langkah 4 — Voting K tetangga terdekat**
Diambil K data teratas (yang jaraknya paling kecil). Setiap data memberikan satu suara untuk kelasnya. Kelas dengan suara terbanyak menjadi hasil prediksi akhir.

---

## FILE 2 — `evaluasi.py`

File ini bertugas **mengukur seberapa akurat model KNN kita** sebelum digunakan untuk prediksi sesungguhnya.

---

### Bagian 1: Import

```python
from knn import DATA, hitung_minmax, normalisasi, jarak_euclidean
```

File ini mengambil fungsi-fungsi yang sudah dibuat di `knn.py` agar tidak perlu menulis ulang kode yang sama.

---

### Bagian 2: Fungsi `hitung_akurasi`

```python
def hitung_akurasi(k):
    benar = 0
    for idx, data_uji in enumerate(DATA):
        data_training = [DATA[i] for i in range(len(DATA)) if i != idx]
        min_val, max_val = hitung_minmax(data_training)
        uji_norm = normalisasi(data_uji[:4], min_val, max_val)

        jarak_list = []
        for baris in data_training:
            train_norm = normalisasi(baris[:4], min_val, max_val)
            j = sum((uji_norm[i] - train_norm[i]) ** 2 for i in range(4))
            jarak_list.append((math.sqrt(j), baris[4]))

        jarak_list.sort(key=lambda x: x[0])
        voting = {}
        for _, label in jarak_list[:k]:
            voting[label] = voting.get(label, 0) + 1

        prediksi = max(voting, key=lambda x: voting[x])
        if prediksi == data_uji[4]:
            benar += 1

    return round((benar / len(DATA)) * 100, 2)
```

Fungsi ini mengukur akurasi menggunakan metode **Leave-One-Out Cross Validation (LOO-CV)**.

Cara kerjanya adalah sebagai berikut: dari 27 data yang ada, satu data diambil sebagai data uji, dan 26 sisanya dijadikan data training. Proses ini diulang sebanyak 27 kali — setiap data mendapat giliran menjadi data uji satu kali. Setiap iterasi, program menjalankan KNN dan mengecek apakah prediksinya sesuai dengan label asli data tersebut.

Rumus akurasi:
```
Akurasi = (jumlah prediksi benar / total data) × 100%
```

Metode LOO-CV dipilih karena dataset kita kecil — hanya 27 data. Dengan metode ini, semua data dimanfaatkan semaksimal mungkin untuk evaluasi.

---

### Bagian 3: Fungsi `cari_k_terbaik`

```python
def cari_k_terbaik():
    k_terbaik = 1
    akurasi_terbaik = 0
    for k in range(1, 10):
        akurasi = hitung_akurasi(k)
        if akurasi > akurasi_terbaik:
            akurasi_terbaik = akurasi
            k_terbaik = k
    return k_terbaik, akurasi_terbaik
```

Fungsi ini mencoba setiap nilai K dari 1 sampai 9, menghitung akurasi untuk masing-masing, lalu menyimpan K yang menghasilkan akurasi tertinggi. K inilah yang nantinya dipakai saat prediksi berlangsung. Dengan cara ini, model secara otomatis memilih konfigurasi terbaiknya sendiri tanpa perlu di-set manual.

---

## FILE 3 — `main.py`

File ini adalah **pintu masuk program** — satu-satunya file yang dijalankan pengguna. Tugasnya murni mengurus tampilan dan interaksi dengan pengguna.

---

### Bagian 1: Import dan Inisialisasi

```python
from knn import DATA, hitung_minmax, prediksi_knn
from evaluasi import cari_k_terbaik

MIN_VAL, MAX_VAL = hitung_minmax(DATA)
K_TERBAIK, AKURASI = cari_k_terbaik()
```

Pertama, program mengimpor semua yang dibutuhkan dari dua file lainnya. Kemudian, **saat program pertama kali dijalankan**, dua hal langsung dihitung otomatis di background:

1. Nilai min dan max dari seluruh data training (untuk normalisasi)
2. Nilai K terbaik beserta akurasinya (dari proses evaluasi)

Ini dilakukan di luar fungsi agar hasilnya tersimpan sebagai variabel global dan tidak perlu dihitung ulang setiap kali pengguna melakukan prediksi.

---

### Bagian 2: Fungsi `input_angka`

```python
def input_angka(pesan):
    while True:
        try:
            return float(input(f"  {pesan} : ").replace('.', '').replace(',', ''))
        except ValueError:
            print("  [!] Masukkan angka yang valid.\n")
```

Fungsi ini meminta input angka dari pengguna secara berulang sampai input yang diberikan valid. Ada dua hal yang dilakukan sebelum konversi: menghapus titik dan koma — sehingga pengguna bisa mengetik `5.000.000` atau `5,000,000` dan tetap terbaca dengan benar sebagai angka lima juta.

---

### Bagian 3: Fungsi `tampilkan_hasil`

```python
def tampilkan_hasil(prediksi, voting, tetangga, k, akurasi):
    print(f"  Status Kesejahteraan : {prediksi.upper()}")
    print(f"  K yang digunakan     : {k}")
    print(f"  Akurasi Model        : {akurasi}%")
    for label, suara in voting.items():
        print(f"    {label:<20}: {suara} suara")
    for i, (jarak, label) in enumerate(tetangga, 1):
        print(f"    {i}. {label:<20} (jarak: {jarak:.4f})")
```

Fungsi ini menampilkan tiga informasi sekaligus ke layar: hasil prediksi kelas kesejahteraan, hasil voting dari K tetangga, dan detail jarak masing-masing tetangga. Tampilan ini membantu pengguna memahami tidak hanya apa hasilnya, tetapi juga *bagaimana* program sampai pada kesimpulan tersebut.

---

### Bagian 4: Fungsi `main` — Alur Utama Program

```python
def main():
    # Tampilkan header dan info model
    ...
    while True:
        pilihan = input("  Pilih : ").strip()
        if pilihan == '1':
            # Ambil input, jalankan KNN, tampilkan hasil
            hasil, voting, tetangga = prediksi_knn(
                [pendapatan, pengeluaran, ump, jam_kerja],
                K_TERBAIK, MIN_VAL, MAX_VAL
            )
            tampilkan_hasil(...)
        elif pilihan == '0':
            break
```

Fungsi `main` menjalankan program dalam sebuah loop yang terus berjalan sampai pengguna memilih keluar. Di dalamnya, program menampilkan header, meminta input, memanggil `prediksi_knn` dari `knn.py`, lalu menampilkan hasilnya. Baris `if __name__ == '__main__'` di bagian bawah memastikan fungsi `main` hanya berjalan ketika file ini dijalankan langsung, bukan ketika diimpor oleh file lain.

---

## Alur Kerja Program Secara Keseluruhan

```
Pengguna jalankan main.py
        ↓
Inisialisasi: hitung min-max + cari K terbaik (otomatis)
        ↓
Pengguna masukkan data (pendapatan, pengeluaran, UMP, jam kerja)
        ↓
Data dinormalisasi → dihitung jaraknya ke 27 data training
        ↓
Diurutkan → ambil K tetangga terdekat → voting
        ↓
Tampilkan hasil: Sejahtera / Cukup Sejahtera / Kurang Sejahtera
```

---

## Ringkasan Poin Penting untuk Presentasi

- **Tidak ada library KNN** — seluruh algoritma ditulis dari nol menggunakan logika dasar Python
- **Normalisasi Min-Max** digunakan agar fitur dengan nilai besar tidak mendominasi perhitungan jarak
- **Euclidean Distance** digunakan untuk mengukur kemiripan antar data
- **Bubble Sort manual** digunakan untuk mengurutkan jarak tanpa mengandalkan `.sort()` secara implisit
- **Leave-One-Out Cross Validation** digunakan untuk mengukur akurasi secara jujur pada dataset kecil
- **K terbaik dipilih otomatis** oleh program dari rentang K=1 hingga K=9
