# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek: Prediksi Gagal Bayar Pinjaman

Sektor keuangan sangat bergantung pada pengelolaan **risiko kredit**, yaitu potensi kerugian akibat **gagal bayar (default)** pinjaman. Gagal bayar dapat menyebabkan kerugian finansial signifikan bagi lembaga keuangan, mulai dari penurunan profitabilitas hingga krisis likuiditas (PWC, 2023).

Masalah ini penting untuk diselesaikan karena metode penilaian risiko tradisional kurang akurat dan efisien di tengah kompleksitas produk keuangan dan volume data yang besar. Tingginya angka gagal bayar tidak hanya merugikan lembaga keuangan tetapi juga menghambat pertumbuhan ekonomi. Dengan memprediksi gagal bayar, institusi dapat:

* **Meningkatkan Efisiensi Operasional:** Keputusan pinjaman lebih cepat dan konsisten.
* **Memitigasi Risiko:** Mengidentifikasi peminjam berisiko tinggi sejak awal untuk mengambil tindakan pencegahan.
* **Meningkatkan Profitabilitas:** Menekan angka gagal bayar untuk menjaga kualitas portofolio pinjaman.
* **Memenuhi Kepatuhan Regulasi:** Membantu memenuhi standar manajemen risiko yang ketat.

### Referensi

1.  PWC. (2023). *Global Banking & Capital Markets Outlook: Adapting to disruption and building for the future.* Diakses dari [https://www.pwc.com/gx/en/industries/financial-services/publications/global-banking-capital-markets-outlook.html](https://www.pwc.com/gx/en/industries/financial-services/publications/global-banking-capital-markets-outlook.html) (Akses pada 12 Juni 2025).

## Business Understanding

### Problem Statements

Proyek ini bertujuan mengatasi masalah utama di lembaga keuangan terkait **prediksi gagal bayar pinjaman**:

* **Pernyataan Masalah 1: Prediksi Risiko yang Kurang Akurat.** Metode lama sering salah dalam mengidentifikasi peminjam berisiko tinggi atau sebaliknya, sehingga menyebabkan kerugian atau hilangnya peluang.
* **Pernyataan Masalah 2: Proses Pinjaman yang Lambat.** Penilaian risiko yang manual memperlambat persetujuan pinjaman, merugikan pengalaman pelanggan.
* 
### Goals

Untuk mengatasi masalah di atas, tujuan proyek ini adalah:

* **Tujuan 1:** Meningkatkan akurasi dalam memprediksi peminjam berisiko tinggi.
* **Tujuan 2:** Membuat proses penilaian risiko yang lebih cepat dan otomatis.

### Solution Statements
* **Membangun Model Klasifikasi Ensemble.**
    Saya akan membuat model klasifikasi menggunakan setidaknya dua algoritma *ensemble* (seperti **Random Forest** dan **Gradient Boosting**). Model ini akan memprediksi apakah pinjaman akan **gagal bayar** atau tidak. Kinerja model akan diukur dengan **ROC AUC Score**, **F1-Score**, dan **Recall** (khususnya untuk kelas gagal bayar) pada data uji. Saya berharap metrik ini akan meningkat dibandingkan model dasar (misalnya, Regresi Logistik).

---

## Data Understanding
This is a synthetic dataset created using actual data from a financial institution. The data has been modified to remove identifiable features and the numbers transformed to ensure they do not link to original source (financial institution).(https://www.kaggle.com/datasets/kmldas/loan-default-prediction).

Berikut adalah contoh penulisan bab **Data Understanding** berdasarkan dataset `Default_Fin.csv` yang telah kamu unggah:

---

## Data Understanding

Pada tahap ini, dilakukan eksplorasi awal terhadap dataset *Default\_Fin.csv* untuk memahami karakteristik data yang digunakan dalam proses analisis. Dataset ini terdiri dari informasi finansial individu yang dapat digunakan untuk memprediksi apakah seseorang berpotensi mengalami gagal bayar (*default*).

### Pemuatan Data dan Informasi Dasar

Dataset dimuat menggunakan pustaka `pandas`. Berdasarkan hasil `df.info()`, dataset terdiri dari **10.000 baris dan 5 kolom**, tanpa adanya nilai kosong (*missing values*) pada seluruh kolom.

```python
# Output ringkasan
df.shape  -> (10000, 5)
df.isnull().sum()  -> semua bernilai 0
```

### Deskripsi Variabel

Berikut adalah penjelasan singkat mengenai tiap kolom dalam dataset:

| Nama Variabel   | Tipe Data | Deskripsi                                                                          |
| --------------- | --------- | ---------------------------------------------------------------------------------- |
| `Index`         | `int64`   | Penomoran baris atau identifikasi unik data (tidak memiliki nilai prediktif)       |
| `Employed`      | `int64`   | Status pekerjaan (1 = bekerja, 0 = tidak bekerja)                                  |
| `Bank Balance`  | `float64` | Saldo tabungan dalam satuan mata uang                                              |
| `Annual Salary` | `float64` | Pendapatan tahunan individu                                                        |
| `Defaulted?`    | `int64`   | Label target, menunjukkan apakah individu mengalami gagal bayar (1) atau tidak (0) |

### Statistik Deskriptif

Berikut ini adalah ringkasan statistik deskriptif untuk kolom numerik:

* Rata-rata saldo bank (`Bank Balance`) adalah sekitar **10.024**, dengan standar deviasi **5.804**.
* Rata-rata gaji tahunan (`Annual Salary`) adalah **402.203**, dengan penyebaran data yang cukup besar (standar deviasi **160.039**).
* Hanya sekitar **3.33%** individu yang mengalami gagal bayar (`Defaulted?` = 1), menunjukkan data yang cukup imbalanced.
* Kolom `Employed` memiliki nilai 1 untuk sekitar **70.56%** data, sisanya adalah individu yang tidak bekerja.

### Analisis Outlier (Identifikasi Awal)

Untuk mengidentifikasi outlier, digunakan visualisasi boxplot dan perhitungan IQR pada fitur numerik seperti `Bank Balance` dan `Annual Salary`. Outlier awal terlihat pada:

* `Bank Balance`: beberapa individu memiliki saldo 0, namun terdapat juga nilai yang sangat tinggi.
* `Annual Salary`: terlihat distribusi gaji yang sangat lebar, yang memungkinkan adanya nilai ekstrem.

Visualisasi boxplot akan dilakukan pada tahap eksplorasi berikutnya untuk memastikan keberadaan outlier secara visual.

### Penanganan Outlier

Outlier dapat berdampak negatif terhadap performa model prediktif. Oleh karena itu, dalam proyek ini akan dilakukan penanganan outlier menggunakan metode IQR (*Interquartile Range*), terutama untuk fitur `Bank Balance` dan `Annual Salary`. Titik data di luar rentang Q1 - 1.5*IQR dan Q3 + 1.5*IQR akan dianggap sebagai outlier dan dihapus dari dataset.

### Penghapusan Fitur yang Tidak Relevan

Kolom `Index` merupakan identifier unik yang tidak memiliki pengaruh terhadap prediksi dan oleh karena itu dihapus dari dataset:

```python
df_cleaned = df.drop(columns=['Index'])
```

### Insight Awal dari Data

* **Insight 1:** Mayoritas individu dalam dataset memiliki pekerjaan, dan kemungkinan gagal bayar lebih rendah pada individu yang bekerja.
* **Insight 2:** Saldo bank dan gaji tahunan tampaknya memiliki peran penting dalam menentukan kemungkinan gagal bayar.
* **Insight 3:** Dataset sangat tidak seimbang dari sisi target (`Defaulted?`), sehingga perlu penanganan khusus seperti resampling atau pemilihan metrik evaluasi yang tepat.

---

Kalau kamu ingin saya lanjutkan ke bagian visualisasi atau feature engineering, tinggal bilang ya!


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
