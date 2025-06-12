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
This is a synthetic dataset created using actual data from a financial institution. The data has been modified to remove identifiable features and the numbers transformed to ensure they do not link to original source (financial institution). Contoh: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/kmldas/loan-default-prediction).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Index : This is the serial number or unique identifier of the loan taker.
- Employed : This is a Boolean 1= employed 0= unemployed.
- Bank Balance : Bank Balance of the loan taker.
- Annual Salary : Annual salary of the loan taker.
- Defaulted : This is a Boolean 1= defaulted 0= not defaulted

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

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
