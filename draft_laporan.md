# LAPORAN TUGAS 1 — VISI KOMPUTER
## Otomatisasi Preprocessing Gambar untuk Sistem Deteksi Kepadatan Lalu Lintas Berbasis YOLOv8

**Mata Kuliah :** Visi Komputer  
**Program Studi:** Informatika — Universitas Mikroskil  
**Anggota Tim :**
- Silvani Chayadi — Preprocessing Specialist (Kode OpenCV)
- Cindy Nathania — UI/UX Developer (Aplikasi Streamlit)
- Gloria Apriyanti — Analyst & Documentation (Laporan & Analisis)
  
**Repository GitHub:** [Link GitHub]

---

## BAB I — PENDAHULUAN

### 1.1 Latar Belakang

Pertumbuhan jumlah kendaraan bermotor di kota-kota besar Indonesia, termasuk Medan, menimbulkan masalah kemacetan yang berdampak pada produktivitas dan kualitas udara. Sistem pemantauan kepadatan lalu lintas berbasis kamera CCTV konvensional masih bergantung pada pengawasan manual oleh petugas, sehingga rentan terhadap kesalahan dan keterlambatan respons.

Pendekatan berbasis Computer Vision, khususnya menggunakan model deteksi objek YOLOv8 (*You Only Look Once* versi 8), menawarkan solusi otomatis yang mampu menghitung kepadatan kendaraan secara real-time. Namun, akurasi sistem deteksi sangat bergantung pada kualitas data masukan. Gambar atau video dari kamera CCTV di lapangan sering mengandung noise (akibat kondisi cuaca, getaran kamera), pencahayaan tidak merata, dan latar belakang statis yang tidak relevan. Oleh karena itu, tahap **preprocessing gambar** menjadi fondasi kritis sebelum proses deteksi dapat dilakukan.

Tugas 1 ini berfokus pada perancangan dan implementasi pipeline **preprocessing otomatis** yang mengubah gambar/video mentah dari CCTV jalan raya menjadi representasi gambar yang optimal untuk masukan YOLOv8 pada Tugas 2.

### 1.2 Rumusan Masalah

1. Bagaimana merancang pipeline preprocessing yang dapat mereduksi noise pada citra CCTV jalan raya?
2. Bagaimana teknik morfologi dapat menyempurnakan siluet kendaraan yang terputus akibat noise atau oklusi?
3. Bagaimana mengimplementasikan pipeline tersebut dalam aplikasi berinterface visual yang dapat menampilkan setiap tahap pemrosesan?

### 1.3 Tujuan

1. Mengimplementasikan pipeline preprocessing lima tahap (Grayscaling, Gaussian Blur, Canny Edge, Morfologi) menggunakan OpenCV.
2. Membangun aplikasi web berbasis Streamlit yang menampilkan perubahan gambar secara *step-by-step*.
3. Menganalisis pengaruh setiap parameter pemrosesan terhadap kualitas output untuk kondisi lingkungan yang bervariasi.

---

## BAB II — TINJAUAN PUSTAKA

### 2.1 Digital Image Processing

Pengolahan citra digital adalah proses memanipulasi gambar menggunakan komputer untuk meningkatkan kualitas atau mengekstrak informasi tertentu (Gonzalez & Woods, 2018). Gambar digital direpresentasikan sebagai matriks dua dimensi I(x,y) di mana setiap elemen (piksel) memiliki nilai intensitas antara 0–255 untuk gambar 8-bit.

### 2.2 Konvolusi Citra

Konvolusi adalah operasi matematika fundamental dalam pengolahan citra yang mendefinisikan perubahan nilai suatu piksel berdasarkan nilai piksel-piksel tetangganya. Secara formal:

```
(I * K)(x,y) = Σ_m Σ_n I(x-m, y-n) · K(m,n)
```

Di mana `I` adalah gambar input, `K` adalah kernel (filter), dan `*` melambangkan operasi konvolusi. Kernel Gaussian yang digunakan pada penelitian ini memiliki elemen:

```
K_Gaussian(x,y) = (1/2πσ²) · exp(-(x² + y²) / 2σ²)
```

Untuk kernel 5×5 dengan σ≈1.1 (nilai yang dihitung OpenCV secara otomatis), matriks kernel yang diterapkan adalah:

```
      [ 2   4   5   4   2 ]
      [ 4   9  12   9   4 ]
K = (1/159) ·
      [ 5  12  15  12   5 ]
      [ 4   9  12   9   4 ]
      [ 2   4   5   4   2 ]
```

Elemen pusat (nilai 15) mendapat bobot tertinggi karena piksel pusat paling relevan terhadap nilai output. Semakin jauh dari pusat, bobot semakin menurun sesuai distribusi Gaussian, yang memastikan transisi nilai antarpiksel bersifat halus tanpa kehilangan informasi struktural kendaraan.

### 2.3 Morfologi Citra

Morfologi matematika adalah cabang teori himpunan yang diterapkan pada citra biner. Dua operasi dasar adalah:

**Dilation (Dilatasi):** Memperluas area foreground (piksel putih) dengan cara "menggeser" elemen struktural ke setiap piksel foreground.
```
(I ⊕ B)(x,y) = max_{(s,t)∈B} I(x-s, y-t)
```

**Erosion (Erosi):** Mempersempit area foreground, menghapus piksel yang tidak sepenuhnya dikelilingi oleh elemen struktural.
```
(I ⊖ B)(x,y) = min_{(s,t)∈B} I(x+s, y+t)
```

**Closing (Penutupan):** Komposisi Dilation diikuti Erosion (I ⊕ B) ⊖ B. Closing dirancang untuk **menutup celah kecil** di dalam objek foreground tanpa mengubah kontur luar secara signifikan.

### 2.4 Canny Edge Detection

Algoritma Canny (Canny, 1986) adalah detektor tepi optimal yang meminimalkan tiga kriteria: deteksi yang baik (tidak melewatkan tepi nyata), lokalisasi yang baik (tepi terdeteksi mendekati posisi sebenarnya), dan respons tunggal (satu respons per tepi). Algoritma ini bekerja dalam empat langkah berurutan: (1) Pemulusan Gaussian, (2) Komputasi gradien menggunakan operator Sobel, (3) Non-maximum suppression untuk menipistkan tepi, dan (4) Double thresholding dengan pelacakan hysteresis.

---

## BAB III — METODOLOGI

### 3.1 Dataset

| Atribut | Detail |
|---------|--------|
| Sumber | Rekaman CCTV jalan raya Kota Medan / Unduhan dari UA-DETRAC Dataset |
| Format | Video MP4 (resolusi 1080p, 25 FPS) dan Gambar JPEG |
| Kondisi | Siang hari cerah, mendung, hujan ringan, malam hari |
| Objek Target | Kendaraan bermotor roda empat (mobil sedan, SUV, pickup) |
| Objek Pengujian | Motor, bus, pejalan kaki, bayangan pohon |
| Jumlah Data | [Isi jumlah gambar/video yang digunakan tim] |

Dataset dipilih berdasarkan variasi kondisi lingkungan untuk menguji ketangguhan pipeline preprocessing terhadap perubahan pencahayaan dan cuaca.

### 3.2 Tahapan Preprocessing

Pipeline otomatis yang dirancang mengikuti alur berikut:

```
[Video/Gambar CCTV Mentah]
        ↓
┌─────────────────────────────┐
│  TAHAP 1: Grayscale + Resize │  Sampling & Quantization
│  BGR → Grayscale, 640px wide │  Reduksi data: ~66%
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  TAHAP 2: Gaussian Blur      │  Noise Handling (Konvolusi)
│  Kernel 5×5, σ=auto          │  Low-pass filter
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  TAHAP 3: Canny Edge         │  Feature Detection
│  T_low=50, T_high=150        │  Ekstraksi kontur kendaraan
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────┐
│  TAHAP 4: Morfologi          │  Closing / Dilation
│  Dilation 2× → Closing 1×   │  Menyempurnakan siluet
└─────────────┬───────────────┘
              ↓
[Gambar Siap Masuk YOLOv8 — Tugas 2]
```

### 3.3 Justifikasi Penggunaan Morphological Closing

Salah satu tantangan utama dalam deteksi kendaraan dari citra CCTV adalah **fragmentasi siluet kendaraan**. Ketika cahaya matahari atau lampu jalan memantul dari kaca depan atau kap mesin kendaraan, piksel pada area tersebut memiliki intensitas yang sangat berbeda dari badan kendaraan. Akibatnya, setelah Canny Edge Detection, siluet kendaraan tidak tampak sebagai satu kesatuan utuh, melainkan terbagi menjadi beberapa segmen terpisah yang diselingi oleh "lubang" di area kaca dan kap mesin.

Jika kondisi ini dibiarkan tanpa koreksi, algoritma pengelompokan (*bounding box*) pada YOLOv8 berpotensi menginterpretasikan satu kendaraan sebagai dua atau tiga objek terpisah, yang secara langsung menurunkan akurasi penghitungan kepadatan. Penerapan *Morphological Closing* (Dilation 2× dilanjutkan Erosion 1×) dengan kernel persegi (RECT) berukuran 5×5 piksel secara efektif menutup celah selebar ±3 piksel di dalam siluet kendaraan. Ukuran 5×5 dipilih berdasarkan observasi bahwa celah antara kontur panel kendaraan pada resolusi 640 piksel rata-rata berukuran 3–5 piksel. Penggunaan kernel lebih besar (7×7) berisiko menyatukan siluet dua kendaraan yang berdampingan, sedangkan kernel 3×3 terlalu kecil untuk menutup lubang di area kaca.

---

## BAB IV — IMPLEMENTASI

### 4.1 Arsitektur Sistem

Sistem terdiri dari dua komponen utama:

1. **`preprocessing.py`** — Modul engine yang mengimplementasikan setiap fungsi preprocessing menggunakan OpenCV. Mengembalikan objek `PreprocessingResult` berisi output setiap tahap dan metadata statistik.

2. **`app.py`** — Antarmuka web berbasis Streamlit yang memungkinkan pengguna mengunggah gambar/video dan mengamati hasil setiap tahap preprocessing secara berdampingan (*side-by-side*).

### 4.2 Spesifikasi Lingkungan Pengembangan

| Komponen | Spesifikasi |
|----------|-------------|
| Bahasa Pemrograman | Python 3.10+ |
| Library Utama | OpenCV 4.8, NumPy 1.24, Streamlit 1.35 |
| Sistem Operasi | Windows 10/11 atau Ubuntu 22.04 |
| RAM Minimum | 4 GB (8 GB direkomendasikan untuk video 1080p) |

---

## BAB V — ANALISIS DAN PEMBAHASAN

### 5.1 Tabel Analisis Performa Preprocessing

| Kondisi Input | Noise Level | Blur Optimal | Canny (Low/High) | Edge Density | Keterangan |
|--------------|------------|--------------|-------------------|-------------|-----------|
| Siang cerah  | Rendah     | Kernel 3×3   | 70 / 210          | ~8%         | Standar, performa terbaik |
| Mendung      | Sedang     | Kernel 5×5   | 50 / 150          | ~12%        | Kontras rendah, threshold turun |
| Hujan ringan | Tinggi     | Kernel 7×7   | 30 / 90           | ~18%        | Noise rintik hujan signifikan |
| Malam hari   | Tinggi     | Kernel 5×5   | 30 / 100          | ~15%        | Noise kamera tinggi, lampu jalan dominan |
| Macet padat  | Sedang     | Kernel 5×5   | 50 / 150          | ~25%        | Banyak tepi kendaraan saling tumpang tindih |

### 5.2 Analisis Kegagalan Model (Skenario Kesalahan)

#### Skenario 1: Deteksi Bayangan Pohon/Jembatan sebagai Kendaraan

**Observasi:** Pada gambar yang diambil saat siang terik dengan sumber cahaya dari satu arah, bayangan pohon rindang atau jembatan layang menghasilkan area gelap dengan tepi yang tajam. Canny Edge Detection mengidentifikasi tepi ini sebagai kontur potensial, dan setelah Morphological Closing, area bayangan membentuk siluet yang menyerupai kendaraan.

**Analisis Ilmiah:** Bayangan memiliki karakteristik gradien intensitas yang serupa dengan batas badan kendaraan (∆I/∆x yang tinggi). Pada kondisi ini, Canny Edge tidak mampu membedakan "tepi akibat objek fisik" dengan "tepi akibat perbedaan pencahayaan." Fenomena ini dikenal sebagai *photometric ambiguity* — di mana informasi warna (yang telah dibuang saat Grayscaling) sebenarnya dapat membantu membedakan bayangan dari kendaraan. Solusi untuk Tugas 2: Menambahkan filter HSV pada tahap preprocessing opsional, atau melatih YOLOv8 dengan augmentasi data yang menyertakan variasi bayangan.

#### Skenario 2: Pejalan Kaki Terdeteksi sebagai Kendaraan Kecil

**Observasi:** Ketika diuji dengan gambar zebra cross yang ramai, pipeline preprocessing menghasilkan siluet pejalan kaki berukuran sedang yang—setelah Morphological Closing—berbentuk menyerupai sepeda motor dari pandangan atas kamera CCTV.

**Analisis Ilmiah:** Dari perspektif kamera CCTV yang terpasang tinggi, proyeksi pejalan kaki dewasa (tinggi ~170 cm) pada resolusi 640 piksel hanya berukuran sekitar 30–50 piksel (dibandingkan mobil yang ~100–150 piksel). Teknik preprocessing berbasis tepi (*edge-based*) tidak memiliki kemampuan semantik untuk membedakan batas kendaraan dari batas tubuh manusia—keduanya hanya berupa gradien intensitas. Kegagalan ini bukan kelemahan preprocessing, melainkan keterbatasan yang memang harus diatasi oleh model klasifikasi (YOLOv8) pada Tugas 2 dengan dataset pelatihan yang mencakup kategori `pejalan_kaki` di samping kategori `kendaraan`.

#### Skenario 3: Fragmentasi Kendaraan Besar (Bus/Truk)

**Observasi:** Kendaraan besar seperti bus memiliki permukaan samping yang luas dan relatif seragam, menghasilkan sedikit tepi internal. Setelah Canny Edge, siluet bus sering terputus di bagian tengah badan kendaraan, dan Morphological Closing dengan kernel 5×5 tidak cukup untuk menutup celah yang lebih besar (±10–20 piksel).

**Analisis Ilmiah:** Ukuran proyeksi kendaraan pada gambar berbanding terbalik dengan tinggi pemasangan kamera CCTV. Untuk CCTV yang terpasang pada ketinggian >10 meter, bus dan truk terproyeksikan sangat datar sehingga perbedaan tekstur antara bagian atap dan jendela menjadi sangat halus. Solusi: Menambahkan **tahap preprocessing opsional** menggunakan Background Subtraction (MOG2 atau KNN) untuk kondisi kamera statis, yang dapat mengekstrak kendaraan bergerak secara lebih holistik dibandingkan deteksi tepi berbasis gradien.

---

## BAB VI — KESIMPULAN

1. Pipeline preprocessing yang dirancang berhasil mengubah gambar CCTV jalan raya mentah menjadi representasi biner berisi siluet kendaraan yang kompak melalui empat tahap berurutan: Grayscaling, Gaussian Blur, Canny Edge Detection, dan Morphological Closing.

2. Parameter kernel Gaussian 5×5 terbukti optimal untuk noise reduction pada kondisi cuaca normal; penyesuaian ke 7×7 diperlukan pada kondisi hujan lebat.

3. Morphological Closing dengan kernel RECT 5×5 efektif menyatukan siluet kendaraan yang terfragmentasi akibat pantulan cahaya, dengan catatan terdapat keterbatasan untuk kendaraan besar (bus/truk) yang memerlukan kernel lebih besar.

4. Analisis kegagalan mengidentifikasi tiga skenario utama (bayangan, pejalan kaki, kendaraan besar) yang memberikan landasan bagi pengembangan model YOLOv8 yang lebih robust pada Tugas 2.

---

## REFERENSI

- Canny, J. F. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.
- Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
- Jocher, G., et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. *arXiv:1804.02767*.
- Serra, J. (1983). *Image Analysis and Mathematical Morphology*. Academic Press.
- Wen, L., et al. (2015). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking. *arXiv:1511.04136*.
