# 🚦 Traffic Preprocessing Pipeline
### Setup Guide & Panduan Penggunaan Aplikasi

> **Tugas 1 — Visi Komputer | Universitas Mikroskil**  
> Pipeline otomatis preprocessing gambar/video CCTV jalan raya menggunakan OpenCV + Streamlit

---

## 📋 Daftar Isi

1. [Prasyarat Sistem](#-prasyarat-sistem)
2. [Instalasi](#-instalasi)
3. [Struktur Project](#-struktur-project)
4. [Menjalankan Aplikasi](#-menjalankan-aplikasi)
5. [Panduan Penggunaan UI](#-panduan-penggunaan-ui)
6. [Penjelasan Parameter](#-penjelasan-parameter)
7. [Tips Dataset](#-tips-dataset)
8. [Troubleshooting](#-troubleshooting)
9. [Kontribusi Tim](#-kontribusi-tim)

---

## ✅ Prasyarat Sistem

Pastikan komputer sudah terinstal software berikut sebelum memulai:

| Software | Versi Minimum | Cek Versi | Download |
|----------|--------------|-----------|----------|
| Python   | 3.10+        | `python --version` | [python.org](https://python.org) |
| pip      | 23.0+        | `pip --version`    | Sudah termasuk Python |
| Git      | 2.x          | `git --version`    | [git-scm.com](https://git-scm.com) |

> ⚠️ **Pengguna Windows:** Pastikan saat instalasi Python, centang opsi **"Add Python to PATH"**

---

## 🛠 Instalasi

### Langkah 1 — Clone Repository

```bash
git clone https://github.com/[username]/traffic-preprocessing.git
cd traffic-preprocessing
```

> Jika belum ada repo GitHub, cukup buat folder manual dan copy semua file ke dalamnya.

---

### Langkah 2 — Buat Virtual Environment

Sangat direkomendasikan agar library project ini tidak bentrok dengan instalasi Python lain di komputermu.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Jika berhasil, terminal akan menampilkan prefix `(venv)` di awal baris:
```
(venv) C:\Users\Nama\traffic-preprocessing>
```

---

### Langkah 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Proses ini akan menginstal:
- `opencv-python-headless` — Library computer vision utama
- `numpy` — Komputasi array/matriks
- `streamlit` — Framework web UI
- `Pillow` — Manipulasi gambar untuk konversi format

Estimasi waktu: **2–5 menit** tergantung kecepatan internet.

Untuk memverifikasi instalasi berhasil:
```bash
python -c "import cv2, streamlit, numpy; print('Semua library OK ✓')"
```

---

### Langkah 4 — Siapkan Dataset (Opsional tapi Direkomendasikan)

Buat folder untuk menyimpan gambar/video uji:
```bash
mkdir sample_data
```

Masukkan gambar jalan raya (`.jpg`, `.png`) atau video CCTV (`.mp4`, `.avi`) ke dalam folder `sample_data/`. Lihat bagian [Tips Dataset](#-tips-dataset) untuk sumber dataset gratis.

---

## 📁 Struktur Project

```
traffic-preprocessing/
│
├── app.py                  ← Aplikasi Streamlit (jalankan ini)
├── preprocessing.py        ← Engine OpenCV (pipeline utama)
├── requirements.txt        ← Daftar library Python
├── README.md               ← File ini
│
├── sample_data/            ← Sample gambar/video dataset di sini
│   ├── contoh_jalan.jpg
│   └── cctv_sample.mp4
│
└── laporan/
    └── draft_laporan.md    ← Kerangka laporan akademik
```

---

## ▶️ Menjalankan Aplikasi

Pastikan virtual environment sudah aktif `(venv)`, lalu jalankan:

```bash
streamlit run app.py
```

Terminal akan menampilkan output seperti ini:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: https://traffic-preprocessing.streamlit.app/
```

Buka browser dan akses **`http://localhost:8501`**
Untuk menghentikan aplikasi: tekan `Ctrl + C` di terminal.

---

## 🖥 Panduan Penggunaan UI

Aplikasi terdiri dari **3 Tab** utama:

---

### Tab 1 — 📷 Upload Gambar

Digunakan untuk memproses gambar statis (`.jpg`, `.jpeg`, `.png`, `.bmp`).

**Langkah-langkah:**

```
1. Klik tombol "Browse files" atau drag & drop gambar ke area upload
2. Tunggu proses otomatis selesai (biasanya < 1 detik)
3. Lihat hasil 5 panel secara berdampingan:

   [ORIGINAL] → [GRAYSCALE] → [BLURRED] → [CANNY EDGE] → [MORFOLOGI ✓]

4. Baca metadata statistik di panel kanan bawah:
   - Resolusi asli vs resolusi setelah resize
   - Jumlah edge pixel yang terdeteksi
   - Persentase area yang terisi setelah morfologi

5. Klik "⬇️ Download Hasil Morfologi" untuk menyimpan output akhir
```

**Contoh tampilan yang diharapkan:**

| Panel | Warna | Keterangan |
|-------|-------|-----------|
| Original | BGR (berwarna) | Gambar asli dari CCTV |
| Grayscale | Abu-abu | Setelah konversi 1-channel + resize |
| Gaussian Blur | Abu-abu (lebih halus) | Noise berkurang, tepi sedikit melembut |
| Canny Edge | Hitam-putih (garis tipis) | Hanya kontur kendaraan terlihat |
| Morfologi | Hitam-putih (siluet tebal) | Siluet kendaraan utuh & kompak |

---

### Tab 2 — 🎬 Upload Video

Digunakan untuk memproses file video CCTV (`.mp4`, `.avi`, `.mov`, `.mkv`).

**Langkah-langkah:**

```
1. Klik "Browse files" dan pilih file video
2. Informasi video otomatis tampil (total frame, FPS, resolusi)
3. Atur slider "Jumlah Frame Sample" (default: 5 frame)
   → Frame diambil secara merata dari awal, tengah, dan akhir video
4. Centang/hilangkan "Tampilkan semua tahap per frame" sesuai kebutuhan
5. Klik tombol "▶️ Proses Frame Sample dari Video"
6. Setiap frame ditampilkan dalam expander yang bisa dibuka/tutup
```

> ⚡ **Catatan Performa:** Video 1080p dengan 5 frame sample membutuhkan waktu ~3–8 detik. Untuk demo ke dosen, gunakan video ≤480p agar lebih cepat.

---

### Tab 3 — 📋 Tentang Pipeline

Menampilkan tabel ringkasan teknis setiap tahap preprocessing dan relevansinya dengan YOLOv8. Berguna untuk ditunjukkan ke dosen saat sesi tanya jawab.

---

## ⚙️ Penjelasan Parameter

Semua parameter dapat diubah secara realtime melalui **Sidebar kiri** tanpa perlu restart aplikasi.

### 📐 Tahap 1 — Resize Width

| Nilai | Kondisi Penggunaan |
|-------|-------------------|
| `320px` | Komputer RAM rendah atau koneksi lambat |
| `640px` ✓ | **Default & Direkomendasikan** — standar input YOLOv8 |
| `960px` | Jika detail kendaraan kecil perlu dipertahankan |
| `1280px` | Analisis detail, butuh RAM ≥8GB |

---

### 🌫️ Tahap 2 — Kernel Gaussian Blur

| Ukuran Kernel | Kondisi Optimal |
|--------------|----------------|
| `3×3` | Cuaca cerah, noise minimal |
| `5×5` ✓ | **Default** — cuaca normal, noise kamera standar |
| `7×7` | Hujan ringan, noise sedang |
| `9×9` | Hujan lebat, kondisi ekstrem |
| `11×11` | Malam hari dengan noise sensor tinggi |

> ⚠️ Semakin besar kernel, semakin halus gambar tetapi tepi kendaraan semakin kabur. Jangan gunakan kernel >7×7 kecuali noise sangat ekstrem.

---

### 🔍 Tahap 3 — Canny Threshold

**Aturan Rasio:** Selalu jaga rasio `High : Low` antara **2:1 hingga 3:1**

| Kondisi | Low | High | Rasio |
|---------|-----|------|-------|
| Siang cerah | 70 | 210 | 3:1 |
| Normal ✓ | 50 | 150 | 3:1 |
| Mendung | 40 | 120 | 3:1 |
| Malam / Hujan | 30 | 90 | 3:1 |

> Jika `High < Low × 2`, aplikasi akan menampilkan warning otomatis di sidebar.

---

### 🔷 Tahap 4 — Morfologi

**Iterasi Dilation:**

| Nilai | Efek |
|-------|------|
| `1×` | Menutup celah ~1–2 piksel (sambungan panel kecil) |
| `2×` ✓ | **Default** — menutup celah ~3–4 piksel (area kaca) |
| `3×` | Menutup celah ~5–6 piksel (kendaraan besar/bus) |
| `4×` | Agresif — hati-hati, bisa menyatukan dua kendaraan berbeda |

**Kernel Morfologi:**

| Ukuran | Gunakan Saat |
|--------|-------------|
| `3×3` | Kendaraan kecil (motor), resolusi rendah |
| `5×5` ✓ | **Default** — optimal untuk mobil di resolusi 640px |
| `7×7` | Kendaraan besar (bus, truk), resolusi tinggi |

---

## 📂 Tips Dataset

### Sumber Dataset Gratis

| Sumber | Tipe | Cara Akses |
|--------|------|-----------|
| **Google Street View** | Gambar | Buka maps.google.com → Street View di jalan raya → Screenshot |
| **UA-DETRAC** | Video + Anotasi | [detrac.smiles.info](http://detrac-db.rit.albany.edu) |
| **KITTI Dataset** | Video + 3D | [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti) |
| **YouTube CCTV** | Video | Cari "traffic CCTV footage free" → Download via yt-dlp |
| **Kaggle** | Gambar/Video | Cari "traffic vehicle dataset" di kaggle.com |

### Cara Cepat Dapat Dataset (< 10 Menit)

```bash
# Install yt-dlp untuk download video YouTube
pip install yt-dlp

# Download video CCTV lalu lintas (ganti URL dengan video yang relevan)
yt-dlp -f "best[height<=720]" "https://www.youtube.com/watch?v=XXXX" -o "sample_data/cctv_sample.mp4"
```

Alternatif paling cepat: **Screenshot Google Maps Street View** di persimpangan jalan sibuk Medan (Simpang Pos, Jalan Gatot Subroto, dll.) — ini sudah cukup untuk demo.

---

## 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'cv2'`

```bash
# Pastikan virtual environment aktif, lalu reinstall
pip install opencv-python --force-reinstall
```

---

### ❌ `streamlit: command not found`

```bash
# Coba dengan python -m
python -m streamlit run app.py
```

---

### ❌ Aplikasi lambat saat upload video besar

Kompres video terlebih dahulu menggunakan FFmpeg:
```bash
# Install FFmpeg dari ffmpeg.org, lalu:
ffmpeg -i input_video.mp4 -vf scale=640:-1 -crf 28 sample_data/compressed.mp4
```

---

### ❌ Gambar hasil preprocessing terlihat terlalu hitam (edge tidak muncul)

Coba turunkan nilai **Canny Threshold Bawah** di sidebar. Kemungkinan gambar memiliki kontras rendah (kondisi mendung/malam). Coba nilai `Low=20`, `High=60`.

---

### ❌ Siluet kendaraan masih terputus-putus setelah morfologi

Naikkan **Iterasi Dilation** ke `3×` dan **Kernel Morfologi** ke `7×7`. Kondisi ini biasanya terjadi pada kendaraan besar (bus/truk) atau gambar beresolusi sangat tinggi.

---

### ❌ `OSError: [Errno 28] No space left on device`

Hapus file video temporary yang tertinggal:
```bash
# Windows
del %TEMP%\*.mp4

# Linux/macOS
rm /tmp/tmp*.mp4
```

---

## 👥 Kontribusi Tim

| Nama | NIM | Tanggung Jawab |
|------|-----|----------------|
| Silvani Chayadi | 231112945 | Preprocessing Specialist — `preprocessing.py` |
| Cindy Nathania | 231111567 | UI/UX Developer — `app.py` |
| Gloria Apriyanti | 231111304 | Analyst & Documentation — Laporan & README |

---

## 📜 Lisensi & Penggunaan

Project ini dibuat untuk keperluan akademik Tugas 1 Mata Kuliah Visi Komputer, Universitas Mikroskil. Tidak untuk dipublikasikan atau dikomersialkan tanpa izin dosen pengampu.

---

<div align="center">

**🚦 Traffic Preprocessing Pipeline**  
Tugas 1 Visi Komputer — Universitas Mikroskil  
Dibuat dengan ❤️ menggunakan Python, OpenCV & Streamlit

</div>
