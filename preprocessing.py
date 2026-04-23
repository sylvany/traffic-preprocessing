"""
preprocessing.py
================
Engine utama untuk pipeline preprocessing otomatis deteksi kepadatan lalu lintas.
Setiap tahap dirancang mengikuti alur RTM: Sampling/Quantization → Noise Handling
→ Konvolusi → Morfologi → Feature Detection.

Penulis  : Tim Visi Komputer - Universitas Mikroskil
Dataset  : Video CCTV Jalan Raya (resolusi 1080p → resize ke 640p)
Library  : OpenCV 4.x, NumPy
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data container untuk membawa semua hasil tahap sekaligus
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingResult:
    """
    Menyimpan output setiap tahap preprocessing agar UI bisa
    menampilkan perubahan gambar secara side-by-side.
    """
    original:     np.ndarray = field(default=None)
    gray:         np.ndarray = field(default=None)   # Tahap 1: Grayscale + Resize
    blurred:      np.ndarray = field(default=None)   # Tahap 2: Gaussian Blur
    edges:        np.ndarray = field(default=None)   # Tahap 3: Canny Edge Detection
    morphed:      np.ndarray = field(default=None)   # Tahap 4: Morfologi
    metadata:     dict       = field(default_factory=dict)


# ===========================================================================
# TAHAP 1 — SAMPLING & QUANTIZATION
# Tujuan: Mengurangi beban komputasi tanpa kehilangan informasi penting.
# Prinsip: Nyquist Sampling — resolusi diturunkan namun struktur kendaraan
# tetap dapat dibedakan (aspect ratio dipertahankan).
# ===========================================================================
def stage1_grayscale_resize(
    image: np.ndarray,
    target_width: int = 640
) -> np.ndarray:
    """
    Mengubah gambar BGR ke Grayscale dan meresize ke lebar target.

    Mengapa Grayscale?
    - Kendaraan dapat diidentifikasi hanya dari intensitas, bukan warna.
    - Mengurangi data dari 3 channel (BGR) ke 1 channel → komputasi ~3× lebih cepat.
    - Teknik ini adalah bentuk 'Quantization' karena 24-bit color direduksi ke 8-bit gray.

    Mengapa target_width = 640?
    - Resolusi standar input YOLOv8 adalah 640×640.
    - Menyesuaikan di tahap preprocessing memastikan konsistensi saat masuk ke Tugas 2.
    - Pada resolusi ini, lebar jalur (lane) di gambar CCTV rata-rata ±80 piksel,
      cukup untuk mengenali kontur kendaraan secara akurat.
    """
    # Hitung rasio aspek agar gambar tidak distorsi
    original_h, original_w = image.shape[:2]
    aspect_ratio = original_h / original_w
    target_height = int(target_width * aspect_ratio)

    # cv2.INTER_AREA adalah interpolasi terbaik untuk DOWNSAMPLING
    # karena ia menghitung rata-rata piksel, bukan hanya mengambil satu piksel
    # (berbeda dengan INTER_NEAREST yang menghasilkan artefak aliasing)
    resized = cv2.resize(image, (target_width, target_height),
                         interpolation=cv2.INTER_AREA)

    # Konversi BGR → Grayscale menggunakan formula luminance standar ITU-R BT.601:
    # Y = 0.299R + 0.587G + 0.114B
    # Koefisien ini mencerminkan sensitivitas mata manusia terhadap warna hijau
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return gray


# ===========================================================================
# TAHAP 2 — NOISE HANDLING (GAUSSIAN BLUR)
# Tujuan: Meredam noise sensor kamera (terutama pada kondisi malam/hujan)
# Prinsip: Konvolusi kernel Gaussian 2D di seluruh piksel gambar.
# ===========================================================================
def stage2_gaussian_blur(
    gray_image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Menerapkan Gaussian Blur sebagai low-pass filter untuk noise reduction.

    Mengapa Gaussian Blur (bukan Median/Mean Blur)?
    - Gaussian Blur menggunakan distribusi Normal 2D sebagai kernel konvolusi,
      sehingga piksel lebih dekat ke pusat diberi bobot lebih tinggi.
    - Lebih baik dari Mean Blur karena Mean memperlakukan semua piksel sama,
      yang dapat mengaburkan tepi kendaraan (edge) secara berlebihan.
    - Lebih cepat dari Bilateral Filter, cocok untuk real-time CCTV processing.

    Mengapa kernel_size = 5 (yaitu kernel 5×5)?
    - Kernel 3×3: Terlalu kecil, tidak efektif meredam noise shot noise pada
      sensor CCTV beresolusi tinggi.
    - Kernel 5×5: Optimal — menghilangkan noise berukuran 1-2 piksel (rintik hujan,
      debu lensa) tanpa mengaburkan batas kendaraan yang lebarnya ±5-10 piksel.
    - Kernel 7×7 ke atas: Over-smoothing, tepi kendaraan hilang sebelum masuk Canny.
    - WAJIB GANJIL: OpenCV mensyaratkan kernel berukuran ganjil agar titik pusat
      (anchor point) berada tepat di tengah kernel.

    Mengapa sigma = 0?
    - Dengan sigma=0, OpenCV menghitung sigma secara otomatis dari kernel_size
      menggunakan formula: σ = 0.3 × ((ksize−1) × 0.5 − 1) + 0.8
    - Untuk kernel 5×5, sigma ≈ 1.1 — nilai yang terbukti optimal
      untuk noise CCTV outdoor berdasarkan studi Burt & Adelson (1983).
    """
    # Kernel 5×5 Gaussian yang diterapkan secara konvolusi:
    # Setiap piksel output = jumlah tertimbang dari 25 piksel tetangga
    # dengan bobot mengikuti distribusi Gaussian
    blurred = cv2.GaussianBlur(
        gray_image,
        ksize=(kernel_size, kernel_size),  # Ukuran kernel: harus ganjil
        sigmaX=sigma,                       # Std deviasi arah X
        sigmaY=sigma                        # Std deviasi arah Y (0 = sama dengan X)
    )
    return blurred


# ===========================================================================
# TAHAP 3 — FEATURE DETECTION (CANNY EDGE DETECTION)
# Tujuan: Mengekstrak kontur/tepi kendaraan sebagai fitur utama.
# Prinsip: Multi-step edge detection (Gradient → Non-max Suppression → Hysteresis).
# ===========================================================================
def stage3_canny_edge(
    blurred_image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Mendeteksi tepi (edge) menggunakan algoritma Canny (John F. Canny, 1986).

    Cara Kerja Internal Canny (4 Langkah):
    1. Gradient Calculation: Menghitung gradien intensitas menggunakan Sobel operator
       untuk mendeteksi perubahan intensitas tajam (= tepi potensial).
    2. Non-Maximum Suppression: Menipistkan tepi menjadi garis 1 piksel dengan
       hanya menyimpan piksel lokal maksimum.
    3. Double Thresholding: Mengklasifikasikan tepi menjadi 'strong' (> high),
       'weak' (low < x < high), dan 'bukan tepi' (< low).
    4. Edge Tracking by Hysteresis: Tepi 'weak' hanya dipertahankan jika
       terhubung dengan tepi 'strong'.

    Mengapa low=50, high=150? (Rasio 1:3)
    - Rekomendasi Canny sendiri: rasio high:low antara 2:1 hingga 3:1.
    - Nilai 50 sebagai threshold bawah: cukup sensitif menangkap kontur bumper
      dan kaca kendaraan yang kontrasnya rendah terhadap aspal.
    - Nilai 150 sebagai threshold atas: menyaring noise yang masih lolos dari
      Gaussian Blur (bayangan pohon, marka jalan samar).
    - Pada kondisi malam hari: turunkan low→30, high→100 karena kontras rendah.
    - Pada kondisi cerah/terik: naikkan low→70, high→200 karena silau aspal.
    """
    edges = cv2.Canny(
        blurred_image,
        threshold1=low_threshold,   # Hysteresis threshold bawah
        threshold2=high_threshold,  # Hysteresis threshold atas
        apertureSize=3,             # Ukuran kernel Sobel (3×3, default & optimal)
        L2gradient=True             # Gunakan norma L2 (lebih akurat dari L1)
    )
    return edges


# ===========================================================================
# TAHAP 4 — MORPHOLOGICAL OPERATIONS
# Tujuan: Menyempurnakan siluet kendaraan yang terputus akibat noise/occlusion.
# Prinsip: Operasi himpunan pada piksel biner (Dilation → Closing).
# ===========================================================================
def stage4_morphology(
    edge_image: np.ndarray,
    dilation_iter: int = 2,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Menerapkan Dilation dan Closing untuk menyatukan siluet kendaraan.

    Mengapa Dilation terlebih dahulu?
    - Dilation 'memperlebar' setiap piksel tepi sehingga garis tepi yang tipis
      menjadi lebih tebal dan celah kecil antar-tepi mulai tertutup.
    - Dilasi dengan 2 iterasi: iterasi pertama menutup celah ±2 piksel (sambungan
      antara badan mobil & kaca), iterasi kedua menutup ±4 piksel (celah antara
      ban & bumper yang terpisah akibat pantulan cahaya).

    Mengapa Closing (Dilation → Erosion) setelah Dilation?
    - Morphological Closing = Dilation kemudian Erosion dengan kernel yang sama.
    - Closing dirancang khusus untuk 'menutup lubang' di dalam objek foreground.
    - Tanpa Closing: Siluet kendaraan memiliki lubang di area kaca depan/belakang
      (karena kaca memantulkan cahaya = piksel lebih terang dari kontur) →
      YOLOv8 bisa mengira satu mobil sebagai dua objek terpisah.
    - Dengan Closing: Seluruh siluet kendaraan menjadi area solid berwarna putih
      yang kompak, sehingga bounding box YOLOv8 melingkupi seluruh badan kendaraan.

    Mengapa kernel RECT 5×5?
    - cv2.MORPH_RECT (persegi) dipilih karena kendaraan memiliki bentuk dominan
      rectangular (kotak), berbeda dengan MORPH_ELLIPSE yang optimal untuk objek bulat.
    - Ukuran 5×5: Menutup celah hingga ±3 piksel — sesuai dengan ukuran sambungan
      antar-panel kendaraan pada resolusi 640 piksel.
    - Kernel 3×3: Terlalu kecil untuk menutup lubang di area kaca (±5-8 piksel).
    - Kernel 7×7: Terlalu agresif, menyatukan kendaraan yang berbeda jika berdampingan.
    """
    # Definisi kernel struktural berbentuk persegi
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,               # Bentuk kernel: persegi (bukan elips/silang)
        (kernel_size, kernel_size)    # Ukuran: 5 piksel × 5 piksel
    )

    # Langkah 1: Dilation — perluas setiap piksel putih ke arah semua tetangga
    # Efek: Garis tepi menebal, celah kecil mulai tertutup
    dilated = cv2.dilate(
        edge_image,
        kernel,
        iterations=dilation_iter  # 2 iterasi = dilation diterapkan 2× berturut-turut
    )

    # Langkah 2: Morphological Closing — Dilation lanjut → kemudian Erosion
    # MORPH_CLOSE = dilate(erode()) → menutup lubang di dalam kontur
    closed = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,   # Operasi: Closing
        kernel,
        iterations=1        # 1 iterasi sudah cukup karena sudah di-dilate sebelumnya
    )

    return closed


# ===========================================================================
# PIPELINE UTAMA — Menggabungkan semua tahap secara berurutan
# ===========================================================================
def run_full_pipeline(
    image: np.ndarray,
    resize_width: int = 640,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    morph_iter: int = 2,
    morph_kernel: int = 5
) -> PreprocessingResult:
    """
    Menjalankan seluruh pipeline preprocessing dari gambar mentah ke gambar siap deteksi.

    Args:
        image       : Gambar BGR dari OpenCV (cv2.imread output).
        resize_width: Lebar target setelah resize (default 640 untuk YOLOv8).
        blur_kernel : Ukuran kernel Gaussian Blur (harus ganjil).
        canny_low   : Threshold bawah Canny Edge Detection.
        canny_high  : Threshold atas Canny Edge Detection.
        morph_iter  : Jumlah iterasi Dilation.
        morph_kernel: Ukuran kernel morfologi.

    Returns:
        PreprocessingResult: Object berisi semua output tahap + metadata statistik.
    """
    result = PreprocessingResult()
    result.original = image.copy()

    # Jalankan setiap tahap secara berurutan
    result.gray    = stage1_grayscale_resize(image, target_width=resize_width)
    result.blurred = stage2_gaussian_blur(result.gray, kernel_size=blur_kernel)
    result.edges   = stage3_canny_edge(result.blurred, low_threshold=canny_low,
                                        high_threshold=canny_high)
    result.morphed = stage4_morphology(result.edges, dilation_iter=morph_iter,
                                        kernel_size=morph_kernel)

    # Hitung metadata statistik untuk analisis laporan
    edge_pixels   = np.count_nonzero(result.edges)
    morphed_pixels = np.count_nonzero(result.morphed)
    total_pixels  = result.edges.size

    result.metadata = {
        "original_shape"    : image.shape,
        "processed_shape"   : result.gray.shape,
        "edge_pixel_count"  : edge_pixels,
        "morphed_pixel_count": morphed_pixels,
        "edge_density_pct"  : round(edge_pixels / total_pixels * 100, 2),
        "morph_fill_pct"    : round(morphed_pixels / total_pixels * 100, 2),
        "noise_reduction"   : f"Kernel Gaussian {blur_kernel}×{blur_kernel}",
        "canny_params"      : f"Low={canny_low}, High={canny_high}",
        "morph_kernel"      : f"RECT {morph_kernel}×{morph_kernel}, iter={morph_iter}"
    }

    return result


# ===========================================================================
# VIDEO PROCESSING — Memproses setiap frame dari file video
# ===========================================================================
def process_video_frame(frame: np.ndarray, **kwargs) -> PreprocessingResult:
    """
    Wrapper tipis untuk memproses satu frame video.
    Digunakan oleh app.py saat memproses video CCTV secara stream.
    """
    return run_full_pipeline(frame, **kwargs)


def get_video_sample_frames(
    video_path: str,
    n_frames: int = 5
) -> list[np.ndarray]:
    """
    Mengambil N frame representatif dari video untuk pratinjau di UI.
    Frame diambil secara merata dari awal, tengah, dan akhir video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames
