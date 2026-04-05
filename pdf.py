#!/usr/bin/env python3
"""
Adaptive PDF Compressor
-----------------------
Klasifikasi PDF: DIGITAL / SCAN / HYBRID berdasarkan heuristik.
- DIGITAL / HYBRID : optimasi struktur PDF + font subsetting via Ghostscript
- SCAN             : rasterisasi halaman ke JPEG lalu rebuild PDF baru

Dependensi: PyMuPDF (fitz), Pillow
Ghostscript opsional — jika tidak ada, hanya structural optimization yang berjalan.
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

import fitz          # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PdfFeatures:
    pages: int
    file_size_bytes: int
    total_text_len: int
    total_images: int
    avg_text_len_per_page: float
    avg_images_per_page: float


# ---------------------------------------------------------------------------
# Deteksi Ghostscript
# ---------------------------------------------------------------------------

def _find_ghostscript() -> Optional[str]:
    """Cari executable Ghostscript yang tersedia di sistem."""
    candidates = ["gs", "gswin64c", "gswin32c"]
    for name in candidates:
        if shutil.which(name):
            return name
    return None


GS_EXECUTABLE: Optional[str] = _find_ghostscript()


# ---------------------------------------------------------------------------
# Ekstraksi fitur
# ---------------------------------------------------------------------------

def extract_features(pdf_path: str) -> PdfFeatures:
    """Baca PDF dan ekstrak fitur untuk klasifikasi."""
    file_size = os.path.getsize(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        pages = doc.page_count
        total_text_len = 0
        total_images = 0

        for i in range(pages):
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            total_text_len += len(txt.strip())
            total_images += len(page.get_images(full=True))

        avg_text = total_text_len / max(pages, 1)
        avg_imgs = total_images / max(pages, 1)

        return PdfFeatures(
            pages=pages,
            file_size_bytes=file_size,
            total_text_len=total_text_len,
            total_images=total_images,
            avg_text_len_per_page=avg_text,
            avg_images_per_page=avg_imgs,
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Klasifikasi
# ---------------------------------------------------------------------------

def classify_pdf(
    feats: PdfFeatures,
    text_scan_threshold: int = 20,
    text_digital_threshold: int = 200,
    min_images_for_scan: float = 1.0,
) -> str:
    """
    Klasifikasi berbasis aturan (repeatable):
      SCAN    : teks sedikit + rata-rata >=1 gambar/halaman
      DIGITAL : teks banyak + gambar sedikit
      HYBRID  : kombinasi lain
    """
    avg_text = feats.avg_text_len_per_page
    avg_imgs = feats.avg_images_per_page

    if avg_text < text_scan_threshold and avg_imgs >= min_images_for_scan:
        return "SCAN"
    if avg_text >= text_digital_threshold and avg_imgs < 1.0:
        return "DIGITAL"
    return "HYBRID"


# ---------------------------------------------------------------------------
# Kompresi: structural optimization (PyMuPDF)
# ---------------------------------------------------------------------------

def optimize_pdf_structure(
    in_path: str,
    out_path: str,
    garbage: int = 4,
    deflate: bool = True,
    clean: bool = True,
) -> Dict[str, float]:
    """
    Optimasi struktur PDF tanpa mengubah konten:
    - garbage collection  : hapus objek yang tidak direferensikan
    - deflate             : rekompresi stream dengan Flate
    - clean               : rebuild xref table
    """
    t0 = time.perf_counter()
    doc = fitz.open(in_path)
    try:
        doc.save(
            out_path,
            garbage=garbage,
            deflate=deflate,
            clean=clean,
            incremental=False,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=1,
        )
    finally:
        doc.close()
    return {"time_ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Kompresi: font subsetting via Ghostscript
# ---------------------------------------------------------------------------

def font_subsetting_gs(
    in_path: str,
    out_path: str,
    pdf_setting: str = "/ebook",
    grayscale: bool = False,
) -> Dict[str, float]:
    """
    Font subsetting + optimasi via Ghostscript.
    Raise RuntimeError jika Ghostscript tidak tersedia.
    """
    if not GS_EXECUTABLE:
        raise RuntimeError(
            "Ghostscript tidak ditemukan. Install Ghostscript atau gunakan mode SCAN."
        )

    t0 = time.perf_counter()
    cmd = [
        GS_EXECUTABLE,
        "-sDEVICE=pdfwrite",
        "-dSubsetFonts=true",
        "-dEmbedAllFonts=false",
    ]
    
    if grayscale:
        cmd.extend([
            "-sColorConversionStrategy=Gray",
            "-dProcessColorModel=/DeviceGray"
        ])
        
    cmd.extend([
        f"-dPDFSETTINGS={pdf_setting}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={out_path}",
        in_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ghostscript error: {result.stderr.strip()}")

    return {"time_ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Kompresi: rasterisasi untuk SCAN PDF
# ---------------------------------------------------------------------------

def rasterize_scan_pdf_to_new_pdf(
    in_path: str,
    out_path: str,
    target_dpi: int = 150,
    jpeg_quality: int = 75,
    grayscale: bool = False,
) -> Dict[str, float]:
    """
    Untuk PDF hasil scan: render tiap halaman ke gambar JPEG lalu rebuild PDF baru.
    Memberikan pengurangan ukuran terbesar untuk dokumen scan.

    Catatan: menghilangkan layer teks OCR jika ada.
    """
    t0 = time.perf_counter()
    src = fitz.open(in_path)
    dst = fitz.open()

    try:
        zoom = target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(src.page_count):
            page = src.load_page(i)
            # Gunakan fitz.csGRAY langsung jika grayscale, atau csRGB untuk aman
            colorspace = fitz.csGRAY if grayscale else fitz.csRGB
            pix = page.get_pixmap(matrix=mat, colorspace=colorspace, alpha=False)

            # Sesuaikan mode dengan channel warna yang dihasilkan PyMuPDF
            mode = "L" if pix.n == 1 else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            jpg_bytes = buf.getvalue()

            rect = page.rect
            new_page = dst.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=jpg_bytes)

        dst.save(out_path, garbage=4, deflate=True, clean=True)
    finally:
        src.close()
        dst.close()

    return {"time_ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Hitung metrik
# ---------------------------------------------------------------------------

def compute_metrics(before_bytes: int, after_bytes: int, time_ms: float) -> Dict[str, float]:
    ratio = after_bytes / before_bytes if before_bytes > 0 else 0.0
    saving_pct = (1.0 - ratio) * 100.0
    throughput = (before_bytes / 1_048_576) / (time_ms / 1000.0) if time_ms > 0 else 0.0
    return {
        "before_bytes": float(before_bytes),
        "after_bytes": float(after_bytes),
        "ratio": ratio,
        "saving_pct": saving_pct,
        "time_ms": time_ms,
        "throughput_mb_s": throughput,
    }


# ---------------------------------------------------------------------------
# Fungsi utama kompresi (dipakai oleh app.py)
# ---------------------------------------------------------------------------

def compress(
    in_path: str,
    out_path: str,
    mode: str = "AUTO",
    dpi: int = 150,
    jpeg_quality: int = 75,
    grayscale: bool = False,
    garbage: int = 4,
    deflate: bool = True,
    clean: bool = True,
    pdf_setting: str = "/ebook",
    scan_text_threshold: int = 20,
    digital_text_threshold: int = 200,
    min_images_for_scan: float = 1.0,
    max_size_for_gs_mb: float = 50.0,  # Skip GS jika file > 50 MB
) -> Dict:
    """
    Entry point utama untuk kompresi PDF.
    Mengembalikan dict lengkap berisi fitur, metrik, dan info Ghostscript.
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"File tidak ditemukan: {in_path}")

    before = os.path.getsize(in_path)
    feats = extract_features(in_path)
    detected = classify_pdf(
        feats,
        text_scan_threshold=scan_text_threshold,
        text_digital_threshold=digital_text_threshold,
        min_images_for_scan=min_images_for_scan,
    )
    used_mode = detected if mode == "AUTO" else mode
    gs_used = True
    gs_available = GS_EXECUTABLE is not None

    time_ms = 0.0

    if used_mode in ("DIGITAL", "HYBRID"):
        temp_struct = out_path + ".struct.tmp.pdf"
        temp_gs = out_path + ".gs.tmp.pdf"
        
        candidates = [(before, in_path, False)] # (size, path, is_gs_used)
        
        try:
            # 1. Coba optimasi PyMuPDF (seringkali lebih baik untuk teks murni)
            s1 = optimize_pdf_structure(in_path, temp_struct, garbage=garbage,
                                        deflate=deflate, clean=clean)
            time_ms += s1["time_ms"]
            if os.path.exists(temp_struct):
                candidates.append((os.path.getsize(temp_struct), temp_struct, False))

            # 2. Coba Ghostscript secara langsung dari in_path (lebih baik kompresi gambar & font)
            skip_gs = not gs_available or (before > max_size_for_gs_mb * 1_048_576)
            if not skip_gs:
                try:
                    s2 = font_subsetting_gs(
                        in_path, 
                        temp_gs, 
                        pdf_setting=pdf_setting,
                        grayscale=grayscale
                    )
                    time_ms += s2["time_ms"]
                    if os.path.exists(temp_gs):
                        candidates.append((os.path.getsize(temp_gs), temp_gs, True))
                except RuntimeError:
                    pass

            # Pilih kandidat dengan ukuran terkecil
            candidates.sort(key=lambda x: x[0])
            best_size, best_path, used_gs = candidates[0]
            
            # Terapkan hasil terbaik
            if best_path != out_path:
                shutil.copy2(best_path, out_path)
            gs_used = used_gs

        finally:
            if os.path.exists(temp_struct):
                try: os.remove(temp_struct)
                except: pass
            if os.path.exists(temp_gs):
                try: os.remove(temp_gs)
                except: pass

    elif used_mode == "SCAN":
        s = rasterize_scan_pdf_to_new_pdf(
            in_path, out_path,
            target_dpi=dpi, jpeg_quality=jpeg_quality, grayscale=grayscale,
        )
        time_ms = s["time_ms"]

    else:
        raise ValueError(f"Mode tidak dikenal: {used_mode}")

    after = os.path.getsize(out_path)

    # Fallback keamanan global: Jika hasil akhir apa pun lebih besar dari file asli,
    # kembalikan file asli (size tidak membengkak).
    if after >= before:
        shutil.copy2(in_path, out_path)
        after = before
        gs_used = False
        time_ms = 0.0

    m = compute_metrics(before, after, time_ms)

    return {
        # fitur
        "pages": feats.pages,
        "file_size_bytes": feats.file_size_bytes,
        "total_text_len": feats.total_text_len,
        "total_images": feats.total_images,
        "avg_text_len_per_page": round(feats.avg_text_len_per_page, 2),
        "avg_images_per_page": round(feats.avg_images_per_page, 2),
        # klasifikasi
        "detected_class": detected,
        "mode_used": used_mode,
        # metrik
        "before_bytes": int(m["before_bytes"]),
        "after_bytes": int(m["after_bytes"]),
        "ratio": round(m["ratio"], 4),
        "saving_pct": round(m["saving_pct"], 2),
        "time_ms": round(m["time_ms"], 2),
        "throughput_mb_s": round(m["throughput_mb_s"], 2),
        # info sistem
        "gs_available": gs_available,
        "gs_used": gs_used,
        "gs_executable": GS_EXECUTABLE,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adaptive PDF Compressor — output tetap berupa .pdf yang bisa dibuka."
    )
    ap.add_argument("input",  help="Path PDF input")
    ap.add_argument("output", help="Path PDF output")

    # Threshold klasifikasi
    ap.add_argument("--scan-text-th",  type=int,   default=20,  help="Batas avg text/halaman → SCAN")
    ap.add_argument("--digital-text-th", type=int, default=200, help="Batas avg text/halaman → DIGITAL")
    ap.add_argument("--min-img-scan",  type=float, default=1.0, help="Min avg gambar/halaman untuk SCAN")

    # Parameter SCAN
    ap.add_argument("--dpi",       type=int, default=150, help="Target DPI rasterisasi (SCAN)")
    ap.add_argument("--jpeg-q",    type=int, default=75,  help="Kualitas JPEG 1-95 (SCAN)")
    ap.add_argument("--grayscale", action="store_true",   help="Konversi ke grayscale (SCAN, lebih kecil)")

    # Parameter DIGITAL/HYBRID
    ap.add_argument("--pdf-setting", default="/ebook",
                    choices=["/screen", "/ebook", "/printer", "/prepress"],
                    help="Setting Ghostscript PDFSETTINGS")
    ap.add_argument("--garbage",   type=int, default=4,   help="Level garbage collection 0-4")
    ap.add_argument("--no-deflate", action="store_true",  help="Nonaktifkan deflate recompression")
    ap.add_argument("--no-clean",   action="store_true",  help="Nonaktifkan clean xref")

    # Mode paksa
    ap.add_argument("--mode", choices=["AUTO", "DIGITAL", "SCAN", "HYBRID"], default="AUTO",
                    help="AUTO = otomatis deteksi; paksa mode tertentu jika perlu")

    args = ap.parse_args()

    result = compress(
        in_path=args.input,
        out_path=args.output,
        mode=args.mode,
        dpi=args.dpi,
        jpeg_quality=args.jpeg_q,
        grayscale=args.grayscale,
        garbage=args.garbage,
        deflate=not args.no_deflate,
        clean=not args.no_clean,
        pdf_setting=args.pdf_setting,
        scan_text_threshold=args.scan_text_th,
        digital_text_threshold=args.digital_text_th,
        min_images_for_scan=args.min_img_scan,
    )

    print("\n=== PDF FEATURES ===")
    print(f"  Halaman              : {result['pages']}")
    print(f"  Ukuran file (bytes)  : {result['file_size_bytes']}")
    print(f"  Total panjang teks   : {result['total_text_len']}")
    print(f"  Total gambar         : {result['total_images']}")
    print(f"  Avg text/halaman     : {result['avg_text_len_per_page']:.2f}")
    print(f"  Avg gambar/halaman   : {result['avg_images_per_page']:.2f}")
    print(f"  Kelas terdeteksi     : {result['detected_class']}")
    print(f"  Mode digunakan       : {result['mode_used']}")
    print(f"  Ghostscript          : {'tersedia (' + result['gs_executable'] + ')' if result['gs_available'] else 'tidak ditemukan'}")
    print(f"  GS digunakan         : {result['gs_used']}")

    print("\n=== HASIL ===")
    print(f"  Sebelum (bytes)      : {result['before_bytes']:,}")
    print(f"  Sesudah (bytes)      : {result['after_bytes']:,}")
    print(f"  Rasio kompresi       : {result['ratio']:.4f}")
    print(f"  Penghematan (%)      : {result['saving_pct']:.2f}%")
    print(f"  Waktu (ms)           : {result['time_ms']:.2f}")
    print(f"  Throughput (MB/s)    : {result['throughput_mb_s']:.2f}")


if __name__ == "__main__":
    main()
