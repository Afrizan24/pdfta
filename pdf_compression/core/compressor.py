"""
PDF compression — multi-pass pipeline for maximum size reduction.

Strategy per mode
-----------------
DIGITAL  : (1) pikepdf image recompress  (2) PyMuPDF struct opt  (3) GS font-subset
           → pick smallest result
SCAN     : rasterise pages → progressive JPEG (auto-grayscale per page)
           → PyMuPDF struct opt on result
HYBRID   : (1) pikepdf image recompress  (2) PyMuPDF struct opt  (3) GS font-subset
           → pick smallest result  (same as DIGITAL but scan pages get rasterised first)

All modes: metadata stripped, duplicate objects removed via pikepdf.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import fitz          # PyMuPDF
import pikepdf
from PIL import Image

from core.classifier import classify_pdf
from core.features import extract_features
from core.ghostscript import GS_EXECUTABLE, font_subsetting_gs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp(dir: str, name: str) -> str:
    return os.path.join(dir, name)


def _size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


# ---------------------------------------------------------------------------
# Pass 1 — pikepdf: recompress embedded images + strip metadata + linearize
# ---------------------------------------------------------------------------

def _recompress_image(img_data: bytes, jpeg_quality: int, grayscale: bool) -> Optional[bytes]:
    """
    Re-encode a single image with Pillow.
    Returns compressed bytes, or None if result is not smaller.
    """
    try:
        buf_in = io.BytesIO(img_data)
        img = Image.open(buf_in)

        # Convert to RGB/L for JPEG output
        if grayscale or img.mode in ("L", "LA", "P"):
            img = img.convert("L")
            mode_out = "L"
        else:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img = img.convert("RGB")
            mode_out = "RGB"

        buf_out = io.BytesIO()
        img.save(
            buf_out,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True,
            subsampling=2 if jpeg_quality < 85 else 0,
        )
        result = buf_out.getvalue()
        return result if len(result) < len(img_data) else None
    except Exception:
        return None


def pikepdf_recompress(
    in_path: str,
    out_path: str,
    jpeg_quality: int = 72,
    grayscale: bool = False,
) -> Dict[str, float]:
    """
    Walk every image XObject in the PDF and re-encode with JPEG.
    Also strips metadata and removes duplicate objects.
    """
    t0 = time.perf_counter()

    with pikepdf.open(in_path) as pdf:
        # Strip document metadata to save space
        with pdf.open_metadata() as meta:
            # Keep only essential keys
            keys_to_remove = [k for k in meta if k not in (
                "dc:title", "dc:creator", "pdf:Producer"
            )]
            for k in keys_to_remove:
                try:
                    del meta[k]
                except Exception:
                    pass

        # Recompress every image XObject
        for page in pdf.pages:
            if "/Resources" not in page:
                continue
            resources = page["/Resources"]
            if "/XObject" not in resources:
                continue
            xobjects = resources["/XObject"]
            for name in list(xobjects.keys()):
                xobj = xobjects[name]
                try:
                    if xobj.get("/Subtype") != "/Image":
                        continue
                    # Skip tiny images (icons, bullets)
                    w = int(xobj.get("/Width", 0))
                    h = int(xobj.get("/Height", 0))
                    if w * h < 4096:
                        continue

                    raw = xobj.read_raw_bytes()
                    compressed = _recompress_image(raw, jpeg_quality, grayscale)
                    if compressed is None:
                        continue

                    # Replace image stream with recompressed JPEG
                    xobj.stream_dict["/Filter"] = pikepdf.Name("/DCTDecode")
                    xobj.stream_dict["/ColorSpace"] = (
                        pikepdf.Name("/DeviceGray")
                        if (grayscale or xobj.get("/ColorSpace") == "/DeviceGray")
                        else pikepdf.Name("/DeviceRGB")
                    )
                    xobj.stream_dict["/BitsPerComponent"] = 8
                    # Remove any existing decode parms
                    for key in ("/DecodeParms", "/Decode"):
                        if key in xobj.stream_dict:
                            del xobj.stream_dict[key]
                    xobj.write(compressed, filter=pikepdf.Name("/DCTDecode"))
                except Exception:
                    continue

        pdf.save(
            out_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
            recompress_flate=True,
            linearize=False,
        )

    return {"time_ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Pass 2 — PyMuPDF: structural optimisation
# ---------------------------------------------------------------------------

def optimize_pdf_structure(
    in_path: str,
    out_path: str,
    garbage: int = 4,
    deflate: bool = True,
    clean: bool = True,
) -> Dict[str, float]:
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
# Pass 3 — Ghostscript: font subsetting (DIGITAL/HYBRID)
# ---------------------------------------------------------------------------
# (imported from core.ghostscript)


# ---------------------------------------------------------------------------
# SCAN rasterisation — progressive JPEG + auto grayscale per page
# ---------------------------------------------------------------------------

def _is_page_grayscale(page: fitz.Page, sample_size: int = 50) -> bool:
    """
    Quick heuristic: render a tiny thumbnail and check colour variance.
    Returns True if the page is effectively grayscale.
    """
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(0.15, 0.15), colorspace=fitz.csRGB, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        r, g, b = img.split()
        import statistics
        # If R≈G≈B across the image, it's grayscale
        r_vals = list(r.getdata())
        g_vals = list(g.getdata())
        b_vals = list(b.getdata())
        diff_rg = statistics.mean(abs(rv - gv) for rv, gv in zip(r_vals, g_vals))
        diff_rb = statistics.mean(abs(rv - bv) for rv, bv in zip(r_vals, b_vals))
        return diff_rg < 8 and diff_rb < 8
    except Exception:
        return False


def rasterize_scan_pdf(
    in_path: str,
    out_path: str,
    target_dpi: int = 150,
    jpeg_quality: int = 75,
    grayscale: bool = False,
) -> Dict[str, float]:
    """
    Render each page to progressive JPEG and rebuild as a new PDF.
    Auto-detects grayscale pages for extra savings unless grayscale=True (force all).
    """
    t0 = time.perf_counter()
    src = fitz.open(in_path)
    dst = fitz.open()

    try:
        zoom = target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for i in range(src.page_count):
            page = src.load_page(i)

            # Per-page grayscale detection (skip if user forced grayscale)
            page_gray = grayscale or _is_page_grayscale(page)

            colorspace = fitz.csGRAY if page_gray else fitz.csRGB
            pix = page.get_pixmap(matrix=mat, colorspace=colorspace, alpha=False)

            mode = "L" if pix.n == 1 else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            buf = io.BytesIO()
            img.save(
                buf,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=True,
                subsampling=2 if jpeg_quality < 85 else 0,
            )

            rect = page.rect
            new_page = dst.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=buf.getvalue())

        dst.save(out_path, garbage=4, deflate=True, clean=True)
    finally:
        src.close()
        dst.close()

    return {"time_ms": (time.perf_counter() - t0) * 1000.0}


# ---------------------------------------------------------------------------
# Metrics
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
# Main entry point
# ---------------------------------------------------------------------------

def compress(
    in_path: str,
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
    max_size_for_gs_mb: float = 50.0,
) -> Tuple[bytes, Dict]:
    """
    Multi-pass PDF compression. Returns (pdf_bytes, info_dict).
    Nothing is persisted after this function returns.
    """
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"File not found: {in_path}")

    before = os.path.getsize(in_path)
    feats = extract_features(in_path)
    detected = classify_pdf(
        feats,
        text_scan_threshold=scan_text_threshold,
        text_digital_threshold=digital_text_threshold,
        min_images_for_scan=min_images_for_scan,
    )
    used_mode = detected if mode == "AUTO" else mode
    gs_used = False
    gs_available = GS_EXECUTABLE is not None
    time_ms = 0.0

    tmp_dir = tempfile.mkdtemp(prefix="pdfcomp_")
    try:
        # ── SCAN ────────────────────────────────────────────────────────────
        if used_mode == "SCAN":
            raster_path = _tmp(tmp_dir, "raster.pdf")
            s = rasterize_scan_pdf(
                in_path, raster_path,
                target_dpi=dpi, jpeg_quality=jpeg_quality, grayscale=grayscale,
            )
            time_ms += s["time_ms"]

            # Second pass: struct opt on the rasterised result
            struct_path = _tmp(tmp_dir, "raster_struct.pdf")
            if os.path.exists(raster_path):
                try:
                    s2 = optimize_pdf_structure(raster_path, struct_path,
                                                garbage=garbage, deflate=deflate, clean=clean)
                    time_ms += s2["time_ms"]
                except Exception:
                    pass

            # Pick best between raster and raster+struct
            candidates: List[Tuple[int, str]] = [(before, in_path)]
            if _size(raster_path):
                candidates.append((_size(raster_path), raster_path))
            if _size(struct_path):
                candidates.append((_size(struct_path), struct_path))
            candidates.sort(key=lambda x: x[0])
            best_size, best_path = candidates[0]

        # ── DIGITAL / HYBRID ─────────────────────────────────────────────────
        elif used_mode in ("DIGITAL", "HYBRID"):
            pike_path   = _tmp(tmp_dir, "pike.pdf")
            struct_path = _tmp(tmp_dir, "struct.pdf")
            gs_path     = _tmp(tmp_dir, "gs.pdf")

            # Recompress quality slightly lower than user setting for embedded images
            img_q = max(jpeg_quality - 5, 40)

            candidates = [(before, in_path)]

            # Pass A — pikepdf image recompress + metadata strip
            try:
                sA = pikepdf_recompress(in_path, pike_path,
                                        jpeg_quality=img_q, grayscale=grayscale)
                time_ms += sA["time_ms"]
                if _size(pike_path):
                    candidates.append((_size(pike_path), pike_path))
            except Exception:
                pass

            # Pass B — PyMuPDF struct opt (on best so far)
            best_so_far = min(candidates, key=lambda x: x[0])[1]
            try:
                sB = optimize_pdf_structure(best_so_far, struct_path,
                                            garbage=garbage, deflate=deflate, clean=clean)
                time_ms += sB["time_ms"]
                if _size(struct_path):
                    candidates.append((_size(struct_path), struct_path))
            except Exception:
                pass

            # Pass C — Ghostscript font subsetting (on original, GS does its own image opt)
            skip_gs = not gs_available or (before > max_size_for_gs_mb * 1_048_576)
            if not skip_gs:
                try:
                    sC = font_subsetting_gs(in_path, gs_path,
                                            pdf_setting=pdf_setting, grayscale=grayscale)
                    time_ms += sC["time_ms"]
                    if _size(gs_path):
                        candidates.append((_size(gs_path), gs_path))
                        gs_used = True
                except RuntimeError:
                    pass

            candidates.sort(key=lambda x: x[0])
            best_size, best_path = candidates[0]
            # If GS result wasn't the winner, mark gs_used False
            if best_path != gs_path:
                gs_used = False

        else:
            raise ValueError(f"Unknown mode: {used_mode}")

        # ── Safety fallback ──────────────────────────────────────────────────
        if best_size >= before:
            best_path = in_path
            best_size = before
            gs_used = False
            time_ms = 0.0

        with open(best_path, "rb") as f:
            pdf_bytes = f.read()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    m = compute_metrics(before, best_size, time_ms)

    info = {
        "pages": feats.pages,
        "file_size_bytes": feats.file_size_bytes,
        "total_text_len": feats.total_text_len,
        "total_images": feats.total_images,
        "avg_text_len_per_page": round(feats.avg_text_len_per_page, 2),
        "avg_images_per_page": round(feats.avg_images_per_page, 2),
        "detected_class": detected,
        "mode_used": used_mode,
        "before_bytes": int(m["before_bytes"]),
        "after_bytes": int(best_size),
        "ratio": round(m["ratio"], 4),
        "saving_pct": round(m["saving_pct"], 2),
        "time_ms": round(m["time_ms"], 2),
        "throughput_mb_s": round(m["throughput_mb_s"], 2),
        "gs_available": gs_available,
        "gs_used": gs_used,
        "gs_executable": GS_EXECUTABLE,
    }

    return pdf_bytes, info
