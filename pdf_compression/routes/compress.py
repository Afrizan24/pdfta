"""Routes: /preview and /compress."""

from __future__ import annotations

import io
import json
import os
import tempfile

from flask import Blueprint, Response, jsonify, request, send_file

from core.compressor import compress as pdf_compress
from core.features import extract_features

compress_bp = Blueprint("compress", __name__)

UPLOAD_FOLDER = "uploads"


@compress_bp.route("/preview", methods=["POST"])
def preview():
    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a .pdf"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        feats = extract_features(tmp_path)
        return jsonify({
            "pages": feats.pages,
            "file_size_bytes": feats.file_size_bytes,
            "total_text_len": feats.total_text_len,
            "total_images": feats.total_images,
            "avg_text_len_per_page": round(feats.avg_text_len_per_page, 2),
            "avg_images_per_page": round(feats.avg_images_per_page, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@compress_bp.route("/compress", methods=["POST"])
def compress():
    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a .pdf"}), 400

    def form_int(key, default):
        try:
            return int(request.form.get(key, default))
        except (ValueError, TypeError):
            return default

    def form_float(key, default):
        try:
            return float(request.form.get(key, default))
        except (ValueError, TypeError):
            return default

    def form_bool(key, default=False):
        val = request.form.get(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes", "on")

    mode        = request.form.get("mode", "AUTO").upper()
    dpi         = form_int("dpi", 150)
    jpeg_q      = form_int("jpeg_q", 75)
    grayscale   = form_bool("grayscale", default=False)
    garbage     = form_int("garbage", 4)
    deflate     = form_bool("deflate", default=True)
    clean       = form_bool("clean", default=True)
    pdf_setting = request.form.get("pdf_setting", "/ebook")
    scan_th     = form_int("scan_th", 20)
    digital_th  = form_int("digital_th", 200)
    min_img     = form_float("min_img", 1.0)
    max_size_gs = form_float("max_size_gs", 50.0)

    if mode not in ("AUTO", "DIGITAL", "SCAN", "HYBRID"):
        return jsonify({"error": f"Invalid mode: {mode}"}), 400

    # Save upload to a temp file, cleaned up after compress()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        pdf_bytes, info = pdf_compress(
            in_path=tmp_path,
            mode=mode,
            dpi=dpi,
            jpeg_quality=jpeg_q,
            grayscale=grayscale,
            garbage=garbage,
            deflate=deflate,
            clean=clean,
            pdf_setting=pdf_setting,
            scan_text_threshold=scan_th,
            digital_text_threshold=digital_th,
            min_images_for_scan=min_img,
            max_size_for_gs_mb=max_size_gs,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Embed metrics as a response header so JS can read them without a second request
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)

    response = send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="compressed.pdf",
    )
    # Pass metrics back via a custom header (JSON-encoded)
    response.headers["X-Compression-Info"] = json.dumps(info)
    response.headers["Access-Control-Expose-Headers"] = "X-Compression-Info"
    return response
