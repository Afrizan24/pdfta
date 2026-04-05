"""
Flask server untuk Adaptive PDF Compressor
------------------------------------------
Endpoint:
  GET  /            → halaman utama (index.html)
  POST /compress    → upload + kompres PDF, kembalikan JSON hasil
  GET  /download/<id> → unduh file hasil kompresi
  GET  /status      → info sistem (GS tersedia, versi Python, dll)
"""

import os
import uuid
import shutil
from flask import Flask, render_template, request, send_file, jsonify

from pdf import compress as pdf_compress, extract_features, GS_EXECUTABLE

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Halaman utama
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Preview klasifikasi (ekstrak fitur saat upload)
# ---------------------------------------------------------------------------

@app.route("/preview", methods=["POST"])
def preview():
    # ── Validasi file ──
    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File harus berformat .pdf"}), 400

    # ── Simpan file sementara ──
    import tempfile
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

@app.route("/status")
def status():
    return jsonify({
        "gs_available": GS_EXECUTABLE is not None,
        "gs_executable": GS_EXECUTABLE,
    })


# ---------------------------------------------------------------------------
# Kompresi PDF
# ---------------------------------------------------------------------------

@app.route("/compress", methods=["POST"])
def compress():
    # ── Validasi file ──
    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File harus berformat .pdf"}), 400

    # ── Baca parameter form ──
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

    def form_bool(key):
        return request.form.get(key, "false").lower() == "true"

    mode        = request.form.get("mode", "AUTO").upper()
    dpi         = form_int("dpi", 150)
    jpeg_q      = form_int("jpeg_q", 75)
    grayscale   = form_bool("grayscale")
    garbage     = form_int("garbage", 4)
    deflate     = form_bool("deflate")  if "deflate"  in request.form else True
    clean       = form_bool("clean")    if "clean"    in request.form else True
    pdf_setting = request.form.get("pdf_setting", "/ebook")
    scan_th     = form_int("scan_th", 20)
    digital_th  = form_int("digital_th", 200)
    min_img     = form_float("min_img", 1.0)
    max_size_gs = form_float("max_size_gs", 50.0)

    # Validasi mode
    if mode not in ("AUTO", "DIGITAL", "SCAN", "HYBRID"):
        return jsonify({"error": f"Mode tidak valid: {mode}"}), 400

    # ── Simpan file upload ──
    uid         = str(uuid.uuid4())
    input_path  = os.path.join(UPLOAD_FOLDER, uid + ".pdf")
    out_name    = "compressed_" + uid + ".pdf"
    output_path = os.path.join(OUTPUT_FOLDER, out_name)

    file.save(input_path)

    try:
        result = pdf_compress(
            in_path=input_path,
            out_path=output_path,
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

        result["ok"] = True
        result["download_id"] = out_name
        return jsonify(result)

    except Exception as e:
        # Bersihkan output yang mungkin setengah jadi
        if os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500

    finally:
        # Selalu hapus file upload setelah selesai
        if os.path.exists(input_path):
            os.remove(input_path)


# ---------------------------------------------------------------------------
# Download hasil
# ---------------------------------------------------------------------------

@app.route("/download/<filename>")
def download(filename):
    # Keamanan: tolak path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Nama file tidak valid."}), 400

    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File tidak ditemukan. Mungkin sudah kadaluarsa."}), 404

    return send_file(path, as_attachment=True, download_name="compressed.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
