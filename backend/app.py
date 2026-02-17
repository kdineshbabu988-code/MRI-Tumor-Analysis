"""
app.py — Flask web application for brain tumor classification.
"""

import os
import traceback
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

import config
from utils import ensure_dir, validate_image_content, safe_format
from predict import load_model_instance, predict_brain_tumor
from data_pipeline import preprocess_single_image

# ═════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

UPLOAD_DIR = os.path.join(config.BASE_DIR, "uploads")
ensure_dir(UPLOAD_DIR)

# ── Load model at startup ────────────────────────────────────────────────
# We preload the model using the singleton in predict.py
try:
    load_model_instance("efficientnet")
except Exception as e:
    print(f"\n[WARNING] Could not preload model: {e}")
    print("    Application will try to load it on first request.\n")


def _allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Render the upload page."""
    return render_template("index.html", class_names=config.CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an MRI image upload and return the prediction as JSON.
    """
    try:
        # ── Validate request ─────────────────────────────────────────────
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        if not _allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # ── Content Validation ───────────────────────────────────────────
        # Reject unwanted non-MRI/CT images
        is_valid_img, error_msg = validate_image_content(filepath)
        if not is_valid_img:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"success": False, "error": error_msg}), 400

        try:
            # ── Main Prediction ──────────────────────────────────────────────
            # ── Main Prediction ──────────────────────────────────────────────
            # Use the unified prediction function from predict.py
            # This handles loading, preprocessing, and formatting internally
            result = predict_brain_tumor(filepath, model_type="efficientnet")
            
            # Check status returned by predict_brain_tumor
            if result["status"] == "reject":
                return jsonify({
                    "success": False,
                    "error": result["message"],
                    "status": "reject",
                    "all_probabilities": result["all_probabilities"]
                }), 422

            # 'accept' or 'review'
            return jsonify({
                "success": True,
                "status": result["status"],
                "message": result["message"],
                "predicted_class": result["predicted_class"],
                "confidence": round(result["confidence"], 4),
                "all_probabilities": result["all_probabilities"],
                "model_type": "efficientnet",
                "tta_enabled": False
            })

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Processing error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        load_model_instance("efficientnet")
        return jsonify({"status": "healthy", "model_loaded": True})
    except:
        return jsonify({"status": "degraded", "model_loaded": False})


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFIER - WEB APPLICATION")
    print("=" * 60)
    print(f"  Server  : http://localhost:{config.FLASK_PORT}")
    print("=" * 60 + "\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
