"""
app.py — Flask web application for brain tumor classification.

Endpoints:
    GET  /         — Upload page with drag-and-drop MRI image upload.
    POST /predict  — JSON API that returns {predicted_class, confidence, all_probabilities}.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import traceback
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

import config
from data_pipeline import preprocess_single_image
from utils import load_model, format_prediction, ensure_dir, validate_image_content


# ═════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

UPLOAD_DIR = os.path.join(config.BASE_DIR, "uploads")
ensure_dir(UPLOAD_DIR)

# ── Load model at startup ────────────────────────────────────────────────
_model = None
_model_type = None


def _get_model():
    """Lazy-load the model (tries custom first, then resnet50)."""
    global _model, _model_type
    if _model is None:
        # Priority: EfficientNet > ResNet50 > Custom
        for mt in ["efficientnet", "resnet50", "custom"]:
            try:
                _model = load_model(mt)
                _model_type = mt
                print(f"[OK] Loaded {mt} model for inference.")
                break
            except FileNotFoundError:
                continue
        if _model is None:
            raise RuntimeError(
                "No trained model found! Train a model first:\n"
                "  python train.py --model efficientnet  (Recommended)\n"
                "  python train.py --model custom"
            )
    return _model


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

    Returns:
        JSON: {
            "success": true,
            "predicted_class": "glioma",
            "confidence": 0.9712,
            "all_probabilities": {"glioma": 0.97, "meningioma": 0.01, ...},
            "model_type": "custom"
        }
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
        # Reject unwanted non-MRI/CT images (e.g. random color photos)
        is_valid, error_msg = validate_image_content(filepath)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"success": False, "error": error_msg}), 400

        try:
            model = _get_model()
            
            # ── Test Time Augmentation (TTA) ──────────────────────────────
            # We improve accuracy by predicting on:
            # 1. The original image
            # 2. The horizontally flipped image
            # Then averaging the probabilities.
            
            # Preprocess original
            img_array = preprocess_single_image(filepath)
            
            # Create a batch of 2 images: [original, flipped]
            # Note: preprocess_single_image returns (1, 224, 224, 3)
            # We access [0] to get (224, 224, 3)
            img_orig = img_array[0]
            img_flip = np.fliplr(img_orig)
            
            batch = np.array([img_orig, img_flip])
            
            # Get predictions for both
            preds = model.predict(batch, verbose=0)
            
            # Average the predictions
            avg_pred = np.mean(preds, axis=0)
            
            # Use safe_format for 3-tier validation (Reject/Review/Accept)
            from utils import safe_format
            is_valid, message, result = safe_format(avg_pred)

            if not is_valid:  # Status is 'reject'
                return jsonify({
                    "success": False,
                    "error": message,
                    "status": result.get("status", "reject"),
                    "all_probabilities": {
                        k: round(v, 4) for k, v in result["all_probabilities"].items()
                    }
                }), 422

            # 'accept' or 'review'
            return jsonify({
                "success": True,
                "status": result["status"],
                "message": message,
                "predicted_class": result["predicted_class"],
                "confidence": round(result["confidence"], 4),
                "all_probabilities": {
                    k: round(v, 4) for k, v in result["all_probabilities"].items()
                },
                "model_type": _model_type,
                "tta_enabled": True
            })
        finally:
            # Clean up uploaded file after prediction
            if os.path.exists(filepath):
                os.remove(filepath)

    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({"status": "healthy", "model_loaded": _model is not None})


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFIER - WEB APPLICATION")
    print("=" * 60)
    print(f"  Server  : http://localhost:{config.FLASK_PORT}")
    print(f"  Debug   : {config.FLASK_DEBUG}")
    print("=" * 60 + "\n")
    print("  [!] DISCLAIMER: For research/educational use only.")
    print("     Not intended for clinical diagnosis.\n")

    # Pre-load model so first request is fast
    try:
        _get_model()
    except RuntimeError as e:
        print(f"\n[!] WARNING: {e}")
        print("    The app will start, but /predict will fail until a model is trained.\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
