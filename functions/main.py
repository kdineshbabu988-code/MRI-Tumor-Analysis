
import os
import tempfile
import numpy as np
from firebase_functions import https_fn
from firebase_admin import initialize_app
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import your custom modules (ensure these files are in functions/ folder)
import config
from data_pipeline import preprocess_single_image
from utils import load_model, ensure_dir, validate_image_content

# Initialize Firebase Admin SDK
initialize_app()

app = Flask(__name__)

# ── Load model globally to cache it between requests ──────────────────────
_model = None
_model_type = None

def get_cached_model():
    """Lazy load the model. Retries allow for cold start handling."""
    global _model, _model_type
    if _model is None:
        # Load the model file located in the same directory or SAVED_MODELS_DIR
        # Since we updated config.py to point SAVED_MODELS_DIR to BASE_DIR (this dir)
        # we can use config.get_model_path("custom")
        model_path = config.get_model_path("custom")
        
        if not os.path.exists(model_path):
            # Fallback for direct path check
            model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_custom_cnn.h5")
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Determine strict structure for Keras loading
        import tensorflow as tf
        print(f"[INFO] Loading model from {model_path}...")
        _model = tf.keras.models.load_model(model_path, compile=False)
        _model_type = "custom"
        print(f"[OK] Model loaded.")
    return _model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── Validate Request ─────────────────────────────────────────────
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        # Save to /tmp (Required for Cloud Functions as file system is read-only)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            temp_path = temp.name

        try:
            # ── Content Validation ───────────────────────────────────────
            # Reject unwanted non-MRI/CT images
            is_valid, error_msg = validate_image_content(temp_path)
            if not is_valid:
                return jsonify({"success": False, "error": error_msg}), 422

            # ── Prediction ──────────────────────────────────────────────
            model = get_cached_model()
            
            # TTA (Test Time Augmentation) Logic
            # We improve accuracy by predicting on:
            # 1. The original image
            # 2. The horizontally flipped image
            # Then averaging the probabilities.
            
            img_array = preprocess_single_image(temp_path)
            img_orig = img_array[0]
            img_flip = np.fliplr(img_orig)
            batch = np.array([img_orig, img_flip])
            
            preds = model.predict(batch, verbose=0)
            avg_pred = np.mean(preds, axis=0)
            
            # ── Format Result ───────────────────────────────────────────
            # Import safe_format here to avoid circular imports if any
            from utils import safe_format
            valid_res, message, result = safe_format(avg_pred)
            
            # Convert numpy floats to native python floats for JSON serialization
            all_probs = {k: float(v) for k, v in result.get("all_probabilities", {}).items()}
            conf = float(result.get("confidence", 0.0))
            
            if not valid_res:  # Status is 'reject'
                return jsonify({
                    "success": False,
                    "error": message,
                    "status": result.get("status", "reject"),
                    "all_probabilities": all_probs
                }), 422

            # 'accept' or 'review'
            return jsonify({
                "success": True,
                "status": result["status"],
                "message": message,
                "predicted_class": result["predicted_class"],
                "confidence": conf,
                "all_probabilities": all_probs,
                "model_type": _model_type,
                "tta_enabled": True
            })

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Expose Flask app as a Cloud Function
# Increased memory to 2GB to handle TensorFlow model loading
@https_fn.on_request(memory=2048, timeout_sec=60, region="us-central1")
def api(req: https_fn.Request) -> https_fn.Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()
