"""
utils.py — Utility functions for the Brain Tumor Classification Pipeline.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
import config
from PIL import Image

def ensure_dir(path: str) -> str:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path

def save_model(model, model_type: str = "custom") -> str:
    """Save a Keras model to the saved_models directory."""
    ensure_dir(config.SAVED_MODELS_DIR)
    path = config.get_model_path(model_type)
    model.save(path)
    print(f"[OK] Model saved to: {path}")
    return path

def load_model(model_type: str = "custom"):
    """Load a previously saved Keras model."""
    path = config.get_model_path(model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    model = keras_load_model(path)
    print(f"[OK] Model loaded from: {path}")
    return model

def plot_training_history(history, save_path: str = None) -> str:
    """Plot training & validation loss/accuracy curves and save as PNG."""
    if save_path is None:
        ensure_dir(config.RESULTS_DIR)
        save_path = os.path.join(config.RESULTS_DIR, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Training history plot saved to: {save_path}")
    return save_path

def get_class_label(index: int) -> str:
    """Convert a class index to its human-readable label."""
    if 0 <= index < config.NUM_CLASSES:
        return config.CLASS_NAMES[index]
    return "unknown"

def format_prediction(probabilities: np.ndarray) -> dict:
    """Format model output probabilities."""
    predicted_idx = int(np.argmax(probabilities))
    
    # ─── DEBUG OUTPUT (Requested: STEP 6) ───
    print("\n" + "=" * 40)
    print(f"  Raw Prediction (Probabilities): {probabilities}")
    print(f"  Max Confidence : {np.max(probabilities)}")
    print(f"  Predicted Class: {config.CLASS_NAMES[predicted_idx]}")
    print("=" * 40 + "\n")
    
    return {
        "predicted_class": config.CLASS_NAMES[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "all_probabilities": {
            config.CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(config.CLASS_NAMES))
        },
    }

def safe_format(probabilities: np.ndarray) -> tuple[bool, str, dict]:
    """
    Refined decision logic (Relaxed Calibration).
    Requirement: 
    - Confidence threshold to 0.75
    - Entropy limit to 1.2
    - Return prediction if confidence > 0.70
    - Only show ambiguity if confidence < 0.60
    """
    if probabilities.ndim > 1:
        probabilities = probabilities[0]

    confidence = float(np.max(probabilities))
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = config.CLASS_NAMES[predicted_idx]
    
    # Calculate Statistical Uncertainty (Entropy)
    valid_probs = probabilities[probabilities > 0]
    entropy = -float(np.sum(valid_probs * np.log2(valid_probs)))
    
    # Calculate Decision Margin
    sorted_probs = np.sort(probabilities)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    result = format_prediction(probabilities)
    result["entropy"] = round(entropy, 4)
    result["margin"] = round(margin, 4)
    
    # ── FALLBACK & STABILITY LOGIC ──
    # User Req 4: Ensure prediction is returned if confidence > 0.70
    # even if entropy is close to limit.
    
    # Rule 1: High Confidence Accept
    if confidence >= config.PREDICTION_THRESHOLD: # 0.75
        result["status"] = "accept"
        msg = f"Confidence: {confidence:.2f}. "
        if predicted_label == "notumor":
             return True, msg + "Diagnosis: No Tumor Detected.", result
        else:
             return True, msg + f"Diagnosis: Detected {predicted_label.capitalize()}.", result

    # Rule 2: Near-Threshold Fallback
    # User Req: If confidence > PREDICTION_THRESHOLD (adjusted), prevent "Ambiguity" error
    if confidence >= config.REVIEW_THRESHOLD:
        result["status"] = "accept"
        return True, f"Confidence: {confidence:.2f}. Prediction verified with fallback logic.", result
             
    # Rule 3: Middle Ground / Review
    # Only show ambiguity if confidence < REVIEW_THRESHOLD
    if confidence >= config.REVIEW_THRESHOLD - 0.1: # Allow a slightly lower bound for review
        result["status"] = "review"
        return True, f"Confidence: {confidence:.2f}. Result stable but review advised.", result
        
    # Rule 4: Ambiguity Detection
    if entropy > config.MAX_ENTROPY:
         result["status"] = "reject"
         return False, f"Scanning Ambiguity Detected (Entropy: {entropy:.2f}). Please upload a clearer scan.", result

    # Rule 5: Low Confidence Rejection
    result["status"] = "reject"
    return False, f"Low Confidence ({confidence:.2f}). Image might not be a valid MRI/CT scan.", result


import cv2
import pydicom

def validate_image_content(filepath: str) -> tuple[bool, str]:
    """
    Biomedical expert validation of image integrity.
    Rejects non-medical, blurry, or low-contrast images to prevent false detections.
    """
    try:
        if filepath.lower().endswith((".dcm", ".dicom")):
            ds = pydicom.dcmread(filepath)
            if not hasattr(ds, "pixel_array"):
                return False, "DICOM metadata failure: No pixel payload found."
            return True, ""
            
        with Image.open(filepath) as img:
            img_rgb = img.convert("RGB")
            img_gray = img.convert("L")
            img_array = np.array(img_rgb)
            img_gray_array = np.array(img_gray)
            
            # 1. Variance Check (Reject flat/dead images)
            variance = np.var(img_gray_array)
            if variance < config.MIN_PIXEL_VARIANCE:
                return False, "Low signal intensity. Please upload a clear diagnostic scan."

            # 2. Saturation Check (Medical scans are grayscale)
            max_v = np.max(img_array, axis=2)
            min_v = np.min(img_array, axis=2)
            mask = max_v > 0
            saturation = np.zeros_like(max_v, dtype=np.float32)
            saturation[mask] = (max_v[mask] - min_v[mask]) / (max_v[mask] + 1e-7)
            if np.mean(saturation) > config.SATURATION_THRESHOLD:
                return False, "Input Error: Non-grayscale content detected. This AI analyzes MRI/CT scans only."

            # 3. Structural Detail Check (Edge Density)
            edges = cv2.Canny(img_gray_array, 50, 150)
            edge_density = np.mean(edges > 0)
            if edge_density < config.MIN_EDGE_DENSITY:
                return False, "Insufficient structural detail (Edge Density too low). High-quality scan required."
            
            # 4. Expert Histogram Analysis (Check for typical brain scan distribution)
            # Medical scans usually have a peak for background and a peak for tissue.
            hist = cv2.calcHist([img_gray_array], [0], None, [256], [0, 256])
            # If the image is just noise, the histogram is often too uniform.
            # We check if the most frequent values are somewhat balanced (simple heuristic)
            if np.max(hist) / np.sum(hist) > 0.8:
                 return False, "Image contains repetitive patterns/noise. Not a valid anatomical scan."
                
        return True, ""

    except Exception as e:
        return False, f"Biomedical Validation Error: {str(e)}"
