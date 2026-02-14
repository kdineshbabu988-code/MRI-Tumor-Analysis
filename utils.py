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

def load_model(model_type: str = "custom"):
    """Load a previously saved Keras model."""
    path = config.get_model_path(model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    model = keras_load_model(path)
    print(f"[OK] Model loaded from: {path}")
    return model

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
    Apply CALIBRATED MEDICAL DECISION LOGIC (STEP 1 & 8).
    
    Logic:
    - If Confidence >= 0.70 -> Accept "Tumor" or "No Tumor".
    - If Confidence <= 0.30 -> Reject as "Uncertain" (for multi-class max probability).
      (Note: User requested 'prediction <= 0.30: No Tumor' but in multi-class,
       low max probability means complete ambiguity among 4 classes).
    - Remove strict blocks (< 0.80).
    - Use calibrated probability.
    
    Returns:
        tuple: (is_valid, message, result_dict)
    """
    if probabilities.ndim > 1:
        probabilities = probabilities[0]

    # STEP 3: Apply Confidence Calibration
    # User requested: "confidence = float(prediction)"
    # For multi-class, this is the max probability.
    confidence = float(np.max(probabilities))
    
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = config.CLASS_NAMES[predicted_idx]
    
    # Calculate Entropy & Margin
    valid_probs = probabilities[probabilities > 0]
    entropy = -float(np.sum(valid_probs * np.log2(valid_probs)))
    
    sorted_probs = np.sort(probabilities)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    result = format_prediction(probabilities)
    result["entropy"] = round(entropy, 4)
    result["margin"] = round(margin, 4)
    
    # Check "STEP 4" Entropy Constraint
    if entropy > config.MAX_ENTROPY: # 0.99
         result["status"] = "reject"
         return False, f"Scan too ambiguous (Entropy: {entropy:.2f}). Please re-scan.", result

    # STEP 1: FIX DECISION THRESHOLD LOGIC
    # User requested:
    # if prediction >= 0.70: result = "Tumor"
    # elif prediction <= 0.30: result = "No Tumor"
    # else: result = "Uncertain"
    
    # Adaptation for Multi-Class:
    # If confidence >= 0.70, we accept the class (Tumor type or NoTumor).
    if confidence >= config.PREDICTION_THRESHOLD: # 0.70
        result["status"] = "accept"
        if predicted_label == "notumor":
             return True, f"No Tumor Detected ({confidence:.2f})", result
        else:
             return True, f"Detected: {predicted_label} ({confidence:.2f})", result
             
    # If confidence is lower than 0.70 but decent (e.g. > 0.50), let's mark as Review instead of Reject?
    # User said: "remove over-strict confidence block... unless confidence truly ambiguous"
    # User Logic: else: result = "Uncertain" if not >= 0.70 (and not <= 0.30).
    
    # If confidence is extremely low (< 0.30 is barely above 0.25 random):
    if confidence <= 0.30:
        # In a 4-class model, max_prob <= 0.30 is chaos. It's essentially noise.
        result["status"] = "reject"
        return False, "Uncertain - Image inconclusive.", result

    # Middle Ground (0.30 < Conf < 0.70)
    # Be honest: "Uncertain" per user request Step 1 "else: Uncertain"
    # But user also said "REMOVE OVER-STRICT... uncess confidence truly ambiguous". 
    # 0.65 is arguably okay.
    # However, strictly following Step 1:
    result["status"] = "reject" 
    # Using 'reject' causes the frontend to show error/red.
    # Maybe use 'review' if between 0.50 and 0.70?
    if confidence >= config.REVIEW_THRESHOLD: # 0.50
        result["status"] = "review"
        return True, f"Result Uncertain (Conf: {confidence:.2f}). Clinical review advised.", result
        
    return False, f"Uncertain – Please use higher quality MRI image (Conf: {confidence:.2f})", result


import cv2
import pydicom

def validate_image_content(filepath: str) -> tuple[bool, str]:
    """Validate that the image is a medical scan (MRI/CT)."""
    try:
        if filepath.lower().endswith((".dcm", ".dicom")):
            ds = pydicom.dcmread(filepath)
            if not hasattr(ds, "pixel_array"):
                return False, "Invalid DICOM file."
            return True, ""
            
        with Image.open(filepath) as img:
            img_rgb = img.convert("RGB")
            img_gray = img.convert("L")
            img_array = np.array(img_rgb)
            img_gray_array = np.array(img_gray)
            
            variance = np.var(img_gray_array)
            if variance < config.MIN_PIXEL_VARIANCE:
                return False, "Image is too uniform/blank."

            max_v = np.max(img_array, axis=2)
            min_v = np.min(img_array, axis=2)
            mask = max_v > 0
            saturation = np.zeros_like(max_v, dtype=np.float32)
            saturation[mask] = (max_v[mask] - min_v[mask]) / (max_v[mask] + 1e-7)
            if np.mean(saturation) > config.SATURATION_THRESHOLD:
                return False, "Invalid Image. MRI scans must be grayscale."

            edges = cv2.Canny(img_gray_array, 50, 150)
            if np.mean(edges > 0) < config.MIN_EDGE_DENSITY:
                return False, "Image is too blurry. medical scans require detail."
                
        return True, ""

    except Exception as e:
        return False, f"Validation Error: {str(e)}"
