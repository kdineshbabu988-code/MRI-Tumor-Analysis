"""
utils.py — Utility functions for the Brain Tumor Classification Pipeline.

Provides model save/load helpers, training history plotting, and directory management.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

import config
from PIL import Image, ImageStat


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_model(model, model_type: str = "custom") -> str:
    """
    Save a Keras model to the saved_models directory.

    Args:
        model: Compiled/trained Keras model.
        model_type: 'custom' or 'resnet50'.

    Returns:
        Absolute path to the saved model file.
    """
    ensure_dir(config.SAVED_MODELS_DIR)
    path = config.get_model_path(model_type)
    model.save(path)
    print(f"[OK] Model saved to: {path}")
    return path


def load_model(model_type: str = "custom"):
    """
    Load a previously saved Keras model.

    Args:
        model_type: 'custom' or 'resnet50'.

    Returns:
        Loaded Keras model ready for inference.

    Raises:
        FileNotFoundError: If no saved model exists at the expected path.
    """
    path = config.get_model_path(model_type)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No saved model found at: {path}\n"
            f"Train a model first using: python train.py --model {model_type}"
        )
    model = keras_load_model(path)
    print(f"[OK] Model loaded from: {path}")
    return model


def plot_training_history(history, save_path: str = None) -> str:
    """
    Plot training & validation loss/accuracy curves and save as PNG.

    Args:
        history: Keras History object from model.fit().
        save_path: Optional custom path. Defaults to results/training_history.png.

    Returns:
        Path to the saved plot image.
    """
    if save_path is None:
        ensure_dir(config.RESULTS_DIR)
        save_path = os.path.join(config.RESULTS_DIR, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──
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
    """
    Format model output probabilities into a structured prediction dict.

    Args:
        probabilities: 1-D array of shape (NUM_CLASSES,).

    Returns:
        Dict with 'predicted_class', 'confidence', and 'all_probabilities'.
    """
    predicted_idx = int(np.argmax(probabilities))
    return {
        "predicted_class": config.CLASS_NAMES[predicted_idx],
        "confidence": float(probabilities[predicted_idx]),
        "all_probabilities": {
            name: float(prob)
            for name, prob in zip(config.CLASS_NAMES, probabilities)
        },
    }


def safe_format(probabilities: np.ndarray) -> tuple[bool, str, dict]:
    """
    Apply safety thresholds to model output to reject unrecognized/ambiguous images.

    Thresholds (from config.py):
    - PREDICTION_THRESHOLD = 0.92
    - MAX_ENTROPY = 0.8
    - MIN_MARGIN = 0.5

    Returns:
        tuple: (is_safe, error_message, result_dict)
    """
    # 1. Calculate Confidence
    confidence = float(np.max(probabilities))
    predicted_idx = int(np.argmax(probabilities))
    
    # 2. Calculate Entropy (Uncertainty)
    # entropy = -sum(p * log(p))
    valid_probs = probabilities[probabilities > 0]
    entropy = -float(np.sum(valid_probs * np.log2(valid_probs)))
    
    # 3. Calculate Margin (Certainty)
    sorted_probs = np.sort(probabilities)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

    result = format_prediction(probabilities)

    # Rejection Logic
    if confidence < config.PREDICTION_THRESHOLD:
        return False, f"Low confidence ({confidence:.2f}). Image might not be a valid MRI/CT scan.", result
    
    if entropy > config.MAX_ENTROPY:
        return False, f"High uncertainty (entropy: {entropy:.2f}). Image is too ambiguous.", result
    
    if margin < config.MIN_MARGIN:
        return False, f"Low class separation (margin: {margin:.2f}). Prediction is too close.", result

    return True, "", result


def validate_image_content(filepath: str) -> tuple[bool, str]:
    """
    Validate that the image is a valid MRI/CT scan (low saturation, not blank).

    Args:
        filepath: Path to the image file.

    Returns:
        tuple (bool, str): (is_valid, error_message)
    """
    try:
        # Open image
        with Image.open(filepath) as img:
            # 1. Convert to RGB for saturation check (even if original is grayscale)
            img_rgb = img.convert("RGB")
            # Convert to HSV to check saturation
            # PIL doesn't have a direct "get saturation average", so we'll use a trick
            # or convert to numpy for faster processing if needed.
            # Using numpy since it's already a dependency for the model.
            img_array = np.array(img_rgb)
            
            # 2. Check for blank images (zero variance)
            variance = np.var(img_array)
            if variance < config.MIN_PIXEL_VARIANCE:
                return False, "Image is too uniform or blank. Please upload a clear MRI/CT scan."

            # 3. Saturation check
            # Convert RGB to HSV
            # HSV: H=0-179, S=0-255, V=0-255 (if using cv2, but we use PIL->numpy)
            # For grayscale, S should be near 0.
            # Formula for S: (max(R,G,B) - min(R,G,B)) / max(R,G,B)
            max_v = np.max(img_array, axis=2)
            min_v = np.min(img_array, axis=2)
            
            # Avoid division by zero
            mask = max_v > 0
            saturation = np.zeros_like(max_v, dtype=np.float32)
            saturation[mask] = (max_v[mask] - min_v[mask]) / max_v[mask]
            
            avg_saturation = np.mean(saturation)
            
            if avg_saturation > config.SATURATION_THRESHOLD:
                return False, (
                    f"Uploaded image appears to be a color photo (saturation: {avg_saturation:.2f}). "
                    "Only MRI or CT scans (grayscale) are allowed for classification."
                )

        return True, ""
    except Exception as e:
        return False, f"Failed to process image: {str(e)}"
