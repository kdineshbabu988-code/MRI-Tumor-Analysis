"""
config.py — Centralised configuration for the Brain Tumor Classification Pipeline.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjusted to point to the correct parent directory if needed, or local
DATASET_DIR = os.path.join(BASE_DIR, "..", "Training") 
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ─── Class Definitions ──────────────────────────────────────────────────────
# Alphabetical Order Mandatory for Keras ImageDataGenerator
# 0: glioma, 1: meningioma, 2: notumor, 3: pituitary
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Settings ─────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# ─── Training Hyperparameters ───────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# ─── Fine-Tuning ─────────────────────────────────────────────────────────────
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LEARNING_RATE = 1e-5

# ─── Regularisation ─────────────────────────────────────────────────────────
DROPOUT_CONV = 0.3
DROPOUT_DENSE = 0.5
L2_WEIGHT_DECAY = 1e-4

# ─── Callbacks ───────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# ─── Data Augmentation Parameters ───────────────────────────────────────────
AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "shear_range": 10,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "nearest",
}

# ─── Flask Settings ──────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom"}
MAX_UPLOAD_SIZE_MB = 20

# ─── Validation Settings ──────────────────────────────────────────────────
# Stricter for Production
SATURATION_THRESHOLD = 0.1
MIN_PIXEL_VARIANCE = 500
MIN_EDGE_DENSITY = 0.02

# ─── Prediction Safety Thresholds ───────────────────────────────────────────
PREDICTION_THRESHOLD = 0.70   
REVIEW_THRESHOLD = 0.60       
MAX_ENTROPY = 2.0             
MIN_MARGIN = 0.05             

# ─── Model File Names ───────────────────────────────────────────────────────
CUSTOM_MODEL_NAME = "brain_tumor_custom_cnn.h5"
RESNET_MODEL_NAME = "brain_tumor_resnet50.h5"
# Corrected as requested
EFFICIENTNET_MODEL_NAME = "model.h5"

def get_model_path(model_type="custom"):
    """Return the full path to the saved model file."""
    if model_type == "custom":
        name = CUSTOM_MODEL_NAME
    elif model_type == "resnet50":
        name = RESNET_MODEL_NAME
    elif model_type == "efficientnet":
        name = EFFICIENTNET_MODEL_NAME
    else:
        # Fallback
        name = EFFICIENTNET_MODEL_NAME
        
    # Check if the file exists in the root of backend first (user common issue)
    local_path = os.path.join(BASE_DIR, name)
    if os.path.exists(local_path):
        return local_path
        
    return os.path.join(SAVED_MODELS_DIR, name)
