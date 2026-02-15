"""
data_pipeline.py — Data preprocessing and augmentation for brain tumor MRI classification.

Provides training (augmented) and validation (clean) data generators using
Keras ImageDataGenerator. All parameters are driven from config.py.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import config
import cv2  # Still needed for some fallback or resizing if needed, but primary path uses load_img

def create_data_generators():
    """
    Create training and validation data generators from the dataset directory.
    Training data is augmented; validation data is only rescaled.
    """
    # ── Training generator with augmentation ─────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=config.AUGMENTATION["rotation_range"],
        width_shift_range=config.AUGMENTATION["width_shift_range"],
        height_shift_range=config.AUGMENTATION["height_shift_range"],
        shear_range=config.AUGMENTATION["shear_range"],
        zoom_range=config.AUGMENTATION["zoom_range"],
        horizontal_flip=config.AUGMENTATION["horizontal_flip"],
        brightness_range=config.AUGMENTATION["brightness_range"],
        fill_mode=config.AUGMENTATION["fill_mode"],
        validation_split=config.VALIDATION_SPLIT,
    )

    # ── Validation generator — rescaling only ────────────────────────────
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config.VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=config.RANDOM_SEED,
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        config.DATASET_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=config.RANDOM_SEED,
        shuffle=False,  # Keep order for evaluation
    )

    _print_generator_info(train_generator, val_generator)
    return train_generator, val_generator


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single MRI image for inference.
    EXACTLY MATCHES TRAINING PREPROCESSING.
    """
    # Use load_img to ensure consistency with training
    # Only use basic Keras preprocessing as requested to fix mismatch
    img = load_img(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
    img_array = img_to_array(img)
    
    # Normalize to [0,1]
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def _print_generator_info(train_gen, val_gen):
    """Print summary information about the data generators."""
    print("\n" + "=" * 60)
    print("  DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Image size       : {config.IMG_SIZE[0]}×{config.IMG_SIZE[1]}")
    print(f"  Batch size       : {config.BATCH_SIZE}")
    print(f"  Training samples : {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes          : {list(train_gen.class_indices.keys())}")
    print(f"  Augmentation     : Enabled (training only)")
    print("=" * 60 + "\n")
