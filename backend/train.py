"""
train.py — Training workflow for the Brain Tumor Classification Pipeline.

Usage:
    python train.py                          # Train default model for 50 epochs
    python train.py --model efficientnet --epochs 50
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
# Import legacy optimizer for consistent Apple/M1 support if needed, though standard Adam usually works fine.

import config
from data_pipeline import create_data_generators
from model import build_model
from utils import ensure_dir, save_model, plot_training_history


def get_callbacks(model_type: str) -> list:
    """
    Build the list of training callbacks for medical accuracy.
    Includes EarlyStopping and ReduceLROnPlateau as requested.
    """
    ensure_dir(config.SAVED_MODELS_DIR)
    ensure_dir(config.LOG_DIR)

    checkpoint_path = config.get_model_path(model_type)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE, # Configured patience
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.REDUCE_LR_MIN,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, model_type),
            histogram_freq=1,
        ),
    ]


def train(model_type: str = "efficientnet", epochs: int = 50, fine_tune: bool = False):
    """
    Complete training workflow for Brain MRI Classification.
    Ensures high accuracy through sufficient epochs (50) and callbacks.
    """
    # Enforce defaults requested by user if not overridden
    final_epochs = epochs if epochs is not None else 50

    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFICATION - TRAINING (HIGH ACCURACY MODE)")
    print("=" * 60)
    print(f"  Model      : {model_type}")
    print(f"  Target Epochs: {final_epochs}")
    print(f"  Batch size : {config.BATCH_SIZE}")
    print("=" * 60 + "\n")

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("[1/4] Loading and augmenting MRI data...")
    train_gen, val_gen = create_data_generators()

    # ── Step 2: Model ────────────────────────────────────────────────────
    print("[2/4] Building CNN architecture...")
    if model_type in ["resnet50", "efficientnet"]:
        model = build_model(model_type, fine_tune=fine_tune)
    else:
        model = build_model(model_type)

    # Compile with specific Adam Optimizer settings as requested
    print(f"[INFO] Compiling model with Adam(learning_rate=0.0001)...")
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Step 3: Train ────────────────────────────────────────────────────
    print(f"[3/4] Starting training for {final_epochs} epochs...\n")
    callbacks = get_callbacks(model_type)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=final_epochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # ── Step 4: Save & Report ────────────────────────────────────────────
    print("\n[4/4] Finalizing model...")
    model_path = save_model(model, model_type)
    plot_path = plot_training_history(history)

    # Final Summary for user requirements
    best_val_acc = max(history.history["val_accuracy"])
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Final Accuracy: {best_val_acc:.4f}")
    print(f"  Model Saved: {model_path}")
    print("=" * 60 + "\n")

    return model, history


def parse_args():
    """Parse command-line arguments properly."""
    parser = argparse.ArgumentParser(description="Expert MRI Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom", "resnet50", "efficientnet"],
        default="efficientnet",  # Changed default to better model
        help="Architecture (default: efficientnet)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,  # Changed default to 50
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze layers for transfer learning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_type=args.model, epochs=args.epochs, fine_tune=args.fine_tune)
