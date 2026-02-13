"""
train.py — Training workflow for the Brain Tumor Classification Pipeline.

Usage:
    python train.py                          # Train custom CNN for 30 epochs
    python train.py --model resnet50         # Train ResNet50 transfer learning model
    python train.py --model custom --epochs 10   # Custom CNN, 10 epochs
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

import config
from data_pipeline import create_data_generators
from model import build_model
from utils import ensure_dir, save_model, plot_training_history


def get_callbacks(model_type: str) -> list:
    """
    Build the list of training callbacks.

    Callbacks:
        - EarlyStopping: Stops training when val_loss stops improving.
        - ReduceLROnPlateau: Halves the LR when val_loss plateaus.
        - ModelCheckpoint: Saves the best model weights during training.
        - TensorBoard: Logs metrics for visualisation.
    """
    ensure_dir(config.SAVED_MODELS_DIR)
    ensure_dir(config.LOG_DIR)

    checkpoint_path = config.get_model_path(model_type)

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
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
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, model_type),
            histogram_freq=1,
        ),
    ]


def train(model_type: str = "custom", epochs: int = None):
    """
    Complete training workflow.

    Steps:
        1. Create data generators (augmented train, clean val).
        2. Build and compile the chosen model architecture.
        3. Train with callbacks (early stopping, LR scheduling, checkpointing).
        4. Save the best model and training history plot.
        5. Print final metrics summary.

    Args:
        model_type: 'custom' or 'resnet50'.
        epochs: Override for number of epochs (defaults to config.EPOCHS).
    """
    epochs = epochs or config.EPOCHS

    # ── Print header ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFICATION - TRAINING")
    print("=" * 60)
    print(f"  Model      : {model_type}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {config.BATCH_SIZE}")
    print(f"  Image size : {config.IMG_SIZE}")
    print(f"  Learning rate : {config.LEARNING_RATE}")
    print("=" * 60 + "\n")

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("[1/4] Loading and augmenting data...")
    train_gen, val_gen = create_data_generators()

    # ── Step 2: Model ────────────────────────────────────────────────────
    print("[2/4] Building model architecture...")
    model = build_model(model_type)

    # ── Step 3: Train ────────────────────────────────────────────────────
    print("[3/4] Starting training...\n")
    callbacks = get_callbacks(model_type)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 4: Save & Report ────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    model_path = save_model(model, model_type)
    plot_path = plot_training_history(history)

    # Final summary
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])
    total_epochs = len(history.history["loss"])

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total epochs trained : {total_epochs}")
    print(f"  Best val accuracy    : {best_val_acc:.4f}")
    print(f"  Best val loss        : {best_val_loss:.4f}")
    print("  Model saved at       : {model_path}".format(model_path=model_path))
    print(f"  History plot at      : {plot_path}")
    print("=" * 60 + "\n")

    return model, history


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a brain tumor classification model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom", "resnet50"],
        default="custom",
        help="Model architecture to train (default: custom)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_type=args.model, epochs=args.epochs)
