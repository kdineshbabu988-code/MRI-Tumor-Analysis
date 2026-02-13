"""
evaluate.py — Performance evaluation for trained brain tumor classifiers.

Usage:
    python evaluate.py                    # Evaluate the custom CNN
    python evaluate.py --model resnet50   # Evaluate the ResNet50 model

Outputs:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix heatmap (saved as PNG)
    - Overall accuracy, macro-averaged metrics
    - ROC-AUC scores per class (One-vs-Rest)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

import config
from data_pipeline import create_data_generators
from utils import load_model, ensure_dir


def evaluate(model_type: str = "custom"):
    """
    Evaluate a saved model on the validation split.

    Steps:
        1. Load model and create validation data generator.
        2. Generate predictions on the full validation set.
        3. Compute and print classification metrics.
        4. Generate and save confusion matrix heatmap.
        5. Compute per-class ROC-AUC scores.

    Args:
        model_type: 'custom' or 'resnet50'.
    """
    print("\n" + "=" * 60)
    print("  BRAIN TUMOR CLASSIFICATION - EVALUATION")
    print("=" * 60)
    print(f"  Model: {model_type}")
    print("=" * 60 + "\n")

    # ── Step 1: Load model & data ────────────────────────────────────────
    print("[1/4] Loading model and data...")
    model = load_model(model_type)
    _, val_gen = create_data_generators()

    # ── Step 2: Predict ──────────────────────────────────────────────────
    print("[2/4] Generating predictions...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    # ── Step 3: Classification report ────────────────────────────────────
    print("\n[3/4] Computing metrics...\n")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4,
    )

    print("─" * 50)
    print(f"  Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print("─" * 50)
    print("\n  Classification Report:\n")
    print(report)

    # ── ROC-AUC (One-vs-Rest) ────────────────────────────────────────────
    try:
        # One-hot encode true labels for ROC computation
        y_true_onehot = np.zeros((len(y_true), config.NUM_CLASSES))
        for i, label in enumerate(y_true):
            y_true_onehot[i, label] = 1

        auc_scores = {}
        for i, name in enumerate(config.CLASS_NAMES):
            auc = roc_auc_score(y_true_onehot[:, i], predictions[:, i])
            auc_scores[name] = auc

        macro_auc = np.mean(list(auc_scores.values()))

        print("  ROC-AUC Scores (One-vs-Rest):")
        print("  " + "─" * 35)
        for name, auc in auc_scores.items():
            print(f"    {name:15s} : {auc:.4f}")
        print(f"    {'Macro Average':15s} : {macro_auc:.4f}")
        print("  " + "─" * 35)
    except Exception as e:
        print(f"  [!] ROC-AUC computation skipped: {e}")

    # ── Step 4: Confusion matrix ─────────────────────────────────────────
    print("\n[4/4] Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, model_type)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60 + "\n")

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def _plot_confusion_matrix(cm: np.ndarray, model_type: str):
    """Generate and save a confusion matrix heatmap."""
    ensure_dir(config.RESULTS_DIR)
    save_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{model_type}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_type.upper()}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Confusion matrix saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom", "resnet50"],
        default="custom",
        help="Model type to evaluate (default: custom)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(model_type=args.model)
