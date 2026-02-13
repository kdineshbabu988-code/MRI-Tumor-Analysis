"""
predict.py — Single-image inference utility for brain tumor classification.

Usage:
    python predict.py path/to/mri_image.jpg
    python predict.py path/to/mri_image.jpg --model resnet50
"""

import argparse
import sys
import os

import config
from data_pipeline import preprocess_single_image
from utils import load_model, format_prediction


def predict(image_path: str, model_type: str = "custom") -> dict:
    """
    Predict the tumor class for a single MRI image.

    Args:
        image_path: Path to the MRI image file.
        model_type: 'custom' or 'resnet50'.

    Returns:
        Dict with 'predicted_class', 'confidence', 'all_probabilities'.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Validate file extension
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Allowed: {config.ALLOWED_EXTENSIONS}"
        )

    model = load_model(model_type)
    img_array = preprocess_single_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    result = format_prediction(predictions[0])

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict tumor class from an MRI image.")
    parser.add_argument("image", type=str, help="Path to the MRI image file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["custom", "resnet50"],
        default="custom",
        help="Model type to use (default: custom)",
    )
    args = parser.parse_args()

    result = predict(args.image, args.model)

    print("\n" + "=" * 50)
    print("  BRAIN TUMOR PREDICTION RESULT")
    print("=" * 50)
    print(f"  Image     : {args.image}")
    print(f"  Model     : {args.model}")
    print(f"  Predicted : {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}  ({result['confidence']*100:.2f}%)")
    print("\n  Class Probabilities:")
    for name, prob in result["all_probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {name:15s} : {prob:.4f}  {bar}")
    print("=" * 50 + "\n")

    # Clinical disclaimer
    print("  [!] DISCLAIMER: This prediction is for research/educational")
    print("     purposes only and NOT a medical diagnosis. Always consult")
    print("     a qualified medical professional.\n")


if __name__ == "__main__":
    main()
