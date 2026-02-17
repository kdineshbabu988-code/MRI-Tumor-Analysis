"""
predict.py — Inference module with global model caching.
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

import config

# ─── Global Model Cache ───────────────────────────────────────────────
_MODEL = None
_MODEL_TYPE = None

def load_model_instance(model_type="efficientnet"):
    """
    Load the Keras model globally. Ensures it is loaded only once.
    """
    global _MODEL, _MODEL_TYPE
    
    # If already loaded and type matches, return it
    if _MODEL is not None and _MODEL_TYPE == model_type:
        return _MODEL
        
    print(f"[INFO] Loading model: {model_type}...")
    try:
        model_path = config.get_model_path(model_type)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        _MODEL = keras_load_model(model_path)
        _MODEL_TYPE = model_type
        print(f"[SUCCESS] Model loaded from {model_path}")
        return _MODEL
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e

def predict_brain_tumor(image_path, model_type="efficientnet"):
    """
    Predict the class of a brain MRI image ensuring consistent results.
    """
    # 1. Ensure Model is Loaded
    model = load_model_instance(model_type)
    
    # 2. Preprocess Image (EXACT match to training)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    try:
        # Load image with target size 224x224
        img = load_img(image_path, target_size=(224, 224))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Rescale specific to training (1./255)
        img_array = img_array / 255.0
        
        # Expand dimensions to create batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")
    
    # 3. Predict
    predictions = model.predict(img_array, verbose=0)
    
    # 4. Format Output
    probabilities = predictions[0]
    predicted_idx = np.argmax(probabilities)
    confidence = float(np.max(probabilities))
    predicted_label = config.CLASS_NAMES[predicted_idx]
    
    # 5. Threshold Logic (User requested)
    status = "accept"
    message = "Prediction successful."
    
    if confidence < 0.35:
        # Low confidence fallback
        status = "reject"
        message = f"Low Confidence ({confidence:.2f}). Please upload a clearer MRI scan."
    
    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "status": status,
        "message": message,
        "all_probabilities": {
            config.CLASS_NAMES[i]: float(probabilities[i]) 
            for i in range(len(config.CLASS_NAMES))
        }
    }

if __name__ == "__main__":
    # Allow CLI usage
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to MRI image")
    parser.add_argument("--model", default="efficientnet", help="Model type")
    args = parser.parse_args()
    
    try:
        result = predict_brain_tumor(args.image, args.model)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
