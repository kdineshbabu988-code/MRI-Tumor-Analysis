# 🧠 Brain Tumor Classification Pipeline

A production-ready deep learning pipeline for **multi-class brain tumor classification** from MRI scans, built with TensorFlow/Keras and Flask.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [CNN Architectures](#cnn-architectures)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Flask Web Application](#flask-web-application)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Regularisation Techniques](#regularisation-techniques)
- [Deployment Recommendations](#deployment-recommendations)
- [Transfer Learning Improvements](#transfer-learning-improvements)
- [Ethical Considerations](#ethical-considerations)

---

## Overview

This pipeline classifies brain MRI scans into **4 categories**:

| Class | Description |
|-------|-------------|
| **Glioma** | Tumor arising from glial cells in the brain |
| **Meningioma** | Tumor originating from meningeal tissue |
| **No Tumor** | Normal brain MRI scan |
| **Pituitary** | Tumor at the pituitary gland |

---

## Dataset

The dataset contains labeled MRI images organised by tumor type:

```
dataset/Training/
├── glioma/       (1,321 images)
├── meningioma/   (1,322 images)
├── notumor/      (1,579 images)
└── pituitary/    (1,457 images)
Total: 5,679 images
```

An 80/20 train-validation split is applied automatically.

---

## Project Structure

```
MRI_project/
├── config.py           # Centralised hyperparameters & paths
├── data_pipeline.py    # Preprocessing + augmentation
├── model.py            # CNN architectures (Custom + ResNet50)
├── train.py            # Training workflow with callbacks
├── evaluate.py         # Metrics, confusion matrix, ROC-AUC
├── predict.py          # Single-image inference CLI
├── app.py              # Flask web application
├── utils.py            # Save/load, plotting helpers
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Upload UI
├── static/
│   └── style.css       # Styling
├── saved_models/       # Trained models (created at runtime)
├── results/            # Plots & reports (created at runtime)
├── logs/               # TensorBoard logs (created at runtime)
└── dataset/
    └── Training/
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a model
python train.py --model custom --epochs 30

# 3. Evaluate
python evaluate.py --model custom

# 4. Run the web app
python app.py
# Open http://localhost:5000
```

---

## Data Preprocessing & Augmentation

**Preprocessing** (all data):
- Resize to **224×224** pixels
- Rescale pixel values to [0, 1]

**Augmentation** (training data only):
| Technique | Value | Purpose |
|-----------|-------|---------|
| Rotation | ±20° | Orientation invariance |
| Width/Height shift | ±15% | Translation invariance |
| Horizontal flip | Yes | Mirror invariance |
| Zoom | 0.85–1.15× | Scale invariance |
| Shear | 10° | Affine invariance |
| Brightness | 0.85–1.15× | Lighting variation |

Validation data receives **no augmentation** — only rescaling — to ensure unbiased evaluation.

---

## CNN Architectures

### 1. Custom CNN (`--model custom`)

A 5-block convolutional architecture (~2M parameters):

```
Input (224×224×3)
→ Block 1: [Conv3×3 → BN → ReLU] ×2 → MaxPool → Dropout(0.3)   [32 filters]
→ Block 2: [Conv3×3 → BN → ReLU] ×2 → MaxPool → Dropout(0.3)   [64 filters]
→ Block 3: [Conv3×3 → BN → ReLU] ×2 → MaxPool → Dropout(0.3)   [128 filters]
→ Block 4: [Conv3×3 → BN → ReLU] ×2 → MaxPool → Dropout(0.3)   [256 filters]
→ Block 5: [Conv3×3 → BN → ReLU] ×2 → MaxPool → Dropout(0.3)   [512 filters]
→ GlobalAveragePooling2D
→ Dense(256, relu) → Dropout(0.5)
→ Dense(4, softmax)
```

### 2. ResNet50 Transfer Learning (`--model resnet50`)

- **Base**: ResNet50 pre-trained on ImageNet (frozen by default)
- **Head**: GAP → BatchNorm → Dense(256) → Dropout(0.5) → Dense(4, softmax)
- **Fine-tuning**: Unfreeze top layers in a second training phase with lower LR

---

## Training

```bash
# Custom CNN (default)
python train.py

# ResNet50 transfer learning
python train.py --model resnet50

# Custom epochs
python train.py --model custom --epochs 15
```

**Training features:**
- **EarlyStopping**: Stops when val_loss plateaus for 7 epochs, restores best weights
- **ReduceLROnPlateau**: Halves LR when val_loss stalls for 3 epochs
- **ModelCheckpoint**: Saves best model by val_accuracy automatically
- **TensorBoard**: Full metric logging for visualisation

**View TensorBoard logs:**
```bash
tensorboard --logdir logs/
```

---

## Evaluation

```bash
python evaluate.py --model custom
```

**Outputs:**
- Overall accuracy
- Per-class precision, recall, F1-score
- Macro-averaged classification report
- Confusion matrix heatmap (saved to `results/`)
- ROC-AUC scores per class (One-vs-Rest)

---

## Inference

```bash
python predict.py path/to/mri_scan.jpg
python predict.py path/to/mri_scan.jpg --model resnet50
```

Returns the predicted class with confidence scores for all 4 categories.

---

## Flask Web Application

```bash
python app.py
# Open http://localhost:5000
```

**Endpoints:**
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Upload page with drag-and-drop UI |
| POST | `/predict` | JSON API — returns prediction + probabilities |
| GET | `/health` | Health check for monitoring |

**API Response Example:**
```json
{
    "success": true,
    "predicted_class": "glioma",
    "confidence": 0.9712,
    "all_probabilities": {
        "glioma": 0.9712,
        "meningioma": 0.0154,
        "notumor": 0.0089,
        "pituitary": 0.0045
    },
    "model_type": "custom"
}
```

---

## Hyperparameter Tuning

All hyperparameters are centralised in `config.py`:

| Parameter | Default | Tuning Guidance |
|-----------|---------|-----------------|
| Learning Rate | 1e-4 | Try 1e-3 to 1e-5; ReduceLROnPlateau handles dynamic adjustment |
| Batch Size | 32 | 16 for more noise/regularisation, 64 for faster convergence |
| Dropout (conv) | 0.3 | Increase if overfitting, decrease if underfitting |
| Dropout (dense) | 0.5 | Standard; try 0.3–0.6 |
| L2 Decay | 1e-4 | Increase for stronger regularisation |
| EarlyStopping patience | 7 | Lower for faster termination |
| Image Size | 224×224 | Required for ResNet50; 150×150 works for custom CNN |

---

## Regularisation Techniques

1. **Batch Normalization** — After every convolutional layer; stabilises training, enables higher LR
2. **Dropout** — 0.3 after conv blocks, 0.5 in dense layers; prevents co-adaptation
3. **L2 Weight Decay** — 1e-4 on all trainable layers; penalises large weights
4. **EarlyStopping** — Halts training when validation loss stops improving
5. **Data Augmentation** — Acts as implicit regularisation by expanding effective training set
6. **ReduceLROnPlateau** — Prevents overshooting optimal minima

---

## Deployment Recommendations

| Environment | Recommendation |
|-------------|----------------|
| **Development** | `python app.py` with Flask's built-in server |
| **Production** | Gunicorn + Nginx: `gunicorn -w 4 -b 0.0.0.0:5000 app:app` |
| **Cloud** | AWS EC2/ECS, Google Cloud Run, Azure App Service |
| **Containerised** | Docker with multi-stage build (Python base + model artefacts) |
| **Serverless** | Convert to TFLite for Lambda/Cloud Functions |
| **Edge** | TFLite or ONNX for on-device inference |

**Production checklist:**
- Set `FLASK_DEBUG = False` in config
- Use HTTPS with valid TLS certificate
- Add authentication/rate limiting
- Log predictions for audit trail
- Implement model versioning
- Set up health monitoring on `/health`

---

## Transfer Learning Improvements

**Beyond the included ResNet50, consider:**

1. **EfficientNetV2** — Best accuracy-efficiency trade-off; fewer parameters than ResNet50
2. **DenseNet121** — Feature reuse via dense connections; works well on medical images
3. **Vision Transformer (ViT)** — State-of-the-art attention mechanism; needs more data
4. **Progressive fine-tuning** — Gradually unfreeze deeper base layers across training phases
5. **Ensemble methods** — Average predictions from multiple architectures for higher accuracy
6. **Test-time augmentation (TTA)** — Average predictions over augmented copies of test images

---

## Ethical Considerations

> ⚠️ **This system is NOT a medical device.**

| Concern | Mitigation |
|---------|------------|
| **Misdiagnosis risk** | Clinical disclaimer displayed on every prediction; model outputs are probabilities, not diagnoses |
| **Dataset bias** | Model is trained on a specific dataset that may not represent all populations, scanner types, or imaging protocols |
| **Regulatory compliance** | Any clinical deployment requires FDA 510(k) / CE marking / local regulatory approval |
| **Patient privacy** | Uploaded images should be anonymised; HIPAA/GDPR compliance required for real patient data |
| **Transparency** | Confidence scores are always shown; low-confidence predictions should trigger human review |
| **Model drift** | Production models require periodic retraining as imaging technology and clinical standards evolve |
| **Explainability** | Consider adding Grad-CAM visualisations to show which regions influenced the prediction |
| **Human oversight** | AI should assist, not replace, radiologists — final decisions must rest with qualified clinicians |

---

## License

This project is for **educational and research purposes only**.
#   M R I - B r a i n - T u m o r - C l a s s i f i c a t i o n  
 #   M R I - B r a i n - T u m o r - C l a s s i f i c a t i o n  
 #   M R I - T u m o r - A n a l y s i s  
 