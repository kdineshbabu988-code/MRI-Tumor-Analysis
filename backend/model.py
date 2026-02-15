"""
model.py — CNN architectures for brain tumor classification.

Provides two model builders:
  1. build_custom_cnn()    — Lightweight custom CNN (~2M params)
  2. build_resnet50_model() — Transfer learning with ResNet50 (ImageNet weights)

Both return a compiled Keras Model ready for training.
"""

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam

import config


# ═════════════════════════════════════════════════════════════════════════════
#  1. CUSTOM CNN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

def _conv_block(x, filters: int, name_prefix: str):
    """
    Convolutional block: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout.

    This pattern ensures:
    - Batch normalization stabilises training and allows higher learning rates.
    - Two conv layers per block capture richer spatial features.
    - MaxPooling reduces spatial dimensions by half.
    - Dropout prevents co-adaptation of feature detectors.
    """
    x = layers.Conv2D(
        filters, (3, 3), padding="same", name=f"{name_prefix}_conv1",
        kernel_regularizer=regularizers.l2(config.L2_WEIGHT_DECAY),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)

    x = layers.Conv2D(
        filters, (3, 3), padding="same", name=f"{name_prefix}_conv2",
        kernel_regularizer=regularizers.l2(config.L2_WEIGHT_DECAY),
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)

    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(x)
    x = layers.Dropout(config.DROPOUT_CONV, name=f"{name_prefix}_drop")(x)
    return x


def build_custom_cnn() -> models.Model:
    """
    Build a custom 5-block CNN for 4-class brain tumor classification.

    Architecture:
        Input (224×224×3)
        → Block 1 (32 filters)  → 112×112
        → Block 2 (64 filters)  → 56×56
        → Block 3 (128 filters) → 28×28
        → Block 4 (256 filters) → 14×14
        → Block 5 (512 filters) → 7×7
        → GlobalAveragePooling2D
        → Dense(256, relu) + Dropout
        → Dense(4, softmax)

    Returns:
        Compiled Keras Model.
    """
    inputs = layers.Input(shape=config.INPUT_SHAPE, name="input_image")

    x = _conv_block(inputs, 32,  "block1")
    x = _conv_block(x,     64,  "block2")
    x = _conv_block(x,     128, "block3")
    x = _conv_block(x,     256, "block4")
    x = _conv_block(x,     512, "block5")

    # ── Classification head ──────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(
        256, activation="relu", name="dense_256",
        kernel_regularizer=regularizers.l2(config.L2_WEIGHT_DECAY),
    )(x)
    x = layers.Dropout(config.DROPOUT_DENSE, name="dense_dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="BrainTumor_CustomCNN")

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[OK] Custom CNN built - {model.count_params():,} parameters")
    model.summary(print_fn=lambda line: print(f"  {line}"))
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  2. TRANSFER LEARNING — ResNet50
# ═════════════════════════════════════════════════════════════════════════════

def build_resnet50_model(fine_tune_layers: int = 0) -> models.Model:
    """
    Build a brain tumor classifier using ResNet50 pre-trained on ImageNet.

    Strategy:
        - The ResNet50 base is loaded without its top classification layers.
        - A custom classification head is added on top.
        - By default, the entire base is frozen (feature extraction only).
        - Set fine_tune_layers > 0 to unfreeze the last N layers for fine-tuning.

    Args:
        fine_tune_layers: Number of base layers to unfreeze (from the top).
                          0 = pure feature extraction (default for initial training).

    Returns:
        Compiled Keras Model.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE,
    )

    # Freeze / selectively unfreeze base layers
    if fine_tune_layers == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    # ── Classification head ──────────────────────────────────────────────
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="resnet_gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(
        256, activation="relu", name="head_dense_256",
        kernel_regularizer=regularizers.l2(config.L2_WEIGHT_DECAY),
    )(x)
    x = layers.Dropout(config.DROPOUT_DENSE, name="head_dropout")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(base_model.input, outputs, name="BrainTumor_ResNet50")

    # Use a lower LR for fine-tuning to avoid destroying pre-trained weights
    lr = config.LEARNING_RATE / 10 if fine_tune_layers > 0 else config.LEARNING_RATE

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"\n[OK] ResNet50 model built - {model.count_params():,} total params")
    print(f"    Trainable layers: {trainable}  |  Frozen layers: {frozen}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  3. TRANSFER LEARNING — EfficientNetB0
# ═════════════════════════════════════════════════════════════════════════════

def build_efficientnet_model(fine_tune: bool = False) -> models.Model:
    """
    Build a brain tumor classifier using EfficientNetB0 (ImageNet weights).
    EfficientNet is generally faster and more accurate than ResNet50.

    Args:
        fine_tune: If True, unfreeze the top 20 layers for adaptation.

    Returns:
        Compiled Keras Model.
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE,
    )

    # Freeze base model
    base_model.trainable = False

    if fine_tune:
        base_model.trainable = True
        # Freeze all except the last 20 layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    # ── Classification head ──────────────────────────────────────────────
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="efficientnet_gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    
    # EfficientNet has a rich feature set, so we can use a slightly larger dense layer or higher dropout
    x = layers.Dense(
        256, activation="relu", name="head_dense_256",
        kernel_regularizer=regularizers.l2(config.L2_WEIGHT_DECAY),
    )(x)
    x = layers.Dropout(config.DROPOUT_DENSE, name="head_dropout")(x)
    
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(base_model.input, outputs, name="BrainTumor_EfficientNetB0")

    # Lower LR if fine-tuning
    lr = config.FINE_TUNE_LEARNING_RATE if fine_tune else config.LEARNING_RATE

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    frozen = sum(1 for l in model.layers if not l.trainable)
    print(f"\n[OK] EfficientNetB0 model built - {model.count_params():,} total params")
    print(f"    Trainable layers: {trainable}  |  Frozen layers: {frozen}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER — Model selection by name
# ═════════════════════════════════════════════════════════════════════════════

def build_model(model_type: str = "custom", **kwargs) -> models.Model:
    """
    Factory function to build a model by name.

    Args:
        model_type: 'custom' or 'resnet50'.
        **kwargs: Forwarded to the specific builder (e.g. fine_tune_layers).

    Returns:
        Compiled Keras Model.
    """
    builders = {
        "custom": build_custom_cnn,
        "resnet50": build_resnet50_model,
        "efficientnet": build_efficientnet_model,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(builders.keys())}")
    return builders[model_type](**kwargs)
