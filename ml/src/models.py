"""
Deep Learning Model Architectures for Parkinson's Disease Drawing Classification.
Complies with specifications in CNN_Model_Pretrained_models.pdf:
1. Custom CNN (from scratch with 3 Conv+BN+MaxPool blocks)
2. VGG16 Transfer Learning (2-phase: frozen feature extractor then fine-tuned top layers)
3. ResNet50 Transfer Learning (2-phase: frozen base then fine-tuning with frozen BatchNorm)
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, regularizers
from typing import Tuple, Optional


def build_custom_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3),
                     learning_rate: float = 1e-3,
                     use_gap: bool = True) -> keras.Model:
    """
    Builds the Custom CNN from scratch as specified in CNN_Model_Pretrained_models.pdf:
    - Input: 224x224x3
    - Rescaling: 1/255.0 (for standalone execution)
    - 3 Conv blocks: [32, 64, 128] filters, kernel 3x3, ReLU, BatchNorm, MaxPool2D(2)
    - GlobalAveragePooling2D (or Flatten), Dropout(0.5)
    - Dense(64, ReLU)
    - Dense(1, Sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = inputs
    
    # 3 Convolutional Blocks
    for idx, filters in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu", name=f"block{idx}_conv")(x)
        x = layers.BatchNormalization(name=f"block{idx}_bn")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"block{idx}_pool")(x)
        
    # Classification Head
    if use_gap:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flatten")(x)
        
    x = layers.Dropout(0.5, name="head_dropout")(x)
    x = layers.Dense(64, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model


def build_vgg16_transfer(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         learning_rate: float = 1e-3) -> Tuple[keras.Model, keras.Model]:
    """
    Builds VGG16 Transfer Learning model:
    - ImageNet pretrained weights, include_top=False
    - Base initially frozen (Phase 1)
    - GlobalAveragePooling2D -> Dropout(0.5) -> Dense(1, Sigmoid)
    """
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze for Phase 1
    
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="vgg16_transfer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model, base_model


def configure_vgg16_finetuning(model: keras.Model,
                               base_model: keras.Model,
                               unfreeze_last_n_layers: int = 4,
                               fine_tune_lr: float = 1e-5):
    """
    Phase 2: Fine-tune the terminal convolutional block of VGG16 (e.g., block5) with a low learning rate.
    """
    base_model.trainable = True
    # Freeze all layers except the last n layers
    for layer in base_model.layers[:-unfreeze_last_n_layers]:
        layer.trainable = False
        
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model


def build_resnet50_transfer(input_shape: Tuple[int, int, int] = (224, 224, 3),
                            learning_rate: float = 1e-3) -> Tuple[keras.Model, keras.Model]:
    """
    Builds ResNet50 Transfer Learning model:
    - ImageNet pretrained weights, include_top=False
    - Base initially frozen (Phase 1)
    - GlobalAveragePooling2D -> Dropout(0.5) -> Dense(1, Sigmoid)
    """
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze for Phase 1
    
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model, base_model


def configure_resnet50_finetuning(model: keras.Model,
                                  base_model: keras.Model,
                                  unfreeze_last_n_layers: int = 15,
                                  fine_tune_lr: float = 1e-5):
    """
    Phase 2: Fine-tune top residual stage of ResNet50.
    CRITICAL: Keep BatchNormalization layers frozen to prevent running statistic corruption on small batch sizes.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n_layers]:
        layer.trainable = False
    
    # Freeze all BatchNorm layers across the entire base
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")]
    )
    return model

