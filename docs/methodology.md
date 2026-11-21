# Technical Methodology and Experimental Design

## 1. End-to-End Workflow Architecture
The methodological pipeline is organized into five rigorous, reproducible stages conforming to CLAIM 2024 standards:

```
[Raw Dataset (204 Images)]
          │
          ▼
[Data Audit & Quality Verification] ──► Subject Extraction & Leakage Quantification
          │
          ▼
[Aspect-Preserving Letterboxing] ──► White-padded 224x224x3 Canvas (Zero geometric warp)
          │
          ├───────────────────────────────┐
          ▼                               ▼
[Split Benchmark 1: Image-Level]   [Split Benchmark 2: Subject-Independent (GroupKFold)]
  - Train: 133 images                - 5-Fold Stratified Grouping
  - Valid: 31 images                 - 0% patient leakage across folds
  - Test: 40 images                  - Unbiased generalization assessment
          │                               │
          └───────────────┬───────────────┘
                          ▼
            [Deep Learning Model Suite]
            ├── Model 1: Custom CNN (from scratch)
            ├── Model 2: VGG16 (ImageNet transfer + fine-tuning)
            └── Model 3: ResNet50 (ImageNet transfer + fine-tuning)
                          │
                          ▼
             [Explainable AI: Grad-CAM]
             - Saliency localization on terminal conv layer
             - Saliency sanity checks
                          │
                          ▼
           [Full-Stack Production System]
           ├── FastAPI REST Inference Service
           └── React + TypeScript Research Dashboard
```

## 2. Model Architectures and Hyperparameters

### 2.1 Model 1: Custom CNN (Trained from Scratch)
- **Input Dimension**: $224 \times 224 \times 3$
- **Input Normalization**: In-graph rescaling layer $\times \frac{1}{255.0}$
- **Convolutional Feature Extractor**:
  - Block 1: Conv2D(32, $3 \times 3$, ReLU) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D($2 \times 2$)
  - Block 2: Conv2D(64, $3 \times 3$, ReLU) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D($2 \times 2$)
  - Block 3: Conv2D(128, $3 \times 3$, ReLU) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D($2 \times 2$)
- **Dense Classification Head**:
  - GlobalAveragePooling2D (or Flatten with Dropout 0.5)
  - Dense(64, ReLU)
  - Dense(1, Sigmoid activation)
- **Compilation**: Adam optimizer ($\text{lr} = 10^{-3}$), Binary Cross-Entropy loss.

### 2.2 Model 2: VGG16 Transfer Learning
- **Base**: VGG16 pretrained on ImageNet (`include_top=False`)
- **Input Normalization**: Caffe-style zero-centering (RGB $\rightarrow$ BGR, subtract ImageNet channel means: $[103.939, 116.779, 123.680]$)
- **Classification Head**: GlobalAveragePooling2D $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(1, Sigmoid)
- **Two-Phase Training Protocol**:
  - **Phase 1 (Head Warmup)**: Backbone frozen (`trainable=False`), Adam ($\text{lr} = 10^{-3}$), up to 15 epochs.
  - **Phase 2 (Fine-Tuning)**: Unfreeze last 4 convolutional layers (`block5_conv1`, `block5_conv2`, `block5_conv3`), freeze early layers, Adam ($\text{lr} = 10^{-5}$), up to 15 epochs.

### 2.3 Model 3: ResNet50 Transfer Learning
- **Base**: ResNet50 pretrained on ImageNet (`include_top=False`)
- **Input Normalization**: Caffe-style zero-centering
- **Classification Head**: GlobalAveragePooling2D $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(1, Sigmoid)
- **Two-Phase Training Protocol**:
  - **Phase 1**: Backbone frozen, Adam ($\text{lr} = 10^{-3}$), up to 15 epochs.
  - **Phase 2**: Unfreeze top residual stage (`conv5_block3`), keeping BatchNormalization layers frozen in inference mode to prevent statistic corruption on small batch sizes, Adam ($\text{lr} = 10^{-5}$), up to 15 epochs.

## 3. Training Regularization and Callbacks
- **EarlyStopping**: Monitored on `val_loss`, patience $= 5$, `restore_best_weights=True`.
- **ReduceLROnPlateau**: Monitored on `val_loss`, reduction factor $= 0.5$, patience $= 3$, minimum $\text{lr} = 10^{-7}$.
- **ModelCheckpoint**: Persisting best weights per training run based on minimal validation loss.
- **Class Balancing**: Balanced class weighting applied during fit:
  $$w_j = \frac{N}{2 \times N_j}$$

