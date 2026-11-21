# System Architecture & Technical Specifications
## NeuroSpiral: Parkinson's Disease Classification and Explainable AI

This document provides a detailed technical description of the hardware, software, algorithmic, and interface architectures comprising the NeuroSpiral capstone system.

---

## 1. High-Level System Architecture

The NeuroSpiral application is structured as a decoupled, multi-tier client-server architecture:

```
+-------------------------------------------------------------------------------+
|                               Presentation Layer                              |
|                    React 18 + TypeScript + Tailwind CSS (Vite)                |
|  - Real-time Drawing Upload & Canvas Drag-and-Drop                            |
|  - Interactive Decision Threshold Adjustment Slider (\tau \in [0.0, 1.0])     |
|  - Dynamic Confidence & Probability Gauge Meters                              |
|  - Grad-CAM Heatmap & Blend Overlay Viewer                                     |
|  - Comprehensive Multi-Model Performance Dashboards & ROC Curves              |
|  - Academic Literature & CLAIM 2024 Audit Matrix Explorer                     |
+-------------------------------------------------------------------------------+
                                      |
                           HTTP / REST (JSON & Base64)
                                      |
+-------------------------------------------------------------------------------+
|                               Application API Layer                           |
|                         FastAPI Asynchronous Microservice                     |
|  - POST /api/v1/predict: Multipart Image Ingestion & Confidence Estimation    |
|  - POST /api/v1/explain: Dynamic Gradient Backpropagation & Heatmap Rendering |
|  - GET  /api/v1/models: Model Metadata & Parameter Inventory                  |
|  - GET  /api/v1/metrics: Test-Set & Subject-Independent Cross-Validation Stats|
|  - GET  /api/v1/plots/{category}/{name}: Static High-Res Diagnostic Plots     |
|  - CORS Middleware, Pydantic Schema Validation, Error Handling                |
+-------------------------------------------------------------------------------+
                                      |
                           Python Execution Runtime
                                      |
+-------------------------------------------------------------------------------+
|                              Deep Learning Core                               |
|                         TensorFlow 2.21 + Keras 3.15                          |
|  - Preprocessing Engine: Aspect-Ratio Preserving Letterbox (White Padding)    |
|  - Channel Normalizers: [0, 1] Division vs ImageNet BGR Mean Subtraction      |
|  - Model Zoo (In-Memory Cached Models):                                       |
|      * Custom CNN (3 Conv Blocks, BatchNorm, GAP, Dropout, Dense)             |
|      * VGG16 Transfer (ImageNet Weights, Frozen Blocks 1-4, Tuned Block 5)    |
|      * ResNet50 Transfer (Residual Bottlenecks, Frozen BatchNorm)             |
|  - Explainability Engine: tf.GradientTape Automatic Differentiation           |
|      * Targeted Layer Saliency Mapping (\alpha_k^c GAP + ReLU Rectification)  |
|      * Matplotlib Jet Colormap Synthesis & Base64 In-Memory Encoding          |
+-------------------------------------------------------------------------------+
                                      |
                           Filesystem Artifact Storage
                                      |
+-------------------------------------------------------------------------------+
|                             Persistence & Assets                              |
|  - Raw & Curated Dataset: ml/data/raw/ (204 HandPD Images)                    |
|  - Serialized Keras Models: ml/artifacts/*.keras                              |
|  - Evaluation Results: ml/artifacts/models_evaluation.json                    |
|  - Training Histories: ml/artifacts/training_history.json                     |
|  - Diagnostic Figures: ml/outputs/eda/ and ml/outputs/evaluation/             |
+-------------------------------------------------------------------------------+
```

---

## 2. Preprocessing & Input Data Pipeline

### 2.1 Geometric Transformation (Aspect-Ratio Preserving Letterbox)
Given an arbitrary input drawing image $I_{\text{raw}}$ of dimensions $(W_{\text{raw}}, H_{\text{raw}}, 3)$:
1. The target dimension is $S = 224$.
2. The isotropic scale factor is:
   $$s = \min\left(\frac{S}{W_{\text{raw}}}, \frac{S}{H_{\text{raw}}}\right)$$
3. The scaled intermediate dimensions are $W_{\text{new}} = \text{round}(W_{\text{raw}} \cdot s)$, $H_{\text{new}} = \text{round}(H_{\text{raw}} \cdot s)$.
4. The image is rescaled using bicubic interpolation.
5. A blank white canvas $I_{\text{canvas}}$ of shape $(S, S, 3)$ filled with value $255$ is allocated.
6. The rescaled image is pasted into the canvas centered at offsets:
   $$\Delta x = \lfloor (S - W_{\text{new}}) / 2 \rfloor, \quad \Delta y = \lfloor (S - H_{\text{new}}) / 2 \rfloor$$

### 2.2 Model-Specific Channel Normalization
- **Custom CNN:**
  $$I_{\text{norm}} = \frac{I_{\text{canvas}}}{255.0} \in [0.0, 1.0]$$
- **VGG16 and ResNet50:**
  1. Convert RGB to BGR: $I_{\text{BGR}} = I_{\text{canvas}}[:, :, ::-1]$.
  2. Subtract ImageNet channel means:
     $$I_{\text{norm}}^{(c)} = I_{\text{BGR}}^{(c)} - \mu^{(c)}, \quad \mu = [103.939, 116.779, 123.68]$$

---

## 3. Deep Learning Model Topologies

### 3.1 Custom CNN (Trained from Scratch)
- Input: $(224, 224, 3)$
- **Stage 1:** Conv2D(32, $3\times3$, padding='same', ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$) $\to$ Dropout(0.20)
- **Stage 2:** Conv2D(64, $3\times3$, padding='same', ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$) $\to$ Dropout(0.20)
- **Stage 3:** Conv2D(128, $3\times3$, padding='same', ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$) $\to$ Dropout(0.30)
- **Dense Head:** GlobalAveragePooling2D $\to$ Dense(128, ReLU) $\to$ Dropout(0.40) $\to$ Dense(1, Sigmoid)

### 3.2 VGG16 (Two-Phase Transfer Learning)
- Base: VGG16 convolutional backbone pretrained on ImageNet (13 conv layers organized into 5 blocks).
- **Phase 1 (Warmup):**
  - All backbone layers frozen (`layer.trainable = False`).
  - Head: GlobalAveragePooling2D $\to$ Dense(256, ReLU, $L_2=10^{-4}$) $\to$ Dropout(0.40) $\to$ Dense(1, Sigmoid).
  - Optimizer: Adam ($\eta = 10^{-4}$). Epochs: 8.
- **Phase 2 (Fine-Tuning):**
  - Unfreeze `block5_conv1`, `block5_conv2`, `block5_conv3`.
  - Optimizer: Adam with Cosine Decay ($\eta_{\text{initial}} = 10^{-5}$). Epochs: 12.

### 3.3 ResNet50 (Two-Phase Transfer with Frozen BatchNorm)
- Base: ResNet50 residual backbone pretrained on ImageNet (50 layers with bottleneck skip connections).
- **Phase 1 (Warmup):**
  - Entire backbone frozen.
  - Head: GlobalAveragePooling2D $\to$ Dense(256, ReLU, $L_2=10^{-4}$) $\to$ Dropout(0.40) $\to$ Dense(1, Sigmoid).
  - Optimizer: Adam ($\eta = 10^{-4}$). Epochs: 8.
- **Phase 2 (Fine-Tuning):**
  - Unfreeze stage 5 residual blocks (`conv5_block1` through `conv5_block3`).
  - **Constraint:** All `BatchNormalization` layers explicitly kept frozen (`trainable = False`, `training=False`) to preserve ImageNet running statistics on small mini-batches.
  - Optimizer: Adam ($\eta = 10^{-5}$). Epochs: 12.

---

## 4. Grad-CAM Explainability Engine

```
Input Image (224x224x3)
         |
         v
Target Conv Layer [e.g., block5_conv3 (7x7x512)]  --> Activation Maps A^k
         |
         v
Dense Classification Head
         |
         v
Class Logit y^c (Parkinson)
         |
         v [tf.GradientTape.gradient]
Gradients dy^c / dA^k
         |
         v [Global Average Pooling: (1/Z) \sum_{i,j} ...]
Channel Weights \alpha_k^c
         |
         v [\sum_k \alpha_k^c A^k]
Linear Combination
         |
         v [ReLU Rectification]
Coarse Saliency Map L_{Grad-CAM}^c (7x7)
         |
         v [Bilinear Upsampling to 224x224 & Min-Max Normalization]
High-Resolution Heatmap
         |
         v [Matplotlib Jet Colormap + Alpha Blend (\alpha = 0.40)]
Rendered Heatmap & Composite Image (Base64 PNG)
```

---

## 5. REST API Interface Contracts

### 5.1 POST `/api/v1/predict`
- **Request:** `multipart/form-data`
  - `file`: Binary image file (JPEG/PNG)
  - `model_name`: String (`"custom_cnn"` | `"vgg16"` | `"resnet50"`, default: `"vgg16"`)
  - `threshold`: Float ($0.0 \le \tau \le 1.0$, default: $0.50$)
- **Response Schema (`PredictionResult`):**
  ```json
  {
    "model_name": "vgg16",
    "predicted_class": "parkinson",
    "prediction_label": "Parkinson's Disease Detected",
    "confidence_score": 0.884,
    "decision_threshold": 0.50,
    "probabilities": {
      "healthy": 0.116,
      "parkinson": 0.884
    },
    "inference_time_ms": 24.6,
    "letterboxed_preview_base64": "data:image/png;base64,...",
    "medical_disclaimer": "This system is a research prototype..."
  }
  ```

### 5.2 POST `/api/v1/explain`
- **Request:** `multipart/form-data`
  - `file`: Binary image file
  - `model_name`: String (`"vgg16"` | `"resnet50"` | `"custom_cnn"`)
- **Response Schema (`ExplanationResult`):**
  ```json
  {
    "model_name": "vgg16",
    "predicted_class": "parkinson",
    "confidence_score": 0.884,
    "target_layer_name": "block5_conv3",
    "original_image_base64": "data:image/png;base64,...",
    "heatmap_image_base64": "data:image/png;base64,...",
    "overlay_image_base64": "data:image/png;base64,...",
    "interpretation": "Strongest activations concentrate on high-frequency stroke oscillations...",
    "limitations": "Grad-CAM represents coarse 7x7 spatial resolution...",
    "medical_disclaimer": "For educational and research evaluation only."
  }
  ```

### 5.3 GET `/api/v1/metrics`
- **Response:** JSON object containing full test-set evaluation metrics for all 3 models (Accuracy, Recall, Specificity, F1, Balanced Acc, ROC-AUC, PR-AUC, MCC, Brier Score, Confusion Matrix, ROC/PR curve points) plus subject-independent 5-fold cross-validation aggregate statistics.

