<div align="center">

# NeuroSpiral: Parkinson's Disease Classification Using Deep Learning & Explainable AI (Grad-CAM)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.15-D00000.svg?logo=keras&logoColor=white)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![CLAIM 2024](https://img.shields.io/badge/Guideline-CLAIM_2024_Aligned-10B981.svg)](https://pubs.rsna.org/doi/10.1148/ryai.230248)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An end-to-end, production-ready biomedical computer vision research platform that classifies Parkinson's disease from spiral and meander drawing examinations using deep transfer learning and localizes pathological tremors via Gradient-Weighted Class Activation Mapping (Grad-CAM).*

---

</div>

## 📌 Executive Summary

Parkinson's Disease (PD) is a progressive neurodegenerative movement disorder affecting more than 10 million individuals globally. While diagnosis relies on clinical examinations scored via the MDS-UPDRS, motor symptoms such as resting tremor (4–6 Hz), bradykinesia, and muscular rigidity leave distinctive kinematic signatures on pen-and-paper motor drawing tests.

**NeuroSpiral** delivers a complete, reproducible academic capstone project that:
1. **Uncovers & Resolves Subject-Level Data Leakage:** Discovered that the standard 204-image HandPD benchmark contains only 28 unique human participants, with **89.3% subject contamination** across standard train/test splits. Established a dual-benchmark protocol featuring **Stratified Group 5-Fold Cross-Validation** with strictly **0.0% subject overlap** (CLAIM 2024 compliant).
2. **Standardizes Computer Vision Preprocessing:** Implemented aspect-ratio preserving letterboxing with white background padding (224×224) to eliminate geometric distortion of rectangular meander drawings.
3. **Compares Deep Learning Paradigms:** Demonstrates why a Custom CNN trained from scratch fails (~47.5% test accuracy) while transfer learning backbones (VGG16 and ResNet50) achieve robust discriminative power (>80% accuracy, AUC ~0.88–0.91) through two-phase fine-tuning with frozen Batch Normalization.
4. **Delivers Explainable AI (Grad-CAM):** Synthesizes 2D saliency heatmaps using gradient backpropagation into terminal convolutional feature maps (`block5_conv3` / `conv5_block3_out`), validating clinical attention on pen stroke tremors and ruling out background shortcuts.
5. **Interactive Full-Stack Web Application:** High-performance asynchronous FastAPI microservice coupled with a modern React 18 / TypeScript / Tailwind CSS dashboard featuring real-time image inference, dynamic decision thresholding ($\tau$), Grad-CAM overlays, and defense documentation.

---

## 🏛️ System Architecture

```
                                  USER INTERFACE
                   [React 18 + TypeScript + Tailwind CSS (Vite)]
             ├── Real-time Drag-and-Drop Image Uploader
             ├── Dynamic Decision Threshold Slider (0.00 – 1.00)
             ├── Real-time Confidence & Probability Gauges
             ├── Grad-CAM Saliency Heatmap & Blend Overlay
             ├── Model Comparison & Evaluation Dashboards
             └── Interactive Viva Defense & Research Mentors
                                     │
                             HTTP REST / JSON
                                     │
                               FASTAPI SERVER
                      [Asynchronous Python 3.12 Service]
             ├── POST /api/v1/predict   (Multipart upload -> PredictionResult)
             ├── POST /api/v1/explain   (Target conv tape -> Base64 Grad-CAM)
             ├── GET  /api/v1/models    (Model inventory & parameters)
             ├── GET  /api/v1/metrics   (Test metrics & 5-fold CV statistics)
             └── GET  /api/v1/plots/*   (High-res ROC, PR & Confusion plots)
                                     │
                             INFERENCE ENGINE
                      [TensorFlow 2.21 + Keras 3.15]
             ├── Preprocessing: Aspect-Ratio Letterbox (White Margin Padding)
             ├── Normalization: [0, 1] Division vs ImageNet BGR Mean Subtraction
             ├── Custom CNN: 3 Conv Blocks + BatchNorm + GAP + Dense Head
             ├── VGG16: 2-Phase Warmup + Block5 Fine-Tuning
             └── ResNet50: 2-Phase Warmup + Stage5 Tuning (Frozen BatchNorm)
```

---

## 🔬 Key Research Findings

### 1. The Subject Leakage Phenomenon
| Evaluation Protocol | Patient Overlap | Test Accuracy | Balanced Acc | Clinical Generalization Validity |
|---|---|---|---|---|
| **Naive Image-Level Split** (133 / 31 / 40) | **89.3% (25/28 subjects)** | **82.5% – 85.0%** | **83.1%** | ⚠️ Optimistically biased (memorizes pen stroke habits) |
| **Subject-Independent Group 5-Fold CV** | **0.0% (Zero Leakage)** | **~68.0% – 74.0%** | **~71.5%** | ✅ Unbiased estimate of out-of-sample patient screening |

### 2. Multi-Model Performance Comparison (Held-Out Test Set)
| Model Architecture | Trainable Params | Test Accuracy | Sensitivity (Recall) | Specificity | ROC-AUC | PR-AUC | Brier Score | Latency (CPU) |
|---|---|---|---|---|---|---|---|---|
| **Custom CNN (Scratch)** | 2.3M | 47.50% | 47.62% | 47.37% | 0.4762 | 0.5054 | 0.2520 | ~12 ms |
| **VGG16 (Transfer)** | 14.8M | **82.50%** | **85.71%** | **78.95%** | **0.8797** | **0.8841** | **0.1580** | ~32 ms |
| **ResNet50 (Transfer)** | 24.1M | **77.50%** | **80.95%** | **73.68%** | **0.8747** | **0.8712** | **0.1720** | ~28 ms |

> **Key Empirical Takeaway:** Scratch CNNs completely fail on small medical datasets ($N=133$ train images). Pretrained transfer learning backbones provide inductive visual priors (edges, curvature, loops) that are essential for successful fine-tuning on pen-and-paper stroke tasks.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ and npm
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/parkinson-detection.git
cd parkinson-detection

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python backend and ML dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Pre-Trained Models or Re-Train
To re-run the entire pipeline (dataset audit, preprocessing, model training, evaluation, and 5-fold cross-validation):
```bash
PYTHONPATH=. .venv/bin/python ml/scripts/train_models.py
```

### 4. Launch Backend API
```bash
# Start FastAPI backend on http://localhost:8000
PYTHONPATH=. .venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI: `http://localhost:8000/docs`

### 5. Launch Frontend Dashboard
```bash
# In a separate terminal
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 📁 Repository Structure

```
PARKINSON/
├── backend/                        # FastAPI REST API Microservice
│   ├── app/
│   │   ├── api/endpoints.py       # REST endpoints (/predict, /explain, /metrics)
│   │   ├── core/config.py         # Application settings, paths, disclaimers
│   │   ├── schemas/schemas.py     # Pydantic request/response validation schemas
│   │   ├── services/model_service.py # Model caching, inference, Grad-CAM tape
│   │   └── main.py                # App entrypoint, CORS, static routes
│   └── tests/                     # Backend API integration tests
│
├── frontend/                       # React 18 + TypeScript + Tailwind Web Application
│   ├── src/
│   │   ├── components/            # Navbar, Footer, MedicalDisclaimer
│   │   ├── pages/                 # Landing, Predictor, Performance, Comparison,
│   │   │                          # Explainability, Research, About, Documentation
│   │   ├── services/api.ts        # Client fetch wrappers for backend API
│   │   ├── types/index.ts         # TypeScript schema contracts
│   │   ├── App.tsx                # Master tab router & application shell
│   │   └── main.tsx               # DOM entrypoint
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── ml/                             # Core Deep Learning Research Pipeline
│   ├── artifacts/                 # Serialized .keras weights, metrics JSON, manifest
│   ├── data/                      # Raw and curated HandPD dataset (204 images)
│   ├── outputs/                   # Generated evaluation curves & EDA visualizations
│   │   ├── eda/                   # Class distribution, dimensions, leakage map
│   │   └── evaluation/            # ROC curves, PR curves, Confusion matrices
│   ├── scripts/
│   │   └── train_models.py        # Master training, evaluation & 5-fold CV runner
│   └── src/
│       ├── data_audit.py          # Programmatic subject-leakage detector
│       ├── data_loader.py         # DrawingDataset & StratifiedGroupKFold builder
│       ├── gradcam.py             # Gradient tape & ReLU feature rectification
│       ├── metrics.py             # Multi-metric biomedical evaluation suite
│       ├── models.py              # Custom CNN, VGG16, ResNet50 builder functions
│       ├── preprocessing.py       # Aspect-ratio preserving letterboxing
│       └── visualization.py       # Matplotlib diagnostic plotting routines
│
├── docs/                           # Academic Capstone Thesis Documentation
│   ├── architecture.md            # Hardware, software, and interface specifications
│   ├── dataset_documentation.md   # Dataset composition, subjects, resolutions
│   ├── limitations.md             # Scientific, clinical, and regulatory constraints
│   ├── literature_review.md       # Comprehensive review of motor drawing biomarkers
│   ├── methodology.md             # Complete experimental protocols and hyperparameters
│   ├── presentation.md            # Slide-by-slide 20-minute defense deck with script
│   ├── research_gap.md            # Formal analysis of the subject leakage artifact
│   ├── research_references.md     # Peer-reviewed bibliography and standards
│   ├── technical_report.md        # Complete 7-chapter university capstone thesis
│   └── viva_questions.md          # 42 in-depth viva examination Q&As
│
├── requirements.txt                # Python dependencies
└── README.md                       # Master project presentation
```

---

## 🧠 Explainable AI: Grad-CAM Formulation

To ensure predictions are grounded in clinical drawing pathology rather than spurious background shortcuts, we calculate local saliency via **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

1. **Gradient Flow:** For target class $c$ (Parkinson), we compute the gradient of unnormalized score $y^c$ with respect to feature maps $A^k$ of the terminal convolutional stage:
   $$\frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Neuron Importance Weights:** Spatial global average pooling collapses gradients into importance scalars $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{u \cdot v} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Rectified Linear Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The resulting 2D heatmap is upsampled to $224 \times 224$ and blended onto the letterboxed drawing. Saliency concentrates directly on pen stroke micro-hesitations and irregular loop transitions.

---

## ⚖️ Academic & Medical Disclaimer

> **IMPORTANT NOTICE:** This software system is an **academic research and educational prototype** developed as a university capstone project. It is **NOT** a clinically validated medical device and has not received regulatory clearance or approval from the US FDA, European CE/MDR, CDSCO, or any medical regulatory agency. It must **NEVER** be utilized for autonomous clinical diagnosis, medical screening, triage, or treatment selection. All medical evaluations must be conducted by licensed neurologists utilizing comprehensive clinical motor examinations.

---

## 📜 Citation & References

```bibtex
@article{pereira2016handwriting,
  title={Handwriting analysis for the diagnosis of Parkinson's disease: a review and deep learning perspectives},
  author={Pereira, Clayton R and Pereira, Danilo R and Weber, Silke AT and others},
  journal={Artificial Intelligence in Medicine},
  volume={66},
  pages={31--45},
  year={2016}
}

@article{mongan2024claim,
  title={Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update},
  author={Mongan, John and Moy, Linda and Kahn, Charles E and others},
  journal={Radiology: Artificial Intelligence},
  volume={6},
  number={2},
  pages={e230248},
  year={2024}
}

@article{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={618--626},
  year={2017}
}
```

---

<div align="center">
  <sub>Developed for University Final Year Engineering Capstone Examination.</sub>
</div>

