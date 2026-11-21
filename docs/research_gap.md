# Research Gap Analysis

## 1. The Gap Between Public Benchmarks and Clinical Reality
Most published computer vision models applied to handwriting and spiral datasets report classification accuracies between 90% and 98%. However, an exhaustive audit of contemporary methodologies reveals significant structural research gaps:

| Research Dimension | Conventional Approaches in Literature | This Capstone Research Project |
| :--- | :--- | :--- |
| **Data Partitioning** | Random image-level split (e.g. 70/15/15) across all files. | **Subject-Independent Group Stratification**: Zero patient overlap across train, validation, and test sets. |
| **Data Leakage Reporting** | Completely omitted or ignored despite identical participant IDs. | **Explicit Contamination Auditing**: Quantified 25/28 subject cross-split contamination in baseline split and conducted dual evaluation. |
| **Input Standardization** | Anisotropic image squashing directly to $224 \times 224$ (distorting spiral spacing). | **Aspect-Ratio Preserving Letterboxing**: Symmetric white padding that preserves ink thickness and curvature radius. |
| **Model Preprocessing** | Uniform $[0, 1]$ rescaling indiscriminately applied across CNN, VGG16, and ResNet50. | **Architecture-Specific Normalization**: Rescaling for custom CNN; ImageNet Caffe zero-centering for VGG16 and ResNet50. |
| **Explainability (XAI)** | None or decorative heatmaps without clinical sanity checking. | **Grad-CAM Saliency Maps**: Layer-specific activation maps evaluated against tremor regions vs background noise. |
| **Evaluation Metrics** | Accuracy only; ignores sensitivity/specificity trade-offs. | **Multi-Metric Clinical Matrix**: Recall (Sensitivity), Specificity, ROC-AUC, PR-AUC, Brier score, and MCC. |

## 2. Key Contributions of This Capstone
1. **Uncovering Dataset Provenance**: Traced the Roboflow dataset back to the Brazilian HandPD / NewHandPD corpus, identifying that image prefixes (`V01`, `V02`, `...`, `HE`, `PE`, `HO`, `PO`) denote specific patients and task protocols (Spirals vs Meanders).
2. **Empirical Proof of Data Leakage**: Proved that random image-level splitting artificially inflates test performance because models learn participant drawing styles rather than clinical disease pathology.
3. **Dual Benchmark Reporting**: Evaluated all three architectures (Custom CNN, VGG16, ResNet50) under BOTH the original image-level split (for benchmark reproducibility) and a rigorously isolated subject-independent split (for scientific integrity).
4. **Transparent Explainability**: Provided interactive Grad-CAM visualizations allowing clinicians and examiners to audit what image areas drive model classifications.

