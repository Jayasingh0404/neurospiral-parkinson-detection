# Dataset Documentation and Scientific Audit

## 1. Dataset Overview
- **Dataset Title**: Parkinson Disease Detection Spiral and Meander Dataset
- **Export Source**: Roboflow Universe (`gokul-vscgs/parkinson-disease-detection` & `jaya-singh/parkinson-disease-detection-6z94a`)
- **Underlying Provenance**: HandPD / NewHandPD Database (Universidade Estadual Paulista - UNESP, Brazil)
- **Total Images**: 204
- **Image Format**: JPEG (auto-oriented, EXIF stripped)
- **Color Mode**: 3-channel RGB across all 204 images
- **Class Labels**:
  1. `healthy` (Negative Class, Label 0): 102 images (50.0%)
  2. `parkinson` (Positive Class, Label 1): 102 images (50.0%)
- **Class Imbalance Ratio**: $1.00$ (Perfect balance)

## 2. Split Partitioning in Original Dataset
The dataset was pre-partitioned into three folders:
| Split | Total Images | Healthy Count | Parkinson Count | Class Ratio (H:P) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 133 (65.2%) | 64 (48.1%) | 69 (51.9%) | 0.93 : 1 |
| **Validation** | 31 (15.2%) | 19 (61.3%) | 12 (38.7%) | 1.58 : 1 |
| **Test** | 40 (19.6%) | 19 (47.5%) | 21 (52.5%) | 0.90 : 1 |
| **Total** | **204** | **102** | **102** | **1.00 : 1** |

## 3. Dimensional Analysis
Our image audit (`ml/artifacts/dataset_manifest.csv`) established that dimensions are **not uniform**:
- **Square Spiral Images**: 99 images have exact dimensions of $256 \times 256$ pixels (aspect ratio 1.00). These correspond to Archimedean spiral drawing tasks.
- **Rectangular Meander Images**: 105 images have widths around 512 pixels with heights varying between 138 and 516 pixels (aspect ratios ranging from 0.99 to 3.71). These correspond to horizontal sinusoidal meander wave drawings.
- **Implication for Preprocessing**: Direct isotropic resizing to $224 \times 224$ severely compresses meander waves vertically, distorting stroke frequency. Our pipeline applies **aspect-ratio preserving letterboxing with white padding**, preventing geometric distortion.

## 4. Subject-Level Data Leakage Audit
- **Unique Subjects Identified**: 28 distinct individuals
  - Healthy Cohort: 13 subjects (`H_V01` through `H_V12`, plus `H_V55`)
  - Parkinson Cohort: 15 subjects (`P_V01` through `P_V15`)
- **Cross-Split Contamination**: **25 out of 28 subjects (89.3%)** have drawing exams distributed across multiple splits simultaneously.
- **Test Set Contamination**: 100% of Parkinson subjects and 100% of Healthy subjects present in the test set have companion images in the training set.
- **Scientific Resolution**: In accordance with CLAIM 2024 and TRIPOD+AI guidelines, we implement an alternative **Subject-Independent Stratified Group 5-Fold Cross-Validation** protocol, ensuring completely disjoint participant cohorts.

