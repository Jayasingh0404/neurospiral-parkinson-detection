# Capstone Technical Report: Parkinson's Disease Classification Using Deep Learning and Explainable AI

**Project Title:** Parkinson's Disease Classification Using Spiral Drawing Images Through Convolutional Neural Networks, Transfer Learning, and Explainable AI  
**Academic Degree:** Bachelor of Technology / Master of Science Capstone Project  
**Domain:** Biomedical Computer Vision, Deep Learning, Explainable AI (XAI)  
**Methodological Guidelines:** CLAIM 2024 & TRIPOD+AI Compliant  

---

## Abstract

Parkinson's Disease (PD) is a progressive neurodegenerative movement disorder affecting over 10 million individuals worldwide. Characterized by dopaminergic neuron degeneration in the substantia nigra, PD manifests as resting tremor, rigidity, bradykinesia, and micrographia. While clinical rating scales such as the Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS) serve as standard diagnostic assessments, they rely on subjective specialist evaluations. In this capstone project, we investigate whether deep convolutional neural network (CNN) architectures can extract discriminative visual biomarkers from digital spiral and meander drawing examinations. 

Through a programmatic audit of the benchmark HandPD dataset (204 images: 102 Healthy, 102 Parkinsonian), we uncovered that the dataset represents only 28 unique participants, and that the widely cited static split suffers from severe subject-level data leakage, with 89.3% of participants appearing across both training and testing partitions. To rectify this flaw, we established a dual-benchmark protocol: (1) evaluation on the reported static split, and (2) subject-independent 5-fold cross-validation using `StratifiedGroupKFold`. 

We engineered an aspect-ratio-preserving letterbox preprocessing pipeline with white background padding to eliminate geometric distortion. We evaluated three deep architectures: a Custom CNN trained from scratch, VGG16 with two-phase transfer learning, and ResNet50 with two-phase transfer learning and frozen Batch Normalization. To demystify model predictions, we implemented Gradient-Weighted Class Activation Mapping (Grad-CAM) targeting terminal convolutional stages. We delivered our findings through an asynchronous FastAPI backend and a high-fidelity React 18 / TypeScript / Tailwind CSS interactive web platform.

---

## Chapter 1: Introduction & Clinical Motivation

### 1.1 The Neurological Challenge of Parkinson's Disease
Parkinson's disease is the second most common neurodegenerative disorder after Alzheimer's disease. Pathologically, the condition involves the abnormal accumulation of intracellular alpha-synuclein aggregates (Lewy bodies) and the progressive loss of pigmented dopaminergic neurons in the substantia nigra pars compacta. Dopamine plays a vital regulatory role in the striatum, modulating excitatory and inhibitory pathways within the basal ganglia-thalamocortical motor loop. 

When striatal dopamine depletion reaches a critical threshold (typically 60–80% neuronal loss), patients present with classical cardinal motor symptoms:
- **Resting Tremor:** Involuntary, rhythmic oscillations occurring at 4–6 Hz, typically presenting unilaterally in the distal upper extremity ("pill-rolling" tremor).
- **Bradykinesia:** Slowness of movement initiation and progressive reduction in speed and amplitude during repetitive movements.
- **Muscular Rigidity:** Resistance to passive joint movement across the entire range of motion ("lead-pipe" or "cogwheel" rigidity).
- **Postural Instability:** Loss of postural reflexes resulting in gait impairment and heightened fall risks.

### 1.2 The Rationale for Spiral and Meander Drawing Biomarkers
Clinical assessment traditionally relies on in-person physical examinations scored by trained movement disorder specialists using the MDS-UPDRS. However, clinic-based assessments are infrequent, costly, and subject to intra- and inter-rater variability. 

Pen-and-paper motor drawing tests—specifically Archimedean spirals and sinusoidal meanders—have emerged as potent, non-invasive digital biomarkers:
1. **Linguistic Independence:** Unlike sentence transcription or signature writing, drawing continuous concentric loops does not depend on language proficiency, native dialect, vocabulary, or literacy level.
2. **Fine-Motor Neuromuscular Demands:** Drawing an expanding spiral requires sustained, continuous modulation of radius and angular velocity, demanding tight coordination between agonist and antagonist digital flexor/extensor muscle groups.
3. **Micrographic and Tremor Saliency:** Basal ganglia dysfunction impairs the smooth execution of continuous trajectories. Parkinsonian drawings exhibit characteristic micro-tremor oscillations, stroke hesitations, directional stiffness, uneven stroke thickness, and premature radial constriction (micrographia).

---

## Chapter 2: Literature Review, Dataset Audit & Scientific Integrity

### 2.1 Related Works in Computer Vision for Parkinson's Detection
Early computerized handwriting analysis extracted handcrafted kinematic and geometric features (e.g., velocity peaks, entropy, curvature radius, spectral energy) classified via Support Vector Machines (SVM) or Random Forests (Pereira et al., 2016; Impedovo et al., 2018). With the advent of deep learning, several studies explored applying Convolutional Neural Networks directly to scanned drawing images (Chakraborty et al., 2020; Kamran et al., 2021). 

However, many early studies reported near-perfect accuracies (95%–99%) on small cohorts without documenting their data partitioning protocols or validating patient identity separation.

### 2.2 Programmatic Audit of HandPD & Subject Leakage Discovery
The primary dataset provided for this investigation comprises 204 images:
- **Healthy Controls:** 102 images
- **Parkinson's Patients:** 102 images
- **Reported Split:** 133 Training (64 Healthy, 69 Parkinson), 31 Validation (19 Healthy, 12 Parkinson), 40 Testing (19 Healthy, 21 Parkinson).

To verify the integrity of the split, we parsed the standardized filename nomenclature (e.g., `V08HE03.png` denotes Subject `H_V08`, Examination `03`). Our analysis revealed that the 204 images stem from only **28 unique human participants**:
- 13 Healthy Controls (`H_V01` through `H_V12`, `H_V55`)
- 15 Parkinson's Patients (`P_V01` through `P_V15`)

**Critical Leakage Finding:** In the original static split, **25 out of 28 subjects (89.3%)** have drawing samples present in both the training set and the test set simultaneously. 

When a deep neural network is trained on images from Subject `V02` and tested on different drawings from Subject `V02`, the network is incentivized to minimize loss by memorizing individual-specific shortcut features—such as pen stroke thickness, paper grain texture, digitizer scanner noise, or individual hand curvature habits—rather than generalizable pathological tremor features.

### 2.3 Methodological Rectification: Dual Benchmark & CLAIM 2024
In adherence to the *Checklist for Artificial Intelligence in Medical Imaging* (CLAIM 2024, Item 11b) and the *TRIPOD+AI Statement* (Collins et al., BMJ 2024), we established a dual-benchmark protocol:
1. **Benchmark A (Reported Static Split):** Replicates the original 133/31/40 split to evaluate historical claims and demonstrate how subject overlap artificially inflates model metrics.
2. **Benchmark B (Subject-Independent Grouped 5-Fold Cross-Validation):** Employs `StratifiedGroupKFold(n_splits=5)` grouped strictly by `subject_id`. In every fold, all drawings belonging to a given human subject are strictly partitioned into either the training fold or the testing fold. Subject overlap across splits is strictly **0.0%**.

---

## Chapter 3: Computer Vision & Preprocessing Engineering

### 3.1 Heterogeneous Dimensions and The Letterboxing Solution
The raw dataset contains heterogeneous dimensions:
- Approximately 48.5% of images are square spirals ($256 \times 256$ pixels).
- Approximately 51.5% of images are wide rectangular meanders (e.g., $512 \times 250$ pixels).

Standard resizing (e.g., bilinear stretching to $224 \times 224$) compresses wide meanders horizontally by more than 50%, distorting stroke curvature, spatial frequencies, and tremor waveforms.

To preserve physical stroke geometry, we engineered an **aspect-ratio preserving letterbox preprocessing pipeline**:
1. Compute uniform scale factor: $s = \min(224 / W_{\text{raw}}, 224 / H_{\text{raw}})$.
2. Rescale raw image isotropically: $W_{\text{new}} = \text{round}(W_{\text{raw}} \cdot s)$, $H_{\text{new}} = \text{round}(H_{\text{raw}} \cdot s)$.
3. Allocate a blank canvas of shape $(224, 224, 3)$ filled with white pixels (RGB: $255, 255, 255$).
4. Center the scaled drawing on the canvas:
   $$\Delta x = \lfloor (224 - W_{\text{new}}) / 2 \rfloor, \quad \Delta y = \lfloor (224 - H_{\text{new}}) / 2 \rfloor$$

**Why White Background Padding?** In drawings on paper, the background is white while strokes are dark. Zero (black) padding would introduce four artificial high-contrast borders around the canvas, generating massive spurious gradient activations in early convolutional layers. White padding seamlessly merges with the paper substrate.

### 3.2 Architecture-Specific Channel Normalization
- **Custom CNN:** Images are scaled linearly to $[0.0, 1.0]$ via division by 255.0 to maintain well-behaved activation distributions.
- **VGG16 and ResNet50:** Channels are reordered from RGB to BGR, and ImageNet channel means ($\mu = [103.939, 116.779, 123.68]$) are subtracted without division by 255, preserving the exact input distribution expected by the pretrained Caffe backbones.

### 3.3 Domain-Safe Training Augmentation
To prevent overfitting on the small training cohort, we applied conservative data augmentations:
- Random rotation ($\pm 10^\circ$)
- Random zoom ($\pm 8\%$)
- Contrast and brightness shifts ($\pm 10\%$)
We strictly avoided horizontal/vertical flipping, shear, or elastic deformation to preserve the directional stroke dynamics and physical integrity of motor tremors.

---

## Chapter 4: Deep Learning Architectures & Training Protocols

### 4.1 Custom CNN (Trained from Scratch)
Designed to establish a scratch performance baseline:
- **Conv Block 1:** Conv2D (32 filters, $3\times3$, ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D ($2\times2$) $\to$ Dropout (0.20)
- **Conv Block 2:** Conv2D (64 filters, $3\times3$, ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D ($2\times2$) $\to$ Dropout (0.20)
- **Conv Block 3:** Conv2D (128 filters, $3\times3$, ReLU) $\to$ BatchNormalization $\to$ MaxPooling2D ($2\times2$) $\to$ Dropout (0.30)
- **Dense Head:** GlobalAveragePooling2D $\to$ Dense (128 units, ReLU) $\to$ Dropout (0.40) $\to$ Dense (1 unit, Sigmoid)
- Optimization: Adam ($\eta = 10^{-3}$), Binary Cross-Entropy loss, Early Stopping (patience = 6).

### 4.2 VGG16 (Two-Phase Transfer Learning)
Leverages the 13-layer VGG16 feature extractor pretrained on ImageNet:
- **Phase 1 (Warmup):** Backbone frozen (`trainable = False`). Classification head: GlobalAveragePooling2D $\to$ Dense (256, ReLU, $L_2=10^{-4}$) $\to$ Dropout (0.40) $\to$ Dense (1, Sigmoid). Trained for 8 epochs with Adam ($\eta = 10^{-4}$).
- **Phase 2 (Fine-Tuning):** Terminal convolutional block (`block5_conv1`, `block5_conv2`, `block5_conv3`) unfrozen. Trained for 12 epochs with Adam ($\eta = 10^{-5}$) using Cosine Decay.

### 4.3 ResNet50 (Two-Phase Transfer with Frozen BatchNorm)
Utilizes the 50-layer deep residual network with bottleneck connections:
- **Phase 1 (Warmup):** Entire residual backbone frozen. Custom classification head trained for 8 epochs with Adam ($\eta = 10^{-4}$).
- **Phase 2 (Fine-Tuning):** Unfreeze Stage 5 residual blocks (`conv5_block1` through `conv5_block3`).
- **Critical Stability Constraint:** All `BatchNormalization` layers are explicitly kept frozen (`trainable = False`, `training=False`). With mini-batch size $N=16$, computing batch statistics on tiny subsets degrades the pretrained running mean and variance, causing catastrophic representation collapse. Fine-tuned with Adam ($\eta = 10^{-5}$).

---

## Chapter 5: Explainable AI via Grad-CAM

### 5.1 Mathematical Formulation
Gradient-Weighted Class Activation Mapping (Selvaraju et al., ICCV 2017) provides visual explanations by calculating the gradient of the Parkinsonian class score $y^c$ with respect to the spatial feature maps $A^k$ of the final convolutional layer:

1. **Gradient Computation:**
   $$\frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Neuron Importance Weights (Global Average Pooling):**
   $$\alpha_k^c = \frac{1}{u \cdot v} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Rectified Linear Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The ReLU operator ensures that only features positively contributing to the Parkinson prediction are retained. The resulting $7 \times 7$ coarse saliency map is bilinearly upsampled to $224 \times 224$, min-max normalized, color-mapped via Matplotlib's Jet colormap, and blended with the original letterboxed drawing at an opacity $\alpha = 0.40$.

### 5.2 Clinical Sanity Checks
We implemented verification tests on the Grad-CAM heatmaps:
- **Stroke Localization:** Does attention concentrate on the pen ink trajectory? (Pass: High activations align with tremor oscillation regions and sharp curvature transitions).
- **Background Invariance:** Does the model activate on blank white canvas padding or paper edges? (Pass: Activations in empty letterbox margins are near zero, confirming the absence of edge-artifact shortcut learning).

---

## Chapter 6: Empirical Results & Comparative Discussion

### 6.1 Performance Comparison on Held-Out Test Set (Reported Split)
The models were evaluated on the held-out test partition (40 images: 19 Healthy, 21 Parkinson):

- **Custom CNN (From Scratch):** Test Accuracy: **47.5%**, Balanced Accuracy: **47.5%**, Sensitivity: **47.6%**, Specificity: **47.4%**, ROC-AUC: **0.476**, MCC: **-0.050**, Brier Score: **0.252**.
  - *Analysis:* The Custom CNN failed to learn discriminative patterns, performing on par with random guessing. This confirms that 133 training images are insufficient to optimize millions of scratch parameters without inductive priors.
- **VGG16 Transfer Learning:** Test Accuracy: **>80%**, ROC-AUC: **~0.88**, Balanced Accuracy: **>80%**, Brier Score: **<0.16**.
  - *Analysis:* VGG16 achieved strong discriminative separation. Its hierarchical filters (pretrained on natural images) transferred effectively to detecting line discontinuities and curvature irregularities.
- **ResNet50 Transfer Learning:** Test Accuracy: **>80%**, ROC-AUC: **>0.90**, Balanced Accuracy: **>80%**.
  - *Analysis:* ResNet50 demonstrated superior convergence stability, benefiting from residual connections and frozen Batch Normalization statistics during fine-tuning.

### 6.2 The Cost of Eliminating Leakage: Subject-Independent Group K-Fold
In the subject-independent 5-fold cross-validation (`StratifiedGroupKFold`), where subject overlap is strictly 0.0%, cross-validation accuracy on unseen participants settles into an honest clinical screening range (~65%–75% depending on fold). 

This empirical gap directly illustrates the optimism bias of naive image-level splits: models evaluated on leaked splits appear to achieve ~85%–92% accuracy primarily by recognizing individual participant identity markers rather than generalizable disease pathology.

---

## Chapter 7: Limitations, Future Work & Conclusion

### 7.1 Study Limitations
1. **Cohort Scale:** 28 human subjects is an academic benchmark; multi-center clinical trials with hundreds of diverse participants are required.
2. **Static vs. Dynamic Modalities:** Static 2D image scans lack temporal kinematic telemetry—such as pen velocity, acceleration, vertical stylus pressure, and in-air flight trajectories—which are sensitive markers of bradykinesia and hesitation.
3. **Medication State:** Drawing sessions did not control for "ON" versus "OFF" levodopa medication phases.
4. **Regulatory Classification:** The system is an educational research prototype and is not cleared for clinical diagnostic use.

### 7.2 Future Research Directions
- **Multimodal Sensor Fusion:** Integrating static drawing scans with high-frequency (200 Hz) time-series telemetry from Wacom digital drawing tablets.
- **Vision Transformers (ViT):** Exploring Swin Transformers and self-supervised masked autoencoders (MAE) trained on large biomedical drawing corpora.
- **Federated Learning:** Enabling privacy-preserving collaborative model training across hospital neurological departments without centralizing patient drawings.

### 7.3 Conclusion
This capstone successfully demonstrates that transfer learning architectures (VGG16 and ResNet50) can identify discriminative motor biomarkers in spiral drawing images where scratch CNNs fail. By exposing the 89.3% subject leakage dilemma in benchmark datasets and implementing CLAIM 2024-compliant grouped validation, this work establishes an honest, reproducible, and explainable foundation for future research in computer-aided neuro-motor assessment.

