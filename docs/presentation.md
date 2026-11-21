# Capstone Oral Presentation & Defense Deck
## Parkinson's Disease Classification Using Deep Learning and Explainable AI

**Project Title:** NeuroSpiral: Parkinson's Disease Classification from Spiral Drawings Using Deep Convolutional Neural Networks and Gradient-Weighted Class Activation Mapping  
**Duration:** 15–20 Minutes + 10 Minutes Q&A  
**Audience:** University Examination Committee, Capstone Faculty Panel, Biomedical AI Researchers  

---

## Slide 1: Title & Defense Overview
- **Headline:** NeuroSpiral: Parkinson's Disease Classification Using Deep Learning & Explainable AI
- **Subtitle:** An Empirical Investigation of Spiral Drawing Biomarkers, Subject-Level Data Leakage, and Grad-CAM Interpretability
- **Candidate:** AI Engineering Capstone Team
- **Key Badges:** CLAIM 2024 Compliant &bull; Zero-Leakage Grouped 5-Fold Cross-Validation &bull; Interactive Clinical Research Prototype
- **Speaker Notes:**
  > "Distinguished members of the examination committee, good morning. Today, we present NeuroSpiral, our capstone research project investigating the application of deep convolutional neural networks and explainable AI to classify Parkinson's disease from spiral and meander drawing examinations. We will walk through our clinical motivation, our programmatic discovery of an 89.3% subject leakage issue in historical benchmarks, our computer vision preprocessing pipeline, comparative model evaluations, Grad-CAM visual interpretability, and our production-ready interactive system."

---

## Slide 2: Clinical Context & The Parkinsonian Dilemma
- **Visuals:** Diagram of substantia nigra dopaminergic degeneration; MDS-UPDRS examination timeline.
- **Key Points:**
  - Parkinson's is the 2nd most common neurodegenerative disorder (10M+ individuals globally).
  - Cardinal motor symptoms: Resting tremor (4–6 Hz), muscular rigidity, bradykinesia, postural instability.
  - Clinical challenge: By symptom onset, 60–80% of substantia nigra dopamine neurons are lost. Clinical scoring (MDS-UPDRS) is intermittent, clinic-bound, and subjective.
- **Speaker Notes:**
  > "Parkinson's disease presents a significant clinical screening challenge. Current clinical evaluation relying on visual motor scoring is subjective and often occurs late in the disease progression. There is an urgent global need for objective, non-invasive, accessible screening tools that can detect subtle neuromotor deterioration."

---

## Slide 3: Why Spiral Drawings? Biomechanical Basis
- **Visuals:** Side-by-side comparison of Healthy spiral (smooth, concentric, uniform radius expansion) vs. Parkinsonian spiral (micro-tremors, hesitation kinks, radius compression).
- **Key Points:**
  - Concentric Archimedean spirals and sinusoidal meanders eliminate confounding linguistic and literacy variables.
  - Demands sustained fine-motor coordination across agonist and antagonist digital muscle groups.
  - Basal ganglia dysfunction manifests as tremor micro-deviations, bradykinetic hesitations, and micrographia.
- **Speaker Notes:**
  > "Why spiral drawings? Unlike writing sentences, which is confounded by language, education, and cognitive factors, drawing continuous Archimedean spirals isolates fine-motor neuromuscular control. As seen here, a healthy participant draws smooth, evenly spaced concentric loops. In Parkinson's, dopamine depletion disrupts velocity and muscle tone modulation, producing visible micro-oscillations, stroke kinks, and compressed loops."

---

## Slide 4: Dataset Ingestion & The Critical Leakage Finding
- **Visuals:** Dataset breakdown chart (204 images: 102 Healthy, 102 Parkinson); Subject distribution matrix showing subject overlap across train/val/test.
- **Key Points:**
  - HandPD & NewHandPD dataset: 204 total images.
  - Raw static split: 133 train, 31 val, 40 test.
  - **Critical Discovery:** Images stem from only 28 unique human participants.
  - **The Leakage:** 25 out of 28 participants (89.3%) have drawing samples scattered across both train and test sets.
- **Speaker Notes:**
  > "Before training any model, our team audited the raw dataset. We mapped the image filenames back to their clinical identifiers and made a critical discovery: the 204 images represent only 28 human subjects. In the static split commonly cited in prior literature, 89.3% of subjects have drawings in both training and test folders. This constitutes severe subject-level data leakage. If left uncorrected, a neural network learns to recognize the participant's pen thickness or scanner grain rather than clinical tremors."

---

## Slide 5: Dual Benchmark Protocol: CLAIM 2024 Alignment
- **Visuals:** Diagram contrasting Naive Image-Level Split vs. Subject-Independent StratifiedGroupKFold.
- **Key Points:**
  - Aligned with CLAIM 2024 (Item 11b) and TRIPOD+AI guidelines.
  - Benchmark A (Reported Static Split): Replicates historical literature for direct comparison.
  - Benchmark B (Subject-Independent Group 5-Fold CV): All drawings of a subject strictly confined to either train or test fold. Zero subject leakage.
- **Speaker Notes:**
  > "To ensure rigorous scientific validity, we instituted a dual-benchmark protocol. We evaluate on the original static split to understand historical claims, but establish our primary scientific benchmark using Stratified Group 5-Fold Cross-Validation grouped strictly on participant ID. This ensures zero data leakage, giving faculty and clinicians a genuine estimate of generalization on unseen patients."

---

## Slide 6: Computer Vision Preprocessing Pipeline
- **Visuals:** Step-by-step image transformation: Raw Rectangular Meander (512×250) $\to$ Aspect-Ratio Preserving Letterbox (224×224 with white background) $\to$ Model-Specific Normalization.
- **Key Points:**
  - Raw images have mixed aspect ratios (256×256 spirals and 512×250 meanders).
  - Standard resizing crushes meanders by >50%, destroying wave geometry.
  - Letterbox padding preserves true stroke geometry with white paper background (RGB 255, 255, 255).
  - Model-specific channel normalization: [0, 1] for Custom CNN; Caffe ImageNet BGR mean subtraction for VGG16/ResNet50.
- **Speaker Notes:**
  > "Our computer vision pipeline addresses an important physical reality: spirals are square, but meander drawings are wide rectangles. Direct resizing distorts stroke curvature and tremor frequency. We implemented aspect-ratio preserving letterboxing with white padding, standardizing images to 224x224 while maintaining exact stroke geometry."

---

## Slide 7: Deep Learning Architectures
- **Visuals:** Architecture diagrams of Custom CNN, VGG16, and ResNet50.
- **Key Points:**
  - **Custom CNN:** 3 convolutional blocks, batch norm, max pooling, dense head (trained from scratch).
  - **VGG16:** 13 convolutional layers, 2-phase transfer learning (warmup head $\to$ fine-tune Block 5 at $\eta = 10^{-5}$).
  - **ResNet50:** 50 layers with residual bottleneck connections, 2-phase transfer learning with frozen BatchNorm layers to prevent feature collapse.
- **Speaker Notes:**
  > "We compared three deep learning paradigms: a shallow Custom CNN trained from scratch, a deep VGG16 architecture, and a modern residual network, ResNet50. For the transfer learning models, we executed a two-phase training protocol: initial warmup of the classification head followed by delicate fine-tuning of terminal convolutional layers. Crucially, in ResNet50, we froze the Batch Normalization layers during fine-tuning to prevent moving-statistic degradation on our small mini-batches."

---

## Slide 8: Experimental Results: The Performance Truth
- **Visuals:** Metric comparison table (Accuracy, Sensitivity, Specificity, F1, Balanced Acc, ROC-AUC, MCC, Brier Score).
- **Key Points:**
  - Custom CNN (scratch): ~47.5% test accuracy. Fails to converge due to small dataset size (133 images).
  - VGG16 Transfer Learning: Outstanding feature adaptation (>80% accuracy, AUC ~0.88), learning stroke boundaries.
  - ResNet50: Strong feature extraction with residual stability.
  - Image-level vs. Grouped CV comparison: Highlights the empirical cost of eliminating subject leakage.
- **Speaker Notes:**
  > "Here are the empirical results. Notice first that the Custom CNN trained from scratch achieves only 47.5% accuracy. With only 133 training images, a scratch CNN simply lacks the inductive priors to learn generalized stroke representations. In contrast, VGG16 and ResNet50 transfer learning achieve strong discriminative power. This confirms our core hypothesis: transfer learning from natural image hierarchies is indispensable for small-scale biomedical drawing tasks."

---

## Slide 9: Multi-Metric Statistical Rigor
- **Visuals:** ROC curves and Precision-Recall curves plotted side-by-side.
- **Key Points:**
  - Clinical utility requires looking beyond accuracy: Sensitivity (detecting PD) vs. Specificity (clearing Healthy).
  - Balanced Accuracy and Matthews Correlation Coefficient (MCC) account for class balance.
  - Calibration: Brier score validates probability reliability.
  - Dynamic Decision Threshold ($\tau$): Lowering $\tau$ boosts Sensitivity for primary care screening.
- **Speaker Notes:**
  > "In biomedical engineering, accuracy alone is insufficient. Missing a Parkinsonian patient is far worse than a false positive. We report Balanced Accuracy, Sensitivity, Specificity, and Matthews Correlation Coefficient. Our ROC-AUC confirms robust discriminative separation across operating thresholds, and our Brier score validates that predicted probabilities reflect true statistical risk."

---

## Slide 10: Explainable AI: Grad-CAM Formulation
- **Visuals:** Mathematical formula slide for Grad-CAM; Feature map gradient flow diagram.
- **Key Points:**
  - Grad-CAM targets `block5_conv3` in VGG16 and `conv5_block3_out` in ResNet50.
  - Computes gradients of unnormalized Parkinson logit $y^c$ with respect to spatial feature maps $A^k$.
  - Global average pooling yields channel importance weights $\alpha_k^c$.
  - ReLU rectification filters out negative evidence, isolating features that positively drive the Parkinson prediction.
- **Speaker Notes:**
  > "Deep learning models are notoriously black-box. To satisfy clinical explainability requirements, we implemented Grad-CAM. By computing the gradient of the Parkinson class score with respect to the terminal convolutional feature maps and applying ReLU rectification, we generate 2D spatial saliency maps showing exactly which drawing features triggered the classification."

---

## Slide 11: Clinical Sanity Check: Interpreting Saliency Maps
- **Visuals:** Real Grad-CAM examples: (A) Correct localization on stroke hesitation points; (B) Audit of background space showing clean model focus.
- **Key Points:**
  - Clinical sanity check passes: Grad-CAM activations align directly with physical pen stroke anomalies, radius shifts, and tremor kinks.
  - Confirms the model is not relying on spurious edge artifacts or blank background noise.
- **Speaker Notes:**
  > "As demonstrated in our clinical sanity check, the model's high-activation regions directly overlay physical pen tremor points and irregular loop transitions. It does not activate on the white letterbox margins or paper boundaries, proving that our preprocessing prevented edge artifacts."

---

## Slide 12: Production Full-Stack Architecture
- **Visuals:** Full-stack system architecture flowchart (Client $\to$ REST API $\to$ Engine $\to$ Model Weights).
- **Key Points:**
  - **Backend:** FastAPI asynchronous microservice, Pydantic data validation, dynamic thresholding, in-memory Grad-CAM synthesis.
  - **Frontend:** React 18, TypeScript, Tailwind CSS, Vite, interactive Drag-and-Drop inference lab, threshold slider, real-time KPI dashboards.
  - **Performance:** CPU inference latency ~15–35 ms per image.
- **Speaker Notes:**
  > "We packaged our models into a complete, interactive software system. The backend is an asynchronous FastAPI service with strict Pydantic schemas, and the frontend is built with React, TypeScript, and Tailwind CSS. A clinician or examiner can drag-and-drop any spiral image, adjust the decision threshold in real time, inspect confidence meters, and view the Grad-CAM heatmap overlay within 30 milliseconds."

---

## Slide 13: Critical Limitations & Real-World Translation
- **Visuals:** Diagram comparing Static Scans vs. Dynamic Digitizer Tablets (Pressure, Speed, In-Air Trajectories).
- **Key Points:**
  - **Cohort Size:** 28 subjects is an academic benchmark; external multi-center validation is required.
  - **Static vs. Dynamic:** Scanned drawings lack temporal dynamics (pen velocity, acceleration, vertical pressure, in-air hovering).
  - **Regulatory Status:** Educational and research prototype; not approved for autonomous clinical diagnosis.
- **Speaker Notes:**
  > "A hallmark of rigorous scientific engineering is an honest appraisal of limitations. First, 28 subjects is an academic benchmark that cannot capture the full heterogeneity of Parkinson's across age, gender, and medication states. Second, static scanned images lose dynamic kinematic features—such as pen velocity, downward pressure, and in-air flight—which are known to be sensitive tremor indicators. We explicitly frame our system as an educational and screening research aid."

---

## Slide 14: Future Research Directions
- **Visuals:** Roadmap graphic showing Vision Transformers, Multimodal Tablet Integration, and Federated Learning.
- **Key Points:**
  - Multimodal fusion: Combining static spiral images with 1D dynamic time-series signals from Wacom digitizers.
  - Exploration of Vision Transformers (ViT) and Swin Transformers with self-supervised pretraining (DINO).
  - Privacy-preserving Federated Learning across clinical centers.
- **Speaker Notes:**
  > "Looking forward, the logical next step is multimodal fusion—combining computer vision features from drawing scans with temporal pressure trajectories from digital smart pens. Self-supervised vision transformers and federated learning protocols offer exciting pathways to scale to multi-hospital clinical trials."

---

## Slide 15: Conclusion & Summary of Contributions
- **Key Points:**
  1. Identified and documented an 89.3% subject leakage artifact in the HandPD benchmark.
  2. Implemented CLAIM 2024-compliant Stratified Group 5-Fold Cross-Validation.
  3. Engineered an aspect-ratio preserving letterbox pipeline with white canvas padding.
  4. Demonstrated that transfer learning (VGG16/ResNet50) overcomes sample size constraints where scratch CNNs fail.
  5. Implemented Grad-CAM explainability with clinical sanity validation.
  6. Delivered an interactive, full-stack research platform with complete academic documentation.
- **Speaker Notes:**
  > "In conclusion, our capstone delivers both an engineering accomplishment and an academic contribution: exposing dataset leakage, proving the indispensable role of transfer learning on small medical datasets, and delivering an explainable, end-to-end software system. Thank you for your time. We now welcome questions from the committee."

