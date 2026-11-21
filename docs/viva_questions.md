# Comprehensive Viva Voce & Technical Defense Preparation Guide
## Parkinson's Disease Classification Using Deep Learning & Explainable AI (Grad-CAM)

This document provides a rigorous, exhaustive preparation manual for capstone oral examinations, faculty defense committees, and external technical evaluators. It is organized into 6 core thematic domains containing 42 high-yield questions with detailed technical answers.

---

## Domain 1: Clinical Neuroscience & Biomechanical Rationale

### Q1: Why investigate spiral and meander drawings rather than standard cursive handwriting or text transcription?
**Answer:**
Standard handwriting involves high-level cognitive, linguistic, lexical, and orthographic processes that introduce significant confounding variance across subjects of different educational backgrounds, native languages, and literacy levels. In contrast, drawing continuous Archimedean spirals and sinusoidal meanders is a purely motor execution task. It requires sustained fine-motor coordination, precise agonist-antagonist digital muscle activation, and continuous velocity-radius modulation governed by the basal ganglia and corticospinal pathways. In Parkinson's disease (PD), degeneration of dopaminergic neurons in the substantia nigra pars compacta impairs automatic motor control, manifesting as micro-fluctuations, hesitation pauses, rigidity, and resting/action tremors (4–6 Hz) that are directly captured in the geometry and smoothness of the spiral loops.

### Q2: How does Parkinsonian tremor differ from Essential Tremor (ET), and can static drawings differentiate them?
**Answer:**
Parkinsonian tremor is predominantly a resting tremor (4–6 Hz) that attenuates during voluntary goal-directed action, although a significant subset of PD patients also exhibit re-emergent or postural tremor. Essential Tremor, by contrast, is primarily a kinetic and postural tremor (4–12 Hz) that intensifies during active movement (e.g., touching a target). In static pen-and-paper drawings, severe kinetic tremors in ET produce jagged, high-frequency oscillations throughout the trajectory, whereas PD drawings often exhibit a combination of micro-hesitations, loop compression (micrographia), directional stiffness, and uneven line thickness. However, dynamic temporal features (pen velocity, acceleration, in-air trajectories, and pressure profiles captured on digitizers) differentiate PD from ET more reliably than static scans alone.

### Q3: What is "micrographia," and how does it manifest in spiral drawing examinations?
**Answer:**
Micrographia is a classic clinical manifestation of Parkinsonian bradykinesia and hypokinesia, characterized by progressive reduction in writing or drawing size over sustained motor tasks. In spiral drawing tests, patients with micrographia often produce tight, constricted inner loops or fail to expand the radius smoothly as the spiral unwinds outward, resulting in an abnormally dense or prematurely terminated spiral structure compared to healthy controls.

### Q4: Why is early screening for Parkinson's disease valuable if the condition is neurodegenerative and incurable?
**Answer:**
Although there is currently no cure for Parkinson's disease, early diagnosis is clinically critical. Dopaminergic pharmacotherapy (such as levodopa/carbidopa) and physical therapy provide substantial symptomatic relief, improving quality of life, preserving motor autonomy, and reducing fall risks. Crucially, by the time classical cardinal motor symptoms appear, approximately 60–80% of dopaminergic neurons in the substantia nigra have already degenerated. Accessible non-invasive screening tools (such as digital pen or vision-based drawing tests) can alert clinicians to early motor subtle decline before severe motor disability develops.

---

## Domain 2: Dataset Integrity, Provenance & Data Leakage

### Q5: What is the provenance and composition of the dataset used in this project?
**Answer:**
The dataset derives from the Brazilian HandPD and NewHandPD clinical drawing databases (originally compiled by Pereira et al., 2016). It comprises 204 digitized drawing images: exactly 102 Healthy Control samples and 102 Parkinson's Disease patient samples. The samples encompass both concentric Archimedean spirals and horizontal sinusoidal meanders drawn by participants using dominant and non-dominant hands across multiple examination sessions.

### Q6: What is "subject-level data leakage," and how did you detect it in the original split?
**Answer:**
Subject-level data leakage occurs when multiple data points (images) originating from the same human participant are distributed across both the training and evaluation (validation/test) splits. 
Through a programmatic audit of the filename conventions (e.g., `V08HE03` indicates Subject `H_V08`, Examination `03`), we mapped all 204 images to exactly 28 unique human subjects (13 healthy, 15 Parkinsonian). In the original static dataset split (133 train, 31 val, 40 test), 25 of the 28 subjects (89.3%) had images appearing in both the training set and the test set simultaneously.

### Q7: Why is subject leakage catastrophic for real-world clinical validity?
**Answer:**
Deep convolutional neural networks possess massive parameter capacities (millions of weights) and will minimize training loss by learning any predictive correlation available in the pixels ("shortcut learning"). When images from the same subject are in both train and test partitions, the network memorizes individual-specific idiosyncrasies—such as unique pen stroke width, scanner noise patterns, paper texture, or drawing radius habits—rather than true pathological tremor features. This produces deceptively high test accuracy (often 90%+) that collapses when deployed on a new, previously unseen patient.

### Q8: How did you resolve the subject leakage problem in your research?
**Answer:**
We implemented two complementary evaluation protocols:
1. **Reported Split Benchmark:** We trained and evaluated on the original static split (133 train, 31 val, 40 test) to document the historical baseline and analyze model behavior under the conditions reported in early literature.
2. **Subject-Independent Benchmark:** We implemented `StratifiedGroupKFold(n_splits=5)` where the grouping parameter is strictly bound to `subject_id`. In each cross-validation fold, all drawings from a given subject are confined exclusively to either the training fold or the testing fold, ensuring 0.0% subject overlap and providing an unbiased, realistic measure of out-of-sample generalization in compliance with CLAIM 2024 (Item 11b) and TRIPOD+AI guidelines.

---

## Domain 3: Computer Vision & Preprocessing Engineering

### Q9: Why did you use aspect-ratio preserving letterboxing instead of standard bilinear resizing?
**Answer:**
The raw dataset contains heterogeneous dimensions: approximately half the images are square spirals (256×256 pixels), while the other half are wide rectangular meanders (e.g., 512×250 pixels). Standard resizing forcibly stretches or squashes the image to 224×224 pixels. For a 512×250 meander, horizontal compression exceeds 50%, artificially altering the geometric angle of strokes, wave frequencies, and curvature radii. Our letterbox preprocessing computes the uniform scale factor $s = \min(224/w, 224/h)$, resizes the image isotropically, and pads the margins with pure white pixels (RGB 255, 255, 255). This preserves the precise spatial trajectory and anatomical curvature of the pen strokes while standardizing the tensor shape for CNN ingestion.

### Q10: Why pad with white pixels (255, 255, 255) rather than zero/black pixels?
**Answer:**
In standard computer vision (e.g., object detection with YOLO), zero/black padding is common because images often have dark backgrounds or zeros are masked. In clinical drawing scans, however, the drawing paper background is white (pixel values ~240–255), while pen strokes are dark (values ~0–80). Padding with black would introduce four harsh artificial high-contrast borders around the canvas. These artificial edges produce massive gradient spikes in the first convolutional layer, distracting attention from the subtle stroke tremors inside the drawing. White padding seamlessly merges with the paper canvas.

### Q11: Why are different normalization strategies required for Custom CNN versus VGG16/ResNet50?
**Answer:**
- **Custom CNN:** Trained from scratch using random initialization (Glorot uniform). Inputs are scaled linearly to $[0, 1]$ via division by 255.0 to maintain well-behaved activation magnitudes and prevent vanishing/exploding gradients.
- **VGG16 and ResNet50:** Pretrained on ImageNet using Caffe-style preprocessing. The input RGB channels are converted to BGR, and the ImageNet dataset channel means ($[103.939, 116.779, 123.68]$) are subtracted without division by 255. Violating this pretrained input distribution by feeding $[0, 1]$ images causes severe domain misalignment and impairs the early convolutional feature extractors.

### Q12: What data augmentations were applied, and why were aggressive transforms avoided?
**Answer:**
We applied conservative, domain-safe augmentations to the training set only:
- Small random rotations ($\pm 10^\circ$)
- Minor random zoom ($\pm 8\%$)
- Subtle contrast/brightness adjustments ($\pm 10\%$)
We deliberately **avoided** aggressive augmentations such as horizontal/vertical flips, elastic deformations, cut-out, or heavy shear. Parkinsonian motor impairment has spatial and directional characteristics (e.g., hand tremor frequency in specific axes, micrographia loop direction); artificial geometric warping could distort clinical tremor signals into artifacts or blur the boundaries between classes.

---

## Domain 4: Deep Learning Architectures & Transfer Learning

### Q13: Detail the architecture of your Custom CNN trained from scratch.
**Answer:**
The Custom CNN comprises:
1. **Input Layer:** $224 \times 224 \times 3$ normalized tensor.
2. **Conv Block 1:** Conv2D (32 filters, $3\times3$, ReLU), BatchNormalization, MaxPooling2D ($2\times2$). Output: $112 \times 112 \times 32$.
3. **Conv Block 2:** Conv2D (64 filters, $3\times3$, ReLU), BatchNormalization, MaxPooling2D ($2\times2$). Output: $56 \times 56 \times 64$.
4. **Conv Block 3:** Conv2D (128 filters, $3\times3$, ReLU), BatchNormalization, MaxPooling2D ($2\times2$). Output: $28 \times 28 \times 128$.
5. **Dense Head:** GlobalAveragePooling2D, Dense (128 units, ReLU), Dropout ($p=0.40$), Dense (1 unit, Sigmoid activation).
Total trainable parameters: ~104,000 to ~2,300,000 depending on pooling strategy.

### Q14: Why did Custom CNN perform poorly (~47.5% test accuracy) compared to transfer learning?
**Answer:**
Training a deep convolutional network from scratch requires thousands of representative images to learn fundamental low-level features (Gabor filters, edge gradients, corners, textures). Our training set contained only 133 images. With a high parameter-to-sample ratio, the Custom CNN has insufficient data to optimize millions of weights from random initialization, leading to empirical underfitting, gradient instability, and failure to discover invariant representations. This empirically confirms the foundational thesis: *shallow or scratch CNNs are inadequate for small medical datasets without transfer learning or heavy inductive priors*.

### Q15: Explain the two-phase transfer learning methodology used for VGG16.
**Answer:**
1. **Phase 1 (Feature Extraction / Warmup):**
   - The entire convolutional base of VGG16 (`block1_conv1` through `block5_pool`) is frozen (`layer.trainable = False`), retaining ImageNet weights.
   - A custom classification head is appended: GlobalAveragePooling2D $\to$ Dense(256, ReLU) $\to$ Dropout(0.40) $\to$ Dense(1, Sigmoid).
   - Trained for 8 epochs using Adam optimizer with moderate learning rate ($\eta = 10^{-4}$). This allows the randomly initialized head weights to converge toward a stable loss landscape without destroying the pretrained convolutional representations.
2. **Phase 2 (Fine-Tuning):**
   - The top convolutional block (`block5_conv1`, `block5_conv2`, `block5_conv3`) is unfrozen, while blocks 1–4 remain frozen.
   - Trained for 12 epochs with a very small learning rate ($\eta = 10^{-5}$) using CosineDecay schedule. This fine-tunes high-level visual patterns (stroke curvature, line density) to drawing specifics while preserving low-level edge detectors.

### Q16: Why must BatchNormalization layers remain frozen during ResNet50 fine-tuning?
**Answer:**
ResNet50 contains multiple BatchNormalization (BN) layers within its residual bottleneck units. During training, BN calculates mini-batch mean and variance to normalize activations and updates its internal moving averages. With small batch sizes (e.g., $N=16$), the mini-batch sample statistics exhibit high stochastic noise. If BN layers are set to `trainable=True` during fine-tuning on a small dataset, their moving statistics degrade rapidly, destabilizing the entire residual stream and destroying the pretrained representations. Therefore, during Phase 2 fine-tuning of ResNet50, we set `layer.trainable = True` only for Conv2D layers in stage 5, while explicitly freezing all BN layers (`layer.trainable = False` or `training=False` in calling convention).

---

## Domain 5: Explainable AI & Grad-CAM Mathematical Formulation

### Q17: Derive the mathematical formulation of Grad-CAM step-by-step.
**Answer:**
1. **Compute Target Gradients:**
   Let $y^c$ be the raw unnormalized score (logit) for class $c$ (Parkinson), and let $A_{i,j}^k$ represent the activation at spatial position $(i, j)$ of feature map channel $k$ in the target convolutional layer. The gradient is computed via automatic differentiation:
   $$\frac{\partial y^c}{\partial A^k_{i,j}}$$
2. **Spatial Average Pooling (Neuron Importance Weights):**
   To quantify the importance of feature map $k$ for class $c$, we compute the global average pooling of the gradients over spatial dimensions of width $u$ and height $v$:
   $$\alpha_k^c = \frac{1}{u \cdot v} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A^k_{i,j}}$$
3. **Linear Combination and Rectification:**
   The class-discriminative saliency map $L_{\text{Grad-CAM}}^c$ is computed by taking a linear combination of all feature maps weighted by $\alpha_k^c$, followed by a Rectified Linear Unit (ReLU):
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
   The ReLU is vital because we are interested only in features that positively contribute to class $c$; negative gradients correspond to pixels that support other classes.

### Q18: Which layer was selected as the Grad-CAM target in VGG16 and ResNet50, and why?
**Answer:**
- In **VGG16**, we target `block5_conv3` (the final convolutional layer before the pooling layer).
- In **ResNet50**, we target `conv5_block3_out` (the final residual addition output before global average pooling).
**Rationale:** Lower convolutional layers capture generic low-level primitives (edges, color, texture) with high spatial resolution but minimal semantic class-specificity. In contrast, the final convolutional layer has the richest semantic abstraction and receptive field covering the entire input image, while still retaining sufficient 2D spatial arrangement ($7 \times 7$) to localize where the decision was triggered.

### Q19: What constitutes a "clinical sanity check" for Grad-CAM in this application?
**Answer:**
A clinical sanity check verifies whether the model is focusing on biologically plausible features:
- **Valid Attention:** Heatmap intensity concentrates along the pen stroke boundaries, specifically at points of tremor hesitation, jagged loop transitions, or abnormal stroke compression.
- **Spurious Attention / Shortcut Learning:** Heatmap intensity concentrates on the empty white margins, corner pixels, or scanner artifacts. If Grad-CAM activates heavily on blank space, the model has learned an artifactual shortcut rather than drawing pathology.

---

## Domain 6: Statistical Evaluation & Comprehensive Metrics

### Q20: Why is Accuracy an insufficient metric for biomedical machine learning?
**Answer:**
Accuracy weights false positives and false negatives equally and can be misleading in imbalanced or clinically sensitive scenarios. In clinical screening, missing a diseased patient (False Negative) has far more severe consequences than misidentifying a healthy individual for follow-up testing (False Positive). Therefore, biomedical evaluations must report:
- **Sensitivity / Recall:** $\frac{TP}{TP + FN}$ (ability to identify true Parkinson cases).
- **Specificity:** $\frac{TN}{TN + FP}$ (ability to identify healthy controls).
- **Balanced Accuracy:** $\frac{\text{Sensitivity} + \text{Specificity}}{2}$.
- **Matthews Correlation Coefficient (MCC):** Evaluates all four confusion matrix quadrants, providing a balanced metric even under extreme class skew.
- **ROC-AUC & PR-AUC:** Measures discrimination ability across all possible operating thresholds independently of a fixed 0.50 cutoff.

### Q21: What is the Brier Score, and why does calibration matter?
**Answer:**
The Brier Score is the mean squared difference between predicted class probabilities and the actual binary outcomes:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
A score of 0.0 indicates perfect calibration and discrimination, while 0.25 represents an uninformative 50/50 guessing model. In clinical decision support, calibrated probabilities are essential: a doctor needs to know if a model predicting $p=0.92$ is truly twice as confident as when predicting $p=0.55$. Uncalibrated deep neural networks often suffer from extreme overconfidence.

### Q22: What role does the decision threshold $\tau$ play in practical deployment?
**Answer:**
The default decision threshold is $\tau = 0.50$. However, in a primary care screening context where the objective is to catch as many potential cases as possible for secondary clinical workup, $\tau$ can be tuned downward (e.g., $\tau = 0.35$). This increases Sensitivity at the expense of Specificity. Our interactive web UI includes an interactive threshold slider so clinicians and examiners can explore the sensitivity-specificity trade-off in real time.

---

## Summary of 20 Additional Quick-Fire Examination Questions

| # | Question | Concise Technical Answer |
|---|---|---|
| 23 | What loss function was optimized? | Binary Cross-Entropy: $\mathcal{L} = -[y \log p + (1-y) \log(1-p)]$. |
| 24 | What optimizer and weight decay were used? | Adam optimizer ($\beta_1=0.9, \beta_2=0.999$) with $L_2$ regularization on dense heads. |
| 25 | What was the learning rate schedule? | Cosine annealing schedule with initial warmup for transfer learning. |
| 26 | What is the total parameter count of VGG16? | ~14.7 million convolutional parameters + ~131,000 dense classification parameters. |
| 27 | What is the total parameter count of ResNet50? | ~23.5 million convolutional parameters + ~524,000 dense classification parameters. |
| 28 | How long was inference latency per image? | ~15–35 ms on CPU, well within interactive sub-second threshold for clinical UI. |
| 29 | What is the input tensor shape across all models? | $(1, 224, 224, 3)$ representing batch size, height, width, and color channels. |
| 30 | Did you test Grayscale conversion? | Yes; while drawings are grayscale, RGB channels are preserved to utilize 3-channel ImageNet pretrained weights. |
| 31 | What is the role of Global Average Pooling (GAP)? | Replaces flatten layers; collapses $7\times7\times C$ to $1\times C$, drastically reducing parameter count and preventing dense overfitting. |
| 32 | What is Dropout, and what rate was applied? | Randomly zeroes activations during training ($p=0.40$) to prevent co-adaptation of neurons. |
| 33 | What is Early Stopping patience? | Monitored validation loss with patience of 5 epochs to prevent overtraining. |
| 34 | What is ModelCheckpoint callback? | Saves only the weights producing the minimum validation loss during training. |
| 35 | How was the backend architected? | Python FastAPI asynchronous REST API with Pydantic type validation and CORS middleware. |
| 36 | How was the frontend built? | React 18, TypeScript, Tailwind CSS, Lucide icons, and Vite bundling. |
| 37 | How are Grad-CAM heatmaps returned to the client? | Rendered with Matplotlib Jet colormap, converted to PNG in-memory, and encoded as Base64 data URIs. |
| 38 | What is the CLAIM guideline? | Checklist for Artificial Intelligence in Medical Imaging, published in Radiology: AI 2024. |
| 39 | What is TRIPOD+AI? | Transparent Reporting of a multivariable prediction model of Individual Prognosis Or Diagnosis + AI (BMJ 2024). |
| 40 | Why is the system labeled an educational prototype? | Non-cleared regulatory status; small sample size (28 subjects); lack of external clinical validation. |
| 41 | Can this system run on consumer hardware? | Yes; all three trained models have lightweight CPU footprints ($<100$ MB each) and execute within 40 ms. |
| 42 | What is the single most important future improvement? | Transitioning from static scanned images to dynamic high-frequency digitizer tablets collecting pen pressure, velocity, and in-air movements. |

