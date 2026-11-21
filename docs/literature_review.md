# Literature Review: Computer Vision and Explainable AI for Parkinson's Disease Screening

## 1. Introduction and Clinical Context
Parkinson's Disease (PD) is the second most prevalent progressive neurodegenerative disorder worldwide, characterized pathologically by the progressive loss of dopaminergic neurons in the substantia nigra pars compacta. The cardinal motor manifestations encompass resting tremor, cogwheel rigidity, bradykinesia, and postural instability. Historically, clinical diagnosis has relied on expert neurological assessment utilizing clinical rating scales such as the Unified Parkinson's Disease Rating Scale (MDS-UPDRS) and Hoehn and Yahr staging. However, clinical diagnosis in early stages exhibits a misdiagnosis rate of 15% to 25% when conducted by general practitioners, underscoring the pressing need for objective, accessible, non-invasive digital screening modalities.

## 2. Drawing Kinematics, Motor Control, and Biomarkers
Handwriting and graphic drawing tasks (specifically Archimedean spirals and sinusoidal meanders) engage complex visuomotor loops involving corticobasal-ganglia-thalamocortical circuits. Motor impairments in Parkinson's disease disrupt fine motor coordination, resulting in:
1. **Micrographia and Tremor**: High-frequency involuntary oscillations (4–6 Hz rest tremor, 6–9 Hz postural/action tremor) superimposed onto intentional pen strokes.
2. **Kinematic Degradation**: Irregular velocity profiles, sudden pauses, lack of stroke fluency, and variable pen pressure.
3. **Geometric Distortion**: Inability to maintain smooth radius dilation in Archimedean spirals ($r = a + b\theta$) and irregular oscillatory wave amplitudes in meanders.

Visual representations of these drawing tasks capture both spatial tremor oscillations and geometric distortion, making them viable inputs for computer vision architectures.

## 3. Dataset Origins: HandPD, NewHandPD, and Roboflow Provenance
Our programmatic investigation of the 204 images in the capstone dataset uncovered file prefixes such as `V08HE03`, `V12PE01`, and `V07PO02`. This naming taxonomy corresponds directly to the benchmark HandPD and NewHandPD database conventions established by Pereira et al. (São Paulo State University, UNESP):
- **Subject Code**: `Vxx` (Volunteer ID)
- **Cohort Code**: `H` (Healthy Control) vs. `P` (Parkinson's Patient)
- **Task Code**: `E` (*Espiral* - Archimedean Spiral) vs. `O` (*Outro/Meandro* - Meander wave)
- **Trial Code**: `01`, `02`, `03` (Sequential trial recordings per subject)

While the dataset was exported via Roboflow Universe as a 2-class classification dataset, tracing its true provenance confirms that it originates from digitalized pen-and-paper assessments collected from 28 individuals (13 Healthy volunteers, 15 Parkinson's patients).

## 4. Deep Learning Architectures in Medical Image Analysis
Deep Convolutional Neural Networks (CNNs) have emerged as the dominant paradigm for feature extraction from graphic drawing representations:
- **Custom Small-Scale CNNs**: A shallow convolutional topology (e.g., 3 convolutional blocks with Batch Normalization and Dropout) acts as a baseline with lower parametric capacity, designed to prevent catastrophic memorization on small image sets ($N \approx 200$).
- **VGG16 Transfer Learning**: Introduced by Simonyan & Zisserman (2014), VGG16 utilizes homogeneous $3 \times 3$ convolutional filters arranged in deep sequential blocks. Pretrained on ImageNet ($1.2 \times 10^6$ natural images), early convolutional filters capture low-level edge detectors, stroke curvatures, and high-frequency textures that readily transfer to line drawing artifacts.
- **ResNet50 Transfer Learning**: Introduced by He et al. (2015), ResNet50 addresses vanishing gradients via residual skip connections ($y = \mathcal{F}(x) + x$). In transfer learning regimes with limited sample sizes, freezing early residual blocks and Batch Normalization layers preserves generalized representation spaces while fine-tuning terminal stages.

## 5. Methodological Pitfalls: The Data Leakage Crisis in Medical AI
A paramount finding in recent critical surveys (such as the 2026 *Scientific Reports* review on handwriting-based Parkinson's detection, CLAIM 2024, and TRIPOD+AI) is the widespread prevalence of **Subject-Level Data Leakage**:
- **The Problem**: When multiple drawing trials from the same volunteer ($V01, V02, \dots$) are randomly partitioned into training, validation, and test splits at the image level, deep neural networks achieve artificially inflated classification accuracies (>95%).
- **Mechanism of Failure**: The network does not learn generalizable Parkinsonian tremor biomarkers; instead, it memorizes participant-specific idiosyncrasies—such as unique ink pressure, paper grain, background scanner brightness, or distinctive stroke idiosyncrasies.
- **Scientific Solution**: Implementing strict **Subject-Independent Grouped Evaluation** (e.g., Stratified Group K-Fold), ensuring zero patient overlap across training and testing partitions.

## 6. Explainable AI (XAI) and Grad-CAM in Healthcare
Healthcare professionals rightfully reject "black-box" predictions. Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017) provides visual explanations by computing gradients of the predicted class score with respect to feature maps of the terminal convolutional layer:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \text{where} \quad \alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
Applying Grad-CAM to spiral drawings reveals whether the model attends to genuine tremor micro-deviations along the spiral trajectory or to irrelevant peripheral artifacts (white borders, paper corners), establishing a critical sanity check for clinical trustworthiness.

