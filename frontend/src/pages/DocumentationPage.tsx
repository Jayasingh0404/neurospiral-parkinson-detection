import React, { useState } from 'react';
import { 
  FileText, 
  HelpCircle, 
  Layers, 
  Database, 
  Code, 
  ChevronDown, 
  ChevronUp, 
  Copy, 
  Check, 
  Download,
  BookOpen,
  Sparkles
} from 'lucide-react';

export const DocumentationPage: React.FC = () => {
  const [activeDocTab, setActiveDocTab] = useState<'viva' | 'api' | 'thesis'>('viva');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(0);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const vivaQuestions = [
    {
      q: 'Why did you select spiral drawing as a biomarker for Parkinson\'s disease rather than general handwriting?',
      a: 'Spiral drawing (specifically Archimedean spirals) removes the confounding linguistic, orthographic, and cognitive variables inherent in cursive handwriting or sentence copying. Drawing continuous concentric loops demands sustained fine-motor coordination, precise agonist-antagonist muscle control, and steady velocity. In Parkinson\'s disease, basal ganglia dopamine depletion manifests as resting tremor (4-6 Hz), action tremor, rigidity, and bradykinesia, producing micro-fluctuations, stroke hesitations, and uneven pen radius progression that are visually salient in spiral drawings.'
    },
    {
      q: 'What is subject-level data leakage, and why does it invalidate naive computer vision models trained on this dataset?',
      a: 'The HandPD dataset consists of 204 images collected from 28 participants (13 healthy, 15 Parkinsonian), with each participant executing multiple examinations (exams 01 to 04 with both dominant and non-dominant hands). If images are shuffled randomly into train and test sets, images from the exact same human subject appear in both splits. Deep neural networks rapidly learn subject-specific identity signatures (such as pen thickness, digitizer paper grain, or stroke velocity habits) rather than true pathological tremors. In our audit, 89.3% of subjects in the static split were contaminated across splits. We resolved this by implementing StratifiedGroupKFold cross-validation grouped strictly on subject_id.'
    },
    {
      q: 'Why did Custom CNN trained from scratch underperform compared to pretrained VGG16 and ResNet50?',
      a: 'Our dataset contains only 133 training images in the static split (and ~163 per fold in 5-fold CV). Deep convolutional networks have millions of parameters (Custom CNN has ~2.3 million) and lack inductive prior knowledge. Training a CNN from scratch on tiny biomedical sample sizes leads to severe empirical overfitting and poor optimization convergence. In contrast, VGG16 and ResNet50 possess rich visual hierarchies (Gabor-like edge detectors, texture filters, curvature detectors) pre-trained on ImageNet\'s 1.2 million natural images, which transfer effectively to stroke boundary analysis when fine-tuned at conservative learning rates.'
    },
    {
      q: 'Why did you use aspect-ratio preserving letterboxing with a white background instead of simple isotropic resizing?',
      a: 'The raw dataset contains both square spiral drawings (256x256) and wide rectangular meander drawings (approx. 512x250). Forcing a 512x250 image directly into a 224x224 square via standard interpolation physically compresses the horizontal strokes by more than 50%, artificially altering loop frequencies, curvature angles, and tremor frequencies. Letterbox padding computes the uniform scaling factor, preserves true geometric stroke aspect ratio, and pads the remaining canvas with white pixels (RGB 255, 255, 255) to seamlessly blend with the paper drawing background.'
    },
    {
      q: 'Explain the mathematical formulation and mechanism of Grad-CAM in this project.',
      a: 'Grad-CAM (Gradient-weighted Class Activation Mapping) calculates the gradients of the unnormalized class score y^c with respect to the feature map activations A^k of the final convolutional layer (block5_conv3 in VGG16, or conv5_block3_out in ResNet50). These gradients are globally average pooled across spatial dimensions (u, v) to compute neuron importance weights alpha_k^c. We then compute a weighted sum of the feature maps followed by a ReLU activation to retain only features that positively correlate with the target class. The resulting 7x7 coarse saliency map is upsampled to 224x224 and rendered as a Jet colormap overlay.'
    },
    {
      q: 'Why did you choose a two-phase transfer learning regimen with frozen BatchNorm for ResNet50?',
      a: 'ResNet50 contains BatchNormalization layers throughout its residual blocks. In small medical datasets with tiny batch sizes (batch size = 16), calculating batch mean and variance on mini-batches during fine-tuning destroys the pretrained running statistics, causing catastrophic feature degradation. In Phase 1 (Warmup), the entire ResNet base is frozen, training only the classification head with Adam (lr = 1e-4). In Phase 2 (Fine-tuning), top residual stages are unfrozen, but BatchNormalization layers remain explicitly in inference mode (training=False), trained with a very low learning rate (lr = 1e-5).'
    },
    {
      q: 'What are the ethical and clinical limitations of using this model in real-world healthcare?',
      a: 'This system is strictly an academic research prototype and must never be utilized for autonomous clinical diagnosis. Limitations include: (1) Small sample size (28 subjects) lacking diverse demographic, age-matched, and medication-phase (ON/OFF levodopa) controls; (2) Static images lose dynamic kinematic features (pen pressure over time, azimuth angle, in-air trajectory, and instantaneous velocity) captured by digitizing tablets; (3) Absence of external prospective clinical validation across independent hospital centers.'
    }
  ];

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-sky-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <FileText className="w-4 h-4" /> Comprehensive Capstone Documentation
        </div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Academic Thesis & Defense Repository</h1>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          Complete documentation repository supporting university evaluation, technical viva voce examinations, and REST API integration.
        </p>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveDocTab('viva')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeDocTab === 'viva'
              ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <HelpCircle className="w-4 h-4 text-sky-400" /> Viva Voce Defense Mentor (Q&A)
        </button>
        <button
          onClick={() => setActiveDocTab('thesis')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeDocTab === 'thesis'
              ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <BookOpen className="w-4 h-4 text-indigo-400" /> Capstone Thesis Synopsis
        </button>
        <button
          onClick={() => setActiveDocTab('api')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeDocTab === 'api'
              ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Code className="w-4 h-4 text-emerald-400" /> REST API Architecture
        </button>
      </div>

      {/* Tab Content: Viva Questions */}
      {activeDocTab === 'viva' && (
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
            <span className="font-semibold text-white">Oral Defense Preparation Guide:</span> This section contains high-yield, probing technical questions commonly asked by academic external examiners, senior faculty, and defense committees. Review these structured answers to master the theoretical and practical nuances of the project.
          </div>

          <div className="space-y-3">
            {vivaQuestions.map((item, idx) => {
              const isOpen = expandedFaq === idx;
              return (
                <div 
                  key={idx} 
                  className="card border-slate-800 bg-slate-900/40 overflow-hidden transition-all duration-200"
                >
                  <button
                    onClick={() => setExpandedFaq(isOpen ? null : idx)}
                    className="w-full text-left p-4 flex items-start justify-between gap-4 group"
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30 shrink-0 mt-0.5">
                        Q{idx + 1}
                      </span>
                      <h3 className="font-semibold text-white text-sm group-hover:text-sky-300 transition-colors">
                        {item.q}
                      </h3>
                    </div>
                    <div className="text-slate-400 group-hover:text-white transition-colors shrink-0">
                      {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </div>
                  </button>

                  {isOpen && (
                    <div className="px-4 pb-4 pt-1 border-t border-slate-800/60 text-xs text-slate-300 leading-relaxed bg-slate-950/40 space-y-3">
                      <p>{item.a}</p>
                      <div className="flex justify-end">
                        <button
                          onClick={() => handleCopy(item.a, idx)}
                          className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors text-[11px]"
                        >
                          {copiedIndex === idx ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                          {copiedIndex === idx ? 'Copied' : 'Copy Answer'}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Tab Content: Thesis Synopsis */}
      {activeDocTab === 'thesis' && (
        <div className="space-y-6">
          <div className="card border-slate-800 bg-slate-900/40 space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-400" /> Executive Thesis Synopsis
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Parkinson's Disease (PD) is the second most prevalent neurodegenerative movement disorder globally, characterized by progressive loss of dopaminergic neurons in the substantia nigra pars compacta. 
              While motor symptoms including resting tremor, bradykinesia, and muscular rigidity are clinically assessed via the Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS), these evaluations are subjective and intermittent.
            </p>
            <p className="text-xs text-slate-300 leading-relaxed">
              This capstone investigates deep learning computer vision architectures applied to digital spiral drawing examinations from the HandPD benchmark dataset. 
              Crucially, our investigation identifies an 89.3% subject-level data leakage artifact in the original dataset split, where drawings from identical participants are scattered across train and test sets. 
              We contrast naive image-level splitting with rigorous Stratified Group 5-Fold Cross-Validation, establishing an honest, transparent performance baseline aligned with CLAIM 2024 standards.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-[10px] font-mono text-sky-400 uppercase font-bold">Chapter 1-2</span>
              <h4 className="font-semibold text-white">Introduction & Clinical Motivation</h4>
              <p className="text-slate-400 leading-relaxed">Basal ganglia circuitry, resting tremor biomechanics, and spiral drawing task rationale.</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-[10px] font-mono text-indigo-400 uppercase font-bold">Chapter 3-4</span>
              <h4 className="font-semibold text-white">Methodology & Architectures</h4>
              <p className="text-slate-400 leading-relaxed">Aspect-ratio preserving letterbox preprocessing, Custom CNN, VGG16, and ResNet50 transfer learning.</p>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <span className="text-[10px] font-mono text-emerald-400 uppercase font-bold">Chapter 5-7</span>
              <h4 className="font-semibold text-white">Evaluation, XAI & Discussion</h4>
              <p className="text-slate-400 leading-relaxed">Comparative performance metrics, Grad-CAM saliency heatmaps, leakage audit, and clinical conclusions.</p>
            </div>
          </div>
        </div>
      )}

      {/* Tab Content: REST API Architecture */}
      {activeDocTab === 'api' && (
        <div className="card border-slate-800 bg-slate-900/40 space-y-6">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Code className="w-5 h-5 text-emerald-400" /> FastAPI Microservice Specification
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict Pydantic-validated endpoints for real-time model inference and Grad-CAM synthesis
            </p>
          </div>

          <div className="space-y-4">
            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
              <div className="flex items-center gap-3">
                <span className="badge badge-emerald font-mono">POST</span>
                <code className="text-xs font-mono text-sky-300">/api/v1/predict</code>
              </div>
              <p className="text-xs text-slate-400">Classifies a spiral drawing image with specified architecture and decision threshold.</p>
              <div className="bg-slate-900 p-2.5 rounded text-[11px] font-mono text-slate-300">
                FormData: <code className="text-amber-300">file</code> (image/png|jpeg), <code className="text-amber-300">model_name</code> ("custom_cnn" | "vgg16" | "resnet50"), <code className="text-amber-300">threshold</code> (float: 0.0-1.0)
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
              <div className="flex items-center gap-3">
                <span className="badge badge-indigo font-mono">POST</span>
                <code className="text-xs font-mono text-sky-300">/api/v1/explain</code>
              </div>
              <p className="text-xs text-slate-400">Generates 2D Grad-CAM saliency heatmaps and blended overlays for local interpretability.</p>
              <div className="bg-slate-900 p-2.5 rounded text-[11px] font-mono text-slate-300">
                FormData: <code className="text-amber-300">file</code> (image/png|jpeg), <code className="text-amber-300">model_name</code> ("vgg16" | "resnet50" | "custom_cnn")
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
              <div className="flex items-center gap-3">
                <span className="badge badge-sky font-mono">GET</span>
                <code className="text-xs font-mono text-sky-300">/api/v1/metrics</code>
              </div>
              <p className="text-xs text-slate-400">Returns comprehensive evaluation metrics and cross-validation summaries across all models.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

