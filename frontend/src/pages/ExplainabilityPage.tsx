import React, { useState } from 'react';
import { 
  Eye, 
  HelpCircle, 
  Layers, 
  CheckCircle2, 
  AlertTriangle, 
  Brain, 
  Sliders, 
  ExternalLink,
  Sparkles
} from 'lucide-react';

export const ExplainabilityPage: React.FC = () => {
  const [activeLayer, setActiveLayer] = useState<'custom_cnn' | 'vgg16' | 'resnet50'>('vgg16');

  return (
    <div className="space-y-12 max-w-6xl mx-auto">
      {/* Title */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-xs text-purple-400 font-mono">
          <Eye className="w-3.5 h-3.5" />
          <span>Explainable Artificial Intelligence (XAI)</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Grad-CAM Saliency & Visual Interpretability
        </h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto">
          Demystifying deep vision models by calculating class-discriminative gradient activation maps 
          over terminal convolutional feature representations.
        </p>
      </div>

      {/* Conceptual Foundation */}
      <div className="glass-panel p-8 rounded-3xl space-y-6">
        <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
          <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">How Grad-CAM Works on Spiral Drawings</h2>
            <p className="text-xs text-slate-400">Mathematical formulation and gradient flow</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-xs text-slate-300 leading-relaxed">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-[10px] font-mono">1</span>
              Gradient Backpropagation to Conv Maps
            </h3>
            <p>
              Given an input drawing, the model computes the unnormalized class score y^c (where c = Parkinson). 
              We calculate the gradient of $y^c$ with respect to the feature map activations $A^k$ of the chosen convolutional stage:
            </p>
            <div className="p-3 bg-slate-900 rounded-xl font-mono text-[11px] text-purple-300 border border-slate-800 text-center">
              {"\\alpha_k^c = \\frac{1}{Z} \\sum_{i=1}^u \\sum_{j=1}^v \\frac{\\partial y^c}{\\partial A_{i,j}^k}"}
            </div>
            <p className="text-slate-400 text-[11px]">
              Global average pooling collapses the gradient spatial dimensions into importance weights $\alpha_k^c$ for each channel $k$.
            </p>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-[10px] font-mono">2</span>
              Rectified Weighted Combination
            </h3>
            <p>
              We compute a weighted linear combination of forward activation maps and pass the result through a Rectified Linear Unit (ReLU):
            </p>
            <div className="p-3 bg-slate-900 rounded-xl font-mono text-[11px] text-purple-300 border border-slate-800 text-center">
              {"L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)"}
            </div>
            <p className="text-slate-400 text-[11px]">
              The ReLU operator filters out negative influences, ensuring the heatmap highlights only features that positively drive the Parkinson prediction.
            </p>
          </div>
        </div>
      </div>

      {/* Target Layer Selector & Interactive Architecture Breakdown */}
      <div className="glass-panel p-6 rounded-3xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-lg font-bold text-white">Target Convolutional Stages</h3>
            <p className="text-xs text-slate-400">Select architecture to inspect its optimal Grad-CAM feature target</p>
          </div>

          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
            {(['custom_cnn', 'vgg16', 'resnet50'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setActiveLayer(m)}
                className={`px-3 py-1.5 rounded-lg font-mono uppercase tracking-wider font-semibold transition-all ${
                  activeLayer === m
                    ? 'bg-purple-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {m.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          <div className="glass-card p-5 rounded-2xl space-y-2">
            <span className="text-[10px] font-mono uppercase text-slate-500">TARGETED CONV STAGE</span>
            <div className="text-base font-bold text-purple-300 font-mono">
              {activeLayer === 'custom_cnn' && 'block3_conv (Conv2D - 128 filters)'}
              {activeLayer === 'vgg16' && 'block5_conv3 (VGG Stage 5 Conv)'}
              {activeLayer === 'resnet50' && 'conv5_block3_out (Terminal Bottleneck)'}
            </div>
            <p className="text-slate-400 text-[11px]">
              Terminal convolutional layers possess the richest combination of semantic detail and spatial location before dense pooling.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl space-y-2">
            <span className="text-[10px] font-mono uppercase text-slate-500">FEATURE MAP DIMENSION</span>
            <div className="text-base font-bold text-sky-400 font-mono">
              {activeLayer === 'custom_cnn' && '28 × 28 × 128'}
              {activeLayer === 'vgg16' && '7 × 7 × 512'}
              {activeLayer === 'resnet50' && '7 × 7 × 2048'}
            </div>
            <p className="text-slate-400 text-[11px]">
              Spatial resolution of activations before upsampling to match the 224×224 canvas.
            </p>
          </div>

          <div className="glass-card p-5 rounded-2xl space-y-2">
            <span className="text-[10px] font-mono uppercase text-slate-500">RECEPTIVE FIELD SCOPE</span>
            <div className="text-base font-bold text-emerald-400 font-mono">
              {activeLayer === 'custom_cnn' && 'Local stroke curvature & tremor'}
              {activeLayer === 'vgg16' && 'Full drawing loop continuity'}
              {activeLayer === 'resnet50' && 'Global Archimedean dilation'}
            </div>
            <p className="text-slate-400 text-[11px]">
              Hierarchy of visual concepts captured at this representation stage.
            </p>
          </div>
        </div>
      </div>

      {/* Clinical Sanity Checks & Limitations Alert */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl space-y-3 border-l-4 border-l-amber-500">
          <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
            <AlertTriangle className="w-4 h-4" />
            <span>Saliency Map Sanity Checks</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            In medical imaging, saliency methods can falsely appear plausible while actually tracking confounding features. 
            We audit Grad-CAM heatmaps against two failure modes:
          </p>
          <ul className="text-xs text-slate-400 space-y-1.5 list-disc pl-4">
            <li><strong>Border Artifacts:</strong> Heatmaps focusing on letterbox padding borders rather than pen strokes.</li>
            <li><strong>Background Contrast:</strong> Saliency clustered around scanner brightness gradients rather than drawing lines.</li>
          </ul>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-3 border-l-4 border-l-purple-500">
          <div className="flex items-center gap-2 text-purple-400 font-bold text-sm">
            <CheckCircle2 className="w-4 h-4" />
            <span>Clinical Biomarker Localization</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            When correctly localized on true tremor strokes, Grad-CAM highlights:
          </p>
          <ul className="text-xs text-slate-400 space-y-1.5 list-disc pl-4">
            <li><strong>Tremor Waves:</strong> 4–6 Hz oscillations along spiral radius expansion.</li>
            <li><strong>Hesitation Breaks:</strong> Sudden stops and stroke thickness irregularities in meanders.</li>
            <li><strong>Trajectory Departure:</strong> Deviations from ideal Archimedean curvature ($r = a + b\theta$).</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
