import React from 'react';
import { 
  Cpu, 
  Layers, 
  Zap,
  ShieldCheck, 
  Terminal,
  Activity,
  Sparkles,
  GitBranch
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  const architectures = [
    {
      name: 'VGG16 Transfer Learning',
      status: 'Primary Production Model',
      params: '14.7M parameters',
      desc: '13-layer deep convolutional feature extractor pretrained on ImageNet. Fine-tuned with two-phase block5 adaptation for micro-stroke tremor detection.',
      badge: 'Best Accuracy & AUC',
      color: 'emerald'
    },
    {
      name: 'ResNet50 Residual Architecture',
      status: 'Deep Residual Benchmark',
      params: '23.6M parameters',
      desc: '50-layer deep architecture featuring residual bottleneck identity mappings. Fine-tuned with frozen BatchNormalization layers to preserve feature stability.',
      badge: 'High Residual Stability',
      color: 'sky'
    },
    {
      name: 'Custom 3-Stage CNN',
      status: 'Baseline Scratch Model',
      params: '102K parameters',
      desc: 'Lightweight convolutional network with Batch Normalization, Max Pooling, and Global Average Pooling trained directly on drawing patterns.',
      badge: 'Ultra-Fast Scratch Model',
      color: 'purple'
    }
  ];

  const pipelineStages = [
    { step: '01', name: 'Raw Ingestion', desc: 'Accepts Archimedean spirals & sinusoidal meanders of variable dimensions (256x256 up to 512x250).' },
    { step: '02', name: 'Letterbox Processing', desc: 'Standardizes input to 224x224 with white background canvas padding, preserving true geometric stroke aspect ratio.' },
    { step: '03', name: 'Dual Channel Normalization', desc: 'Routes images to model-specific transforms: [0, 1] scaling for Custom CNN; Caffe ImageNet BGR mean subtraction for VGG/ResNet.' },
    { step: '04', name: 'Neural Inference', desc: 'Evaluates Parkinsonian motor tremor probability with sub-100ms latency and dynamic threshold adjustment.' },
    { step: '05', name: 'Lab Report Synthesis', desc: 'Generates structured, print-ready clinical neuromotor reports with risk stratification and JSON export.' }
  ];

  return (
    <div className="space-y-8 animate-fade-in pb-12 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-sky-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <Cpu className="w-4 h-4" /> System Architecture & Engineering Specifications
        </div>
        <h1 className="text-3xl font-bold text-white tracking-tight">NeuroSpiral Engine Overview</h1>
        <p className="text-sm text-slate-400 mt-1 max-w-2xl">
          Complete engineering blueprint of the deep learning computer vision pipeline, neural architectures, and real-time inference microservice.
        </p>
      </div>

      {/* High-Level Architecture Flow */}
      <div className="card border-slate-800 bg-slate-900/40 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-400" /> End-to-End Processing Pipeline
          </h3>
          <span className="text-xs font-mono text-emerald-400 flex items-center gap-1">
            <Zap className="w-3.5 h-3.5" /> High-Throughput Engine
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {pipelineStages.map((stage) => (
            <div key={stage.step} className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors">
              <span className="text-xs font-bold text-sky-400 font-mono">{stage.step}</span>
              <h4 className="font-semibold text-white text-xs">{stage.name}</h4>
              <p className="text-[11px] text-slate-400 leading-relaxed">{stage.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Model Topologies */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-indigo-400" /> Deep Learning Topologies
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {architectures.map((arch, idx) => (
            <div key={idx} className="card border-slate-800 bg-slate-900/50 space-y-3 p-5">
              <div className="flex items-center justify-between">
                <span className={`text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded font-bold ${
                  arch.color === 'emerald' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
                  arch.color === 'sky' ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' :
                  'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                }`}>
                  {arch.badge}
                </span>
                <span className="text-[11px] font-mono text-slate-400">{arch.params}</span>
              </div>
              <h4 className="text-base font-bold text-white">{arch.name}</h4>
              <p className="text-xs text-slate-400 leading-relaxed">{arch.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Technical Specifications */}
      <div className="card border-slate-800 bg-slate-900/40 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-400" /> Technology Stack
          </h3>
          <span className="text-xs font-mono text-slate-400">Production Ready</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-mono text-[10px] uppercase">Deep Learning Core</span>
            <div className="text-white font-semibold">TensorFlow 2.21 & Keras 3</div>
            <p className="text-[11px] text-slate-400">Transfer learning, Grad-CAM gradient tape</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-mono text-[10px] uppercase">Backend Microservice</span>
            <div className="text-white font-semibold">FastAPI & Pydantic</div>
            <p className="text-[11px] text-slate-400">Asynchronous REST API, typed contracts</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-mono text-[10px] uppercase">Frontend Application</span>
            <div className="text-white font-semibold">React 18 & TypeScript</div>
            <p className="text-[11px] text-slate-400">Vite bundler, Tailwind CSS, Lucide icons</p>
          </div>
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
            <span className="text-slate-400 font-mono text-[10px] uppercase">Data Analytics</span>
            <div className="text-white font-semibold">scikit-learn & SciPy</div>
            <p className="text-[11px] text-slate-400">Group K-Fold validation, ROC-AUC, Brier score</p>
          </div>
        </div>
      </div>
    </div>
  );
};
