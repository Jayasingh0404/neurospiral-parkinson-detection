import React, { useState } from 'react';
import {
  ArrowRight,
  ArrowUpRight,
  ShieldAlert,
  Layers,
  Cpu,
  CheckCircle2,
  FileCheck2
} from 'lucide-react';
import { NavTab } from '../components/Navbar';

interface Props {
  setActiveTab: (tab: NavTab) => void;
}

type ArchKey = 'vgg16' | 'resnet50' | 'custom_cnn';

const ARCH_SPECS: Record<
  ArchKey,
  {
    name: string;
    tag: string;
    summary: string;
    params: string;
    accuracy: string;
    rocAuc: string;
    brier: string;
    stages: string[];
    highlight: string;
  }
> = {
  vgg16: {
    name: 'VGG16 Transfer Learning',
    tag: 'Top Performing Architecture',
    summary:
      'ImageNet-pretrained 13-layer convolutional backbone paired with a Global Average Pooling classification head. Trained via a two-phase protocol: head warmup followed by terminal Block5 fine-tuning at lr = 1e-5.',
    params: '14,715,201',
    accuracy: '80.0%',
    rocAuc: '0.895',
    brier: '0.123',
    stages: ['Input 224×224 BGR', 'Blocks 1–4 (Frozen)', 'Block 5 (Fine-Tuned)', 'GAP + Dense(256)', 'Sigmoid Output'],
    highlight: 'Homogeneous 3×3 convolutional filters transfer cleanly to pen-stroke curvature and micro-tremor discontinuities.',
  },
  resnet50: {
    name: 'ResNet50 Residual Network',
    tag: 'Deep Residual Benchmark',
    summary:
      '50-layer deep residual architecture utilizing identity skip connections to stabilize gradient flow. Fine-tuned across Stage 5 residual blocks while keeping all BatchNormalization layers explicitly frozen.',
    params: '23,589,761',
    accuracy: '77.5%',
    rocAuc: '0.875',
    brier: '0.136',
    stages: ['Input 224×224 BGR', 'Stages 1–4 (Frozen)', 'Stage 5 (Frozen BN)', 'GAP + Dense(256)', 'Sigmoid Output'],
    highlight: 'Freezing BatchNormalization prevents running-statistic degradation on small biomedical mini-batches (N = 16).',
  },
  custom_cnn: {
    name: 'Custom CNN (From Scratch)',
    tag: 'Non-Pretrained Baseline',
    summary:
      'Lightweight 3-block convolutional neural network initialized from scratch without external visual priors. Designed to empirically test whether shallow scratch models can generalize on small cohorts.',
    params: '102,465',
    accuracy: '47.5%',
    rocAuc: '0.476',
    brier: '0.270',
    stages: ['Input 224×224 [0,1]', 'Conv(32) + BN + Pool', 'Conv(64) + BN + Pool', 'Conv(128) + GAP', 'Dense(128) → Sigmoid'],
    highlight: 'Demonstrates that training from scratch on 133 images converges near chance, validating transfer learning.',
  },
};

const PIPELINE_STEPS = [
  {
    num: '01',
    title: 'Dataset',
    detail: '204 HandPD spiral & meander exams across 28 distinct subjects (102 Healthy, 102 Parkinson).',
  },
  {
    num: '02',
    title: 'Preprocessing',
    detail: 'Aspect-ratio-preserving 224×224 white-canvas letterboxing & model-specific normalization.',
  },
  {
    num: '03',
    title: 'Deep Learning',
    detail: 'Custom CNN from scratch benchmarked against two-phase VGG16 & ResNet50 transfer learning.',
  },
  {
    num: '04',
    title: 'Evaluation',
    detail: 'Dual benchmarking: held-out test metrics plus zero-leakage 5-fold StratifiedGroupKFold CV.',
  },
  {
    num: '05',
    title: 'Explainable AI',
    detail: 'Spatial attribution verification ensuring predictions align with pen-stroke tremor geometry.',
  },
];

export const LandingPage: React.FC<Props> = ({ setActiveTab }) => {
  const [selectedArch, setSelectedArch] = useState<ArchKey>('vgg16');
  const activeSpec = ARCH_SPECS[selectedArch];

  return (
    <div className="space-y-28 pb-16">
      {/* ================================================================= */}
      {/* HERO SECTION                                                      */}
      {/* ================================================================= */}
      <section className="relative pt-8 lg:pt-14 pb-8 overflow-hidden">
        <div className="absolute -top-24 right-0 w-[520px] h-[520px] bg-sky-500/10 blur-[140px] rounded-full pointer-events-none" />
        <div className="absolute bottom-0 -left-24 w-[420px] h-[420px] bg-indigo-500/10 blur-[130px] rounded-full pointer-events-none" />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-10 items-center relative">
          {/* Left Editorial Column */}
          <div className="lg:col-span-7 space-y-7">
            <div className="inline-flex items-center gap-2.5 text-xs font-mono tracking-widest uppercase text-sky-400">
              <span className="w-8 h-px bg-sky-400" />
              <span>Neuromotor Computer Vision Research</span>
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-[4.25rem] font-extrabold text-white tracking-tight leading-[1.06]">
              Understanding <br />
              Movement <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-400 via-indigo-300 to-emerald-400">
                Through AI.
              </span>
            </h1>

            <p className="text-base sm:text-lg text-slate-300/90 max-w-xl leading-relaxed font-light">
              Investigating how deep convolutional neural networks quantify subtle motor tremors,
              stroke hesitations, and micrographic patterns in digitized spiral and meander drawing
              examinations for Parkinson&apos;s disease classification research.
            </p>

            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => setActiveTab('predictor')}
                className="group inline-flex items-center gap-2.5 px-7 py-3.5 rounded-full bg-white text-slate-950 hover:bg-sky-100 font-semibold text-sm transition-all shadow-lg shadow-white/10"
              >
                <span>Explore Image Analysis</span>
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
              </button>

              <button
                onClick={() => setActiveTab('performance')}
                className="group inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-transparent hover:bg-slate-800/70 border border-slate-700 text-slate-200 font-semibold text-sm transition-all"
              >
                <span>Explore Research</span>
                <ArrowUpRight className="w-4 h-4 text-sky-400 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </button>
            </div>

            {/* Subtle editorial metadata strip */}
            <div className="pt-6 border-t border-slate-800/80 flex flex-wrap items-center gap-x-8 gap-y-2 text-xs font-mono text-slate-400">
              <div>
                <span className="text-white font-semibold">204</span> HandPD Specimens
              </div>
              <div>
                <span className="text-white font-semibold">3</span> Deep Architectures
              </div>
              <div>
                <span className="text-emerald-400 font-semibold">0.895</span> Peak ROC-AUC
              </div>
            </div>
          </div>

          {/* Right Custom Scientific Visual */}
          <div className="lg:col-span-5">
            <div className="relative rounded-3xl border border-slate-800/90 bg-slate-950/90 p-6 overflow-hidden shadow-2xl">
              {/* Top Scientific Telemetry Bar */}
              <div className="flex items-center justify-between text-[10px] font-mono uppercase tracking-widest text-slate-500 pb-4 border-b border-slate-800/80">
                <span>FIG 01. KINEMATIC SPIRAL EXTRACTION</span>
                <span className="text-sky-400">224 &times; 224 &times; 3</span>
              </div>

              {/* Bespoke Scientific SVG: Archimedean Spiral + Spatial Sampling Grid + Feature Flow */}
              <div className="py-5 flex items-center justify-center">
                <svg
                  viewBox="0 0 420 300"
                  className="w-full h-auto"
                  role="img"
                  aria-label="Scientific diagram of a spiral drawing processed through a convolutional neural network"
                >
                  <defs>
                    <linearGradient id="spiralGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#38bdf8" />
                      <stop offset="55%" stopColor="#818cf8" />
                      <stop offset="100%" stopColor="#34d399" />
                    </linearGradient>
                    <linearGradient id="tremorGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.9" />
                      <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.9" />
                    </linearGradient>
                    <pattern id="spatialGrid" width="20" height="20" patternUnits="userSpaceOnUse">
                      <path
                        d="M 20 0 L 0 0 0 20"
                        fill="none"
                        stroke="#1e293b"
                        strokeWidth="0.8"
                      />
                    </pattern>
                  </defs>

                  {/* Spatial sampling canvas */}
                  <rect x="14" y="24" width="210" height="210" rx="12" fill="url(#spatialGrid)" stroke="#334155" strokeWidth="1" />

                  {/* Reference Smooth Archimedean Spiral */}
                  <path
                    d="M 119 129
                       C 122 123, 130 124, 130 131
                       C 130 140, 117 146, 108 140
                       C 96 132, 97 113, 111 106
                       C 128 97, 150 107, 154 127
                       C 159 151, 138 173, 113 172
                       C 83 171, 65 141, 73 112
                       C 82 78, 121 61, 154 74
                       C 191 89, 205 135, 186 171
                       C 166 208, 116 221, 78 201
                       C 40 181, 28 131, 47 90"
                    fill="none"
                    stroke="url(#spiralGrad)"
                    strokeWidth="2.4"
                    strokeLinecap="round"
                  />

                  {/* Micro-tremor perturbation overlay segment */}
                  <path
                    d="M 154 74 L 161 79 L 166 77 L 173 86 L 177 85 L 184 96 L 186 104 L 193 111 L 192 121 L 197 130 L 194 140 L 196 150 L 190 160 L 186 171"
                    fill="none"
                    stroke="url(#tremorGrad)"
                    strokeWidth="2.2"
                    strokeDasharray="3 2"
                    strokeLinecap="round"
                  />

                  {/* Receptive field inspection window */}
                  <rect
                    x="148"
                    y="68"
                    width="56"
                    height="56"
                    rx="4"
                    fill="rgba(244, 63, 94, 0.08)"
                    stroke="#f43f5e"
                    strokeWidth="1.2"
                    strokeDasharray="4 3"
                  />
                  <circle cx="176" cy="96" r="3" fill="#f43f5e" />

                  {/* Projection lines from receptive field to Conv Feature Maps */}
                  <line x1="204" y1="68" x2="262" y2="56" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />
                  <line x1="204" y1="124" x2="262" y2="146" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />

                  {/* Convolutional Feature Map Stack */}
                  <g transform="translate(262, 48)">
                    <rect x="0" y="0" width="44" height="98" rx="6" fill="#0f172a" stroke="#38bdf8" strokeWidth="1.2" />
                    <rect x="10" y="8" width="44" height="98" rx="6" fill="#0f172a" stroke="#818cf8" strokeWidth="1.2" />
                    <rect x="20" y="16" width="44" height="98" rx="6" fill="#0f172a" stroke="#34d399" strokeWidth="1.2" />
                    {/* Activation dots */}
                    <circle cx="34" cy="38" r="2.5" fill="#34d399" />
                    <circle cx="48" cy="52" r="3" fill="#38bdf8" />
                    <circle cx="38" cy="76" r="2.5" fill="#f43f5e" />
                    <circle cx="52" cy="92" r="2.5" fill="#818cf8" />
                  </g>

                  {/* Connector to Output Node */}
                  <line x1="326" y1="112" x2="358" y2="112" stroke="#64748b" strokeWidth="1.2" />
                  <polygon points="358,108 366,112 358,116" fill="#38bdf8" />

                  {/* Sigmoid Probability Output Column */}
                  <g transform="translate(368, 72)">
                    <rect x="0" y="0" width="38" height="80" rx="8" fill="#0f172a" stroke="#334155" strokeWidth="1.2" />
                    <rect x="8" y="12" width="22" height="22" rx="4" fill="rgba(52, 211, 153, 0.2)" stroke="#34d399" strokeWidth="1" />
                    <text x="19" y="27" textAnchor="middle" fill="#34d399" fontSize="9" fontFamily="monospace" fontWeight="bold">H</text>
                    <rect x="8" y="44" width="22" height="22" rx="4" fill="rgba(244, 63, 94, 0.2)" stroke="#f43f5e" strokeWidth="1" />
                    <text x="19" y="59" textAnchor="middle" fill="#f43f5e" fontSize="9" fontFamily="monospace" fontWeight="bold">P</text>
                  </g>

                  {/* Bottom Scientific Labels */}
                  <text x="119" y="254" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">
                    LETTERBOXED SPIRAL INPUT
                  </text>
                  <text x="294" y="182" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="monospace">
                    CONV STAGES
                  </text>
                  <text x="387" y="172" textAnchor="middle" fill="#94a3b8" fontSize="9" fontFamily="monospace">
                    P(y|x)
                  </text>

                  {/* Legend Row */}
                  <g transform="translate(16, 274)">
                    <line x1="0" y1="6" x2="16" y2="6" stroke="#38bdf8" strokeWidth="2" />
                    <text x="22" y="9" fill="#94a3b8" fontSize="9" fontFamily="monospace">Smooth Trajectory</text>
                    <line x1="145" y1="6" x2="161" y2="6" stroke="#f43f5e" strokeWidth="2" strokeDasharray="3 2" />
                    <text x="167" y="9" fill="#94a3b8" fontSize="9" fontFamily="monospace">Tremor Oscillation (4–6 Hz)</text>
                  </g>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* SECTION A: PROJECT OVERVIEW (Editorial Asymmetric Layout)         */}
      {/* ================================================================= */}
      <section className="border-t border-slate-800/80 pt-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-16 items-start">
          <div className="lg:col-span-5 space-y-4">
            <span className="text-xs font-mono uppercase tracking-widest text-sky-400">
              01 / Project Overview
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight leading-tight">
              Isolating Fine-Motor Biomarkers from Static Drawings.
            </h2>
            <p className="text-sm text-slate-400 italic border-l-2 border-sky-500/60 pl-4 leading-relaxed">
              &ldquo;Archimedean spirals remove linguistic and literacy confounders, isolating pure
              neuromuscular execution across continuous radial loops.&rdquo;
            </p>
          </div>

          <div className="lg:col-span-7 space-y-6">
            <p className="text-base text-slate-300 leading-relaxed">
              Parkinson&apos;s disease impairs dopaminergic signaling in the basal ganglia, often
              producing resting tremor, rigidity, and bradykinesia before severe motor disability is
              clinically apparent. Standard drawing tasks—concentric spirals and sinusoidal
              meanders—capture micro-hesitations, irregular line spacing, and directional stiffness
              directly on paper.
            </p>
            <p className="text-base text-slate-400 leading-relaxed">
              This project investigates whether convolutional neural networks can reliably
              distinguish healthy control drawings from Parkinsonian drawings under strict
              scientific evaluation. Rather than relying on inflated image-level splits, our work
              audits participant provenance, standardizes heterogeneous aspect ratios without
              geometric distortion, and compares scratch-trained networks against fine-tuned
              transfer learning architectures.
            </p>

            {/* Hairline Key-Value Strip (No Cards) */}
            <div className="grid grid-cols-3 gap-6 pt-6 border-t border-slate-800/80">
              <div>
                <div className="text-2xl font-extrabold text-white font-mono">102 / 102</div>
                <div className="text-xs text-slate-400 mt-1">Balanced Healthy &amp; Parkinson Classes</div>
              </div>
              <div>
                <div className="text-2xl font-extrabold text-white font-mono">28</div>
                <div className="text-xs text-slate-400 mt-1">Audited Human Participants</div>
              </div>
              <div>
                <div className="text-2xl font-extrabold text-sky-400 font-mono">224&times;224</div>
                <div className="text-xs text-slate-400 mt-1">Aspect-Preserving Letterbox Input</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* SECTION B: RESEARCH PIPELINE (Connected Linear Flow Track)        */}
      {/* ================================================================= */}
      <section className="border-t border-slate-800/80 pt-16 space-y-12">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="space-y-2">
            <span className="text-xs font-mono uppercase tracking-widest text-sky-400">
              02 / Research Pipeline
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
              From Raw Specimen to Interpretable Output.
            </h2>
          </div>
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">
            End-to-End Reproducible Workflow
          </p>
        </div>

        {/* Horizontal Connected Flow */}
        <div className="relative">
          {/* Connecting rail on desktop */}
          <div className="hidden lg:block absolute top-5 left-6 right-6 h-px bg-gradient-to-r from-sky-500/60 via-indigo-400/60 to-emerald-400/60" />

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8 relative">
            {PIPELINE_STEPS.map((step, idx) => (
              <div key={step.num} className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-slate-950 border border-sky-500/50 text-sky-300 font-mono text-xs font-bold flex items-center justify-center shrink-0 relative z-10 shadow-md shadow-sky-950">
                    {step.num}
                  </div>
                  {idx < PIPELINE_STEPS.length - 1 && (
                    <ArrowRight className="w-4 h-4 text-slate-600 lg:hidden" />
                  )}
                </div>

                <div className="space-y-1.5 pr-2">
                  <h3 className="text-base font-bold text-white tracking-tight">{step.title}</h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* SECTION C: MODEL ARCHITECTURE (Interactive Editorial Spec Ledger) */}
      {/* ================================================================= */}
      <section className="border-t border-slate-800/80 pt-16 space-y-10">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="space-y-2">
            <span className="text-xs font-mono uppercase tracking-widest text-sky-400">
              03 / Model Architecture
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
              Three Convolutional Paradigms.
            </h2>
          </div>

          {/* Segmented Architecture Switcher */}
          <div className="inline-flex p-1 rounded-full bg-slate-900 border border-slate-800 self-start">
            {(['vgg16', 'resnet50', 'custom_cnn'] as ArchKey[]).map((key) => (
              <button
                key={key}
                onClick={() => setSelectedArch(key)}
                className={`px-4 py-2 rounded-full text-xs font-semibold transition-all ${
                  selectedArch === key
                    ? 'bg-sky-500 text-slate-950 shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {key === 'vgg16' ? 'VGG16' : key === 'resnet50' ? 'ResNet50' : 'Custom CNN'}
              </button>
            ))}
          </div>
        </div>

        {/* Structured Editorial Ledger */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start pt-2">
          {/* Left Narrative & Layer Flow */}
          <div className="lg:col-span-7 space-y-6">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 text-xs font-mono text-emerald-400">
                <Cpu className="w-3.5 h-3.5" />
                <span>{activeSpec.tag}</span>
              </div>
              <h3 className="text-2xl font-bold text-white">{activeSpec.name}</h3>
              <p className="text-sm text-slate-300 leading-relaxed">{activeSpec.summary}</p>
            </div>

            {/* Horizontal Layer Sequence */}
            <div className="space-y-2">
              <span className="text-[10px] font-mono uppercase tracking-widest text-slate-500">
                Forward Pass Topology
              </span>
              <div className="flex flex-wrap items-center gap-2">
                {activeSpec.stages.map((stage, i) => (
                  <React.Fragment key={stage}>
                    <span className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-200">
                      {stage}
                    </span>
                    {i < activeSpec.stages.length - 1 && (
                      <ArrowRight className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="pt-2 flex items-start gap-2.5 text-xs text-slate-300 border-l-2 border-emerald-500/60 pl-4">
              <Layers className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <span>{activeSpec.highlight}</span>
            </div>
          </div>

          {/* Right Tabular Spec Readout */}
          <div className="lg:col-span-5 divide-y divide-slate-800/90 border-y border-slate-800/90 font-mono text-sm">
            <div className="py-4 flex items-center justify-between">
              <span className="text-xs uppercase tracking-wider text-slate-400">Total Parameters</span>
              <span className="text-white font-bold">{activeSpec.params}</span>
            </div>
            <div className="py-4 flex items-center justify-between">
              <span className="text-xs uppercase tracking-wider text-slate-400">Held-Out Test Accuracy</span>
              <span className="text-sky-400 font-bold text-base">{activeSpec.accuracy}</span>
            </div>
            <div className="py-4 flex items-center justify-between">
              <span className="text-xs uppercase tracking-wider text-slate-400">ROC-AUC Score</span>
              <span className="text-emerald-400 font-bold text-base">{activeSpec.rocAuc}</span>
            </div>
            <div className="py-4 flex items-center justify-between">
              <span className="text-xs uppercase tracking-wider text-slate-400">Brier Calibration</span>
              <span className="text-slate-200 font-bold">{activeSpec.brier}</span>
            </div>
            <div className="py-4 flex items-center justify-between">
              <span className="text-xs uppercase tracking-wider text-slate-400">Full Benchmark View</span>
              <button
                onClick={() => setActiveTab('comparison')}
                className="text-xs text-sky-400 hover:text-sky-300 inline-flex items-center gap-1 font-sans font-semibold"
              >
                <span>Compare All Models</span>
                <ArrowUpRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* SECTION D: RESEARCH HIGHLIGHTS (Typographic Magazine Layout)      */}
      {/* ================================================================= */}
      <section className="border-t border-slate-800/80 pt-16 space-y-12">
        <div className="space-y-2">
          <span className="text-xs font-mono uppercase tracking-widest text-sky-400">
            04 / Research Highlights
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
            Verified Empirical Findings.
          </h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          {/* Giant Typographic Focal Metric */}
          <div className="lg:col-span-5 space-y-4 lg:pr-6">
            <div className="text-6xl sm:text-7xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-amber-300 via-amber-400 to-rose-400 font-mono">
              89.3%
            </div>
            <h3 className="text-xl font-bold text-white">
              Subject-Level Split Contamination Audited &amp; Exposed.
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Programmatic filename analysis revealed that 25 of 28 unique participants appear
              simultaneously across training, validation, and testing folders in the raw dataset
              distribution. Grouping by participant ID via 5-fold{' '}
              <code className="text-sky-300 bg-slate-900 px-1.5 py-0.5 rounded text-xs">
                StratifiedGroupKFold
              </code>{' '}
              eliminates identity memorization and provides unbiased generalization reporting.
            </p>
            <div className="pt-2">
              <button
                onClick={() => setActiveTab('report')}
                className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-400 hover:text-emerald-300 transition-colors"
              >
                <FileCheck2 className="w-4 h-4" />
                <span>Inspect Diagnostic Lab Report Output</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Hairline-Separated Empirical Ledger */}
          <div className="lg:col-span-7 divide-y divide-slate-800/80 border-t border-slate-800/80 lg:border-t-0">
            <div className="py-5 first:pt-0 flex items-start gap-4">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-white">
                  Transfer Learning Outperforms Scratch Training by +32.5% Accuracy
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Fine-tuned VGG16 achieved 80.0% test accuracy (0.895 ROC-AUC, 0.123 Brier score)
                  and ResNet50 achieved 77.5% (0.875 ROC-AUC), whereas the Custom CNN trained from
                  scratch on 133 images remained at 47.5% (0.476 ROC-AUC).
                </p>
              </div>
            </div>

            <div className="py-5 flex items-start gap-4">
              <CheckCircle2 className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-white">
                  Zero Geometric Distortion on Heterogeneous Meander Drawings
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Half the corpus consists of square 256&times;256 spirals and half wide rectangular
                  meanders (~512&times;250). White-canvas aspect-ratio letterboxing prevents over
                  50% horizontal compression of stroke wave frequencies.
                </p>
              </div>
            </div>

            <div className="py-5 flex items-start gap-4">
              <CheckCircle2 className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-white">
                  Frozen BatchNormalization Preserves Residual Stability
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Keeping ResNet50 BatchNormalization layers in inference mode during Stage 5
                  fine-tuning prevented running-mean collapse on small mini-batches, yielding a
                  well-calibrated 0.898 PR-AUC.
                </p>
              </div>
            </div>

            <div className="py-5 flex items-start gap-4">
              <CheckCircle2 className="w-5 h-5 text-slate-300 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-white">
                  Complete Zero-Leakage Cross-Validation Baseline
                </h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Under strict 5-fold subject-independent cross-validation (0.0% participant
                  overlap), the non-pretrained Custom CNN averaged 43.81% &plusmn; 6.14% accuracy
                  (0.531 mean ROC-AUC), confirming that inductive visual priors are mandatory.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* SECTION E: RESPONSIBLE AI (Calm Authoritative Statement Band)     */}
      {/* ================================================================= */}
      <section className="border-t border-slate-800/80 pt-16">
        <div className="border-l-4 border-amber-400/80 pl-6 sm:pl-8 py-2 space-y-6">
          <div className="flex items-center gap-2.5 text-xs font-mono uppercase tracking-widest text-amber-400">
            <ShieldAlert className="w-4 h-4" />
            <span>05 / Responsible AI &amp; Clinical Boundaries</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-6 space-y-2">
              <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                A Research Prototype — Not a Medical Diagnostic Device.
              </h2>
              <p className="text-sm text-slate-300 leading-relaxed">
                NeuroSpiral is built solely for computer vision research and observational motor
                pattern analysis. It has not received regulatory clearance from any medical
                authority and must never be used to diagnose, triage, or guide clinical treatment.
              </p>
            </div>

            <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-3 gap-6 text-xs lg:border-l lg:border-slate-800 lg:pl-8">
              <div className="space-y-1">
                <span className="font-mono text-amber-300 font-semibold uppercase block">
                  Cohort Scale
                </span>
                <p className="text-slate-400 leading-relaxed">
                  Evaluated on 28 participants; multi-center clinical trials are required for
                  broad demographic generalization.
                </p>
              </div>
              <div className="space-y-1">
                <span className="font-mono text-amber-300 font-semibold uppercase block">
                  Static Modality
                </span>
                <p className="text-slate-400 leading-relaxed">
                  2D scans omit temporal velocity, stylus pressure, and in-air hesitation kinematics
                  captured by digital tablets.
                </p>
              </div>
              <div className="space-y-1">
                <span className="font-mono text-amber-300 font-semibold uppercase block">
                  Clinical Authority
                </span>
                <p className="text-slate-400 leading-relaxed">
                  Parkinson&apos;s diagnosis requires comprehensive neurological assessment by a
                  qualified movement disorder specialist.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
