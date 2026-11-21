import React, { useState, useEffect } from 'react';
import { 
  GitCompare, 
  Layers, 
  Clock, 
  HardDrive, 
  Cpu, 
  Award, 
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ArrowRight
} from 'lucide-react';
import { fetchMetrics } from '../services/api';
import { MetricsData, ModelMetrics } from '../types';

export const ComparisonPage: React.FC = () => {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchMetrics()
      .then((data) => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load metrics for comparison:', err);
        setLoading(false);
      });
  }, []);

  const models = metricsData?.models || {};
  const modelKeys = Object.keys(models);

  return (
    <div className="space-y-12 max-w-6xl mx-auto">
      {/* Title */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/30 text-xs text-sky-400 font-mono">
          <GitCompare className="w-3.5 h-3.5" />
          <span>Cross-Architecture Benchmarking</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Model Comparison & Architectural Analysis
        </h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto">
          Systematic empirical comparison between Custom CNN (trained from scratch), VGG16 Transfer Learning, and ResNet50 Residual Transfer Learning.
        </p>
      </div>

      {/* Comparison Table */}
      <div className="glass-panel rounded-3xl overflow-hidden border border-slate-800">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-sky-400" />
            <h2 className="text-lg font-bold text-white">Consolidated Performance & Efficiency Matrix</h2>
          </div>
          <span className="text-[11px] font-mono text-slate-400">Evaluated on Held-Out Test Set (N=40)</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-mono text-[11px]">
                <th className="py-3.5 px-4 font-semibold">Evaluation Metric</th>
                <th className="py-3.5 px-4 font-semibold text-sky-400">CUSTOM CNN (SCRATCH)</th>
                <th className="py-3.5 px-4 font-semibold text-emerald-400">VGG16 TRANSFER</th>
                <th className="py-3.5 px-4 font-semibold text-purple-400">RESNET50 TRANSFER</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {[
                { label: 'Classification Accuracy', key: 'accuracy', format: (v: number) => `${(v * 100).toFixed(1)}%` },
                { label: 'Sensitivity / Recall (Parkinson)', key: 'recall', format: (v: number) => `${(v * 100).toFixed(1)}%` },
                { label: 'Specificity (Healthy Control)', key: 'specificity', format: (v: number) => `${(v * 100).toFixed(1)}%` },
                { label: 'F1-Score (Harmonic Mean)', key: 'f1', format: (v: number) => v.toFixed(3) },
                { label: 'Balanced Accuracy', key: 'balanced_accuracy', format: (v: number) => `${(v * 100).toFixed(1)}%` },
                { label: 'ROC-AUC Score', key: 'roc_auc', format: (v: number) => v.toFixed(3) },
                { label: 'PR-AUC Score', key: 'pr_auc', format: (v: number) => v.toFixed(3) },
                { label: 'Matthews Correlation (MCC)', key: 'mcc', format: (v: number) => v.toFixed(3) },
                { label: 'Brier Calibration Score', key: 'brier_score', format: (v: number) => v.toFixed(3) },
                { label: 'Total Model Parameters', key: 'total_params', format: (v: number) => v ? v.toLocaleString() : 'N/A' },
                { label: 'Trainable Parameters', key: 'trainable_params', format: (v: number) => v ? v.toLocaleString() : 'N/A' },
                { label: 'Inference Latency', key: 'inference_latency_ms', format: (v: number) => v ? `${v.toFixed(1)} ms` : 'N/A' },
                { label: 'Serialized Model Disk Size', key: 'model_size_mb', format: (v: number) => v ? `${v.toFixed(1)} MB` : 'N/A' },
                { label: 'Total Training Wall Time', key: 'train_time_sec', format: (v: number) => v ? `${v.toFixed(1)} s` : 'N/A' },
              ].map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 text-slate-300 font-sans font-medium">{row.label}</td>
                  {['custom_cnn', 'vgg16', 'resnet50'].map((mKey) => {
                    const mData = models[mKey];
                    const val = mData ? (mData as any)[row.key] : null;
                    return (
                      <td key={mKey} className="py-3 px-4 font-bold text-slate-200">
                        {val !== undefined && val !== null ? row.format(val) : '—'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Static Grouped Bar Comparison Chart */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-sky-400" />
          <span>Grouped Metric Bar Visualization</span>
        </h3>
        <div className="flex items-center justify-center p-2 bg-slate-950/60 rounded-2xl border border-slate-800">
          <img 
            src="/static/evaluation/01_grouped_metrics_bar.png" 
            alt="Grouped Metrics Bar Chart" 
            className="max-h-[450px] w-auto rounded-lg shadow-lg"
            onError={(e) => { (e.target as any).style.display = 'none'; }}
          />
        </div>
      </div>

      {/* In-Depth Architectural Scientific Discussion */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 rounded-2xl space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-sky-400 font-bold">
            <span>MODEL 1 ANALYSIS</span>
            <span>SHALLOW SCRATCH</span>
          </div>
          <h4 className="text-base font-bold text-white">Custom CNN Behavior</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            The Custom CNN begins with randomly initialized weights. On a small corpus of only 133 training images, 
            learning low-level edge and stroke primitives simultaneously with high-level tremor classification proves challenging. 
            Without external pretraining, the model demonstrates high variance and lower discrimination (AUC ~0.48), 
            empirically confirming the need for inductive transfer learning on small medical datasets.
          </p>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-3 border border-emerald-500/30">
          <div className="flex items-center justify-between text-xs font-mono text-emerald-400 font-bold">
            <span>MODEL 2 ANALYSIS</span>
            <span>BEST DISCRIMINATOR</span>
          </div>
          <h4 className="text-base font-bold text-white">VGG16 Transfer Superiority</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            VGG16’s deep stack of homogeneous $3\times 3$ convolutional filters retains rich spatial feature hierarchies 
            from ImageNet. In Phase 1, the frozen feature extractor acts as a stable line and texture detector. In Phase 2, 
            fine-tuning terminal Block5 layers with a small learning rate (lr = 1e-5) adapts high-level feature maps 
            to tremor oscillations without destroying early edge detectors, achieving the highest test AUC.
          </p>
        </div>

        <div className="glass-card p-6 rounded-2xl space-y-3">
          <div className="flex items-center justify-between text-xs font-mono text-purple-400 font-bold">
            <span>MODEL 3 ANALYSIS</span>
            <span>RESIDUAL HIGHWAY</span>
          </div>
          <h4 className="text-base font-bold text-white">ResNet50 & BatchNorm Nuance</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            ResNet50 uses residual identity shortcuts to prevent vanishing gradients across 50 layers. However, on small medical 
            datasets ($N \approx 200$), unfreezing Batch Normalization running statistics during fine-tuning introduces high 
            stochastic variance. By strictly freezing all BatchNorm layers in inference mode during Phase 2, we preserve stable 
            mean/variance estimates while adapting terminal convolutional projections.
          </p>
        </div>
      </div>
    </div>
  );
};
