import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  ShieldCheck, 
  Clock, 
  Database, 
  Activity, 
  CheckCircle2, 
  XCircle,
  HelpCircle
} from 'lucide-react';
import { fetchMetrics, API_ORIGIN } from '../services/api';
import { MetricsData, ModelMetrics } from '../types';

export const PerformancePage: React.FC = () => {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [activeCurveTab, setActiveCurveTab] = useState<'roc' | 'pr' | 'cm' | 'curves'>('roc');

  useEffect(() => {
    fetchMetrics()
      .then((data) => {
        setMetricsData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load metrics:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="py-24 text-center space-y-4">
        <div className="w-10 h-10 border-4 border-sky-500/30 border-t-sky-500 rounded-full animate-spin mx-auto" />
        <p className="text-sm text-slate-400">Loading experimental telemetry and evaluation metrics...</p>
      </div>
    );
  }

  const models = metricsData?.models || {};
  const currentModelKeys = Object.keys(models);
  const activeMetrics: ModelMetrics | null = selectedModel !== 'all' && models[selectedModel] ? models[selectedModel] : null;

  return (
    <div className="space-y-12 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-sky-400 uppercase tracking-wider">
            <BarChart3 className="w-4 h-4" />
            <span>Experimental Verification Dashboard</span>
          </div>
          <h1 className="text-3xl font-black text-white mt-1">Model Performance Telemetry</h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical evaluation conducted on held-out test data (40 images) and 5-fold cross-validation.
          </p>
        </div>

        {/* Model switcher */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-xl">
          <button
            onClick={() => setSelectedModel('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              selectedModel === 'all'
                ? 'bg-sky-500 text-white shadow'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            All Architectures
          </button>
          {currentModelKeys.map((k) => (
            <button
              key={k}
              onClick={() => setSelectedModel(k)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-all ${
                selectedModel === k
                  ? 'bg-sky-500 text-white shadow'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {k.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {[
          { 
            label: 'Top Accuracy', 
            val: activeMetrics ? `${(activeMetrics.accuracy * 100).toFixed(1)}%` : 'Multi-Model', 
            sub: 'Held-out test set', 
            color: 'text-sky-400' 
          },
          { 
            label: 'Recall (Sensitivity)', 
            val: activeMetrics ? `${(activeMetrics.recall * 100).toFixed(1)}%` : 'Diagnostic Power', 
            sub: 'Catches true cases', 
            color: 'text-emerald-400' 
          },
          { 
            label: 'Specificity', 
            val: activeMetrics ? `${(activeMetrics.specificity * 100).toFixed(1)}%` : 'True Negative', 
            sub: 'Avoids false alarms', 
            color: 'text-indigo-400' 
          },
          { 
            label: 'F1-Score', 
            val: activeMetrics ? activeMetrics.f1.toFixed(3) : 'Balanced F1', 
            sub: 'Harmonic mean', 
            color: 'text-purple-400' 
          },
          { 
            label: 'ROC-AUC', 
            val: activeMetrics ? activeMetrics.roc_auc.toFixed(3) : 'Discrimination', 
            sub: 'Area under ROC', 
            color: 'text-amber-400' 
          },
          { 
            label: 'Brier Score', 
            val: activeMetrics ? activeMetrics.brier_score.toFixed(3) : 'Calibration', 
            sub: 'Lower is better', 
            color: 'text-rose-400' 
          },
        ].map((kpi, idx) => (
          <div key={idx} className="glass-card p-4 rounded-xl space-y-1">
            <span className="text-[10px] uppercase font-mono tracking-wider text-slate-500 block">
              {kpi.label}
            </span>
            <div className={`text-xl font-black ${kpi.color}`}>{kpi.val}</div>
            <span className="text-[10px] text-slate-400 block">{kpi.sub}</span>
          </div>
        ))}
      </div>

      {/* Primary Visualizations Tabs */}
      <div className="glass-panel p-6 rounded-3xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-sky-400" />
            <h2 className="text-lg font-bold text-white">Comparative Evaluation Charts</h2>
          </div>

          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setActiveCurveTab('roc')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeCurveTab === 'roc' ? 'bg-slate-800 text-sky-400 font-bold' : 'text-slate-400'
              }`}
            >
              ROC Curves
            </button>
            <button
              onClick={() => setActiveCurveTab('pr')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeCurveTab === 'pr' ? 'bg-slate-800 text-sky-400 font-bold' : 'text-slate-400'
              }`}
            >
              Precision-Recall
            </button>
            <button
              onClick={() => setActiveCurveTab('cm')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeCurveTab === 'cm' ? 'bg-slate-800 text-sky-400 font-bold' : 'text-slate-400'
              }`}
            >
              Confusion Matrices
            </button>
            <button
              onClick={() => setActiveCurveTab('curves')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeCurveTab === 'curves' ? 'bg-slate-800 text-sky-400 font-bold' : 'text-slate-400'
              }`}
            >
              Training Histories
            </button>
          </div>
        </div>

        {/* Dynamic Static Chart Viewer from Matplotlib Outputs */}
        <div className="flex items-center justify-center p-4 bg-slate-950/60 rounded-2xl border border-slate-800">
          {activeCurveTab === 'roc' && (
            <img 
              src={`${API_ORIGIN}/static/evaluation/03_roc_curves.png`}
              alt="ROC Curves" 
              className="max-h-[500px] w-auto rounded-lg shadow-lg"
              onError={(e) => { (e.target as any).style.display = 'none'; }}
            />
          )}
          {activeCurveTab === 'pr' && (
            <img 
              src={`${API_ORIGIN}/static/evaluation/04_precision_recall_curves.png`}
              alt="Precision Recall Curves" 
              className="max-h-[500px] w-auto rounded-lg shadow-lg"
              onError={(e) => { (e.target as any).style.display = 'none'; }}
            />
          )}
          {activeCurveTab === 'cm' && (
            <img 
              src={`${API_ORIGIN}/static/evaluation/02_confusion_matrices.png`}
              alt="Confusion Matrices" 
              className="max-h-[500px] w-auto rounded-lg shadow-lg"
              onError={(e) => { (e.target as any).style.display = 'none'; }}
            />
          )}
          {activeCurveTab === 'curves' && (
            <img 
              src={`${API_ORIGIN}/static/evaluation/05_training_curves.png`}
              alt="Training and Validation Curves" 
              className="max-h-[500px] w-auto rounded-lg shadow-lg"
              onError={(e) => { (e.target as any).style.display = 'none'; }}
            />
          )}
        </div>
      </div>

      {/* Subject-Independent Cross-Validation Section */}
      <div className="glass-panel p-6 rounded-2xl space-y-4 border-l-4 border-l-emerald-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            <h3 className="text-base font-bold text-white">Subject-Independent 5-Fold Cross-Validation Audit</h3>
          </div>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            0% Participant Leakage
          </span>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed">
          In contrast to image-level test splits where drawings from the same patient can leak between training and testing sets, 
          our <strong>Stratified Group 5-Fold Cross-Validation</strong> partitions all 28 subjects into disjoint cohorts. 
          This provides the scientifically rigorous estimate of true generalization to unseen participants required by CLAIM 2024.
        </p>

        {metricsData?.subject_independent_summary?.custom_cnn && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono pt-2">
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 block">MEAN ACCURACY</span>
              <span className="text-emerald-400 font-bold">
                {(metricsData.subject_independent_summary.custom_cnn.mean_accuracy * 100).toFixed(1)}% 
                <span className="text-[10px] text-slate-500 font-normal"> (±{(metricsData.subject_independent_summary.custom_cnn.std_accuracy * 100).toFixed(1)}%)</span>
              </span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 block">MEAN SENSITIVITY</span>
              <span className="text-slate-200 font-bold">
                {(metricsData.subject_independent_summary.custom_cnn.mean_recall * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 block">MEAN SPECIFICITY</span>
              <span className="text-slate-200 font-bold">
                {(metricsData.subject_independent_summary.custom_cnn.mean_specificity * 100).toFixed(1)}%
              </span>
            </div>
            <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              <span className="text-[10px] text-slate-500 block">MEAN ROC-AUC</span>
              <span className="text-sky-400 font-bold">
                {metricsData.subject_independent_summary.custom_cnn.mean_roc_auc.toFixed(3)}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

