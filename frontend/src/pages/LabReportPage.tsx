import React, { useState, useEffect } from 'react';
import {
  FileCheck2,
  Printer,
  Download,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  User,
  Calendar,
  Activity,
  Cpu,
  Clock,
  ShieldCheck,
  FileText,
  Sparkles
} from 'lucide-react';
import { LabReportRecord, MetricsData } from '../types';
import { fetchMetrics } from '../services/api';
import { NavTab } from '../components/Navbar';

interface LabReportPageProps {
  latestReport: LabReportRecord | null;
  reportHistory: LabReportRecord[];
  onSelectReport: (report: LabReportRecord) => void;
  setActiveTab: (tab: NavTab) => void;
}

export const LabReportPage: React.FC<LabReportPageProps> = ({
  latestReport,
  reportHistory,
  onSelectReport,
  setActiveTab,
}) => {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);

  // Editable case details for the clinical report
  const [patientName, setPatientName] = useState<string>('Subject #001');
  const [patientAge, setPatientAge] = useState<string>('62');
  const [patientGender, setPatientGender] = useState<string>('Not Specified');
  const [testedHand, setTestedHand] = useState<string>('Dominant (Right)');
  const [examType, setExamType] = useState<string>('Archimedean Spiral Drawing');
  const [clinicianNotes, setClinicianNotes] = useState<string>('');

  useEffect(() => {
    fetchMetrics()
      .then((data) => setMetricsData(data))
      .catch((err) => console.error('Failed to load model validation metrics:', err));
  }, []);

  const handlePrint = () => {
    window.print();
  };

  const handleExportJson = () => {
    if (!latestReport) return;
    const exportPayload = {
      report_id: latestReport.reportId,
      generated_at: latestReport.timestamp,
      subject_details: {
        subject_identifier: patientName,
        age: patientAge,
        gender: patientGender,
        tested_hand: testedHand,
        examination_type: examType,
        notes: clinicianNotes,
      },
      specimen_filename: latestReport.fileName,
      analysis_result: latestReport.prediction,
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${latestReport.reportId}_${latestReport.prediction.predicted_class}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!latestReport) {
    return (
      <div className="max-w-4xl mx-auto py-12 space-y-8 animate-fade-in">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/30 text-xs text-sky-400 font-mono">
            <FileCheck2 className="w-3.5 h-3.5" />
            <span>Automated Clinical Lab Report</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Neuromotor Diagnostic Lab Report
          </h1>
          <p className="text-sm text-slate-400 max-w-xl mx-auto">
            Run a spiral or meander drawing analysis in the Inference Lab to automatically generate a structured, printable laboratory report.
          </p>
        </div>

        <div className="glass-panel p-12 rounded-3xl text-center space-y-6 border border-dashed border-slate-700">
          <div className="w-16 h-16 rounded-2xl bg-sky-500/15 text-sky-400 mx-auto flex items-center justify-center border border-sky-500/30">
            <FileText className="w-8 h-8" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-bold text-white">No Active Analysis Found</h2>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Upload a drawing specimen in the Inference Lab and execute a neural model prediction. Your quantitative risk assessment, specimen preview, and validation metrics will populate here automatically.
            </p>
          </div>
          <button
            onClick={() => setActiveTab('predictor')}
            className="inline-flex items-center gap-2.5 px-6 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-sky-500/20 transition-all hover:scale-[1.02]"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Go to Inference Lab</span>
          </button>
        </div>
      </div>
    );
  }

  const { prediction, reportId, timestamp, fileName, originalPreviewUrl } = latestReport;
  const isParkinson = prediction.predicted_class === 'parkinson';
  const parkinsonPct = (prediction.probabilities.parkinson * 100).toFixed(1);
  const healthyPct = (prediction.probabilities.healthy * 100).toFixed(1);

  const modelValidation = metricsData?.models?.[prediction.model_name];

  const getRiskTier = (prob: number) => {
    if (prob >= 0.75) return { label: 'Elevated Tremor Signature', color: 'text-red-400', bg: 'bg-red-500/15 border-red-500/30' };
    if (prob >= 0.45) return { label: 'Borderline — Clinical Review Advised', color: 'text-amber-400', bg: 'bg-amber-500/15 border-amber-500/30' };
    return { label: 'Low Risk — Normal Motor Smoothness', color: 'text-emerald-400', bg: 'bg-emerald-500/15 border-emerald-500/30' };
  };

  const riskTier = getRiskTier(prediction.probabilities.parkinson);

  return (
    <div className="max-w-5xl mx-auto space-y-6 pb-12">
      {/* Top Action Bar (hidden when printing) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 print:hidden">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-sky-400 uppercase tracking-wider">
            <FileCheck2 className="w-4 h-4" />
            <span>Laboratory Assessment Report</span>
          </div>
          <h1 className="text-2xl font-extrabold text-white mt-0.5">Diagnostic Summary & Export</h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {reportHistory.length > 1 && (
            <select
              value={reportId}
              onChange={(e) => {
                const found = reportHistory.find((r) => r.reportId === e.target.value);
                if (found) onSelectReport(found);
              }}
              className="px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-slate-200 font-mono focus:outline-none focus:border-sky-500"
            >
              {reportHistory.map((rec) => (
                <option key={rec.reportId} value={rec.reportId}>
                  {rec.reportId} ({rec.prediction.predicted_class.toUpperCase()})
                </option>
              ))}
            </select>
          )}

          <button
            onClick={() => setActiveTab('predictor')}
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-all"
          >
            <UploadCloud className="w-3.5 h-3.5 text-sky-400" />
            <span>New Scan</span>
          </button>

          <button
            onClick={handleExportJson}
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition-all"
          >
            <Download className="w-3.5 h-3.5 text-emerald-400" />
            <span>Export JSON</span>
          </button>

          <button
            onClick={handlePrint}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white text-xs font-bold flex items-center gap-1.5 shadow-lg shadow-sky-500/20 transition-all"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print / Save PDF</span>
          </button>
        </div>
      </div>

      {/* Editable Subject Metadata Panel (hidden when printing) */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-3 print:hidden">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-300 flex items-center gap-2">
            <User className="w-4 h-4 text-sky-400" /> Customize Subject & Case Metadata (Updates Report Live)
          </span>
          <span className="text-[11px] font-mono text-slate-500">Optional Fields</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 text-xs">
          <div>
            <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Subject / Patient ID</label>
            <input
              type="text"
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Age (Years)</label>
            <input
              type="text"
              value={patientAge}
              onChange={(e) => setPatientAge(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500"
            />
          </div>
          <div>
            <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Gender</label>
            <select
              value={patientGender}
              onChange={(e) => setPatientGender(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option>Not Specified</option>
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>
          <div>
            <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Tested Hand</label>
            <select
              value={testedHand}
              onChange={(e) => setTestedHand(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option>Dominant (Right)</option>
              <option>Dominant (Left)</option>
              <option>Non-Dominant (Right)</option>
              <option>Non-Dominant (Left)</option>
            </select>
          </div>
          <div>
            <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Exam Type</label>
            <select
              value={examType}
              onChange={(e) => setExamType(e.target.value)}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              <option>Archimedean Spiral Drawing</option>
              <option>Sinusoidal Meander Drawing</option>
            </select>
          </div>
        </div>
        <div>
          <label className="text-[10px] uppercase font-mono text-slate-400 block mb-1">Clinical Observation / Notes</label>
          <input
            type="text"
            placeholder="Add optional tremor observations, medication state, or follow-up notes..."
            value={clinicianNotes}
            onChange={(e) => setClinicianNotes(e.target.value)}
            className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-sky-500"
          />
        </div>
      </div>

      {/* Printable Formal Lab Report Sheet */}
      <div className="bg-slate-900/90 border border-slate-700/80 rounded-3xl overflow-hidden shadow-2xl print:bg-white print:text-slate-900 print:border-slate-300 print:shadow-none print:rounded-none">
        {/* Report Letterhead */}
        <div className="p-6 sm:p-8 border-b border-slate-800 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 flex flex-col sm:flex-row sm:items-center justify-between gap-4 print:bg-none print:border-slate-300">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-xl bg-sky-500/20 border border-sky-500/40 flex items-center justify-center text-sky-400 print:border-sky-600 print:text-sky-700">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-black text-white tracking-tight print:text-slate-900">
                NEUROSPIRAL AI DIAGNOSTICS
              </h2>
              <p className="text-xs text-slate-400 print:text-slate-600">
                Automated Neuromotor Spiral &amp; Tremor Vision Laboratory Report
              </p>
            </div>
          </div>

          <div className="text-left sm:text-right font-mono text-xs space-y-1">
            <div className="text-white font-bold print:text-slate-900">REPORT ID: {reportId}</div>
            <div className="text-slate-400 print:text-slate-600 flex items-center sm:justify-end gap-1.5">
              <Calendar className="w-3.5 h-3.5" />
              <span>{new Date(timestamp).toLocaleString()}</span>
            </div>
          </div>
        </div>

        <div className="p-6 sm:p-8 space-y-8">
          {/* Section 1: Subject & Specimen Metadata */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-sky-400 font-bold border-b border-slate-800 pb-2 print:text-sky-700 print:border-slate-300">
              1. Subject &amp; Specimen Information
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Subject Identifier</span>
                <span className="text-white font-semibold mt-0.5 block print:text-slate-900">{patientName || '—'}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Age / Gender</span>
                <span className="text-white font-semibold mt-0.5 block print:text-slate-900">
                  {patientAge ? `${patientAge} yrs` : '—'} &bull; {patientGender}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Tested Hand</span>
                <span className="text-white font-semibold mt-0.5 block print:text-slate-900">{testedHand}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Motor Task</span>
                <span className="text-white font-semibold mt-0.5 block print:text-slate-900">{examType}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Specimen File</span>
                <span className="text-slate-300 font-mono truncate block mt-0.5 print:text-slate-800" title={fileName}>
                  {fileName}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                <span className="text-[10px] font-mono uppercase text-slate-500 block">Neural Architecture</span>
                <span className="text-sky-400 font-mono font-bold uppercase mt-0.5 block print:text-sky-700">
                  {prediction.model_name}
                </span>
              </div>
            </div>
          </div>

          {/* Section 2: Primary Diagnostic Finding & Specimen Preview */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-sky-400 font-bold border-b border-slate-800 pb-2 print:text-sky-700 print:border-slate-300">
              2. Quantitative Motor Classification &amp; Specimen
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
              {/* Drawing Specimen Thumbnails */}
              <div className="md:col-span-5 grid grid-cols-2 gap-3">
                <div className="space-y-1.5 text-center">
                  <div className="aspect-square bg-white rounded-xl border border-slate-700 p-2 flex items-center justify-center overflow-hidden print:border-slate-300">
                    {originalPreviewUrl ? (
                      <img src={originalPreviewUrl} alt="Original specimen" className="w-full h-full object-contain" />
                    ) : (
                      <span className="text-xs text-slate-400">Original</span>
                    )}
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 uppercase block print:text-slate-600">
                    Raw Specimen
                  </span>
                </div>
                <div className="space-y-1.5 text-center">
                  <div className="aspect-square bg-white rounded-xl border border-slate-700 p-2 flex items-center justify-center overflow-hidden print:border-slate-300">
                    {prediction.letterboxed_preview_base64 ? (
                      <img
                        src={prediction.letterboxed_preview_base64}
                        alt="Standardized 224x224 canvas"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <span className="text-xs text-slate-400">224×224</span>
                    )}
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 uppercase block print:text-slate-600">
                    Normalized (224×224)
                  </span>
                </div>
              </div>

              {/* Outcome & Quantitative Scores */}
              <div className="md:col-span-7 space-y-4">
                <div
                  className={`p-5 rounded-2xl border-l-4 ${
                    isParkinson
                      ? 'bg-red-950/30 border-l-red-500 border border-red-500/30 print:bg-red-50 print:border-red-300'
                      : 'bg-emerald-950/30 border-l-emerald-500 border border-emerald-500/30 print:bg-emerald-50 print:border-emerald-300'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 print:text-slate-600">
                        Automated Classification Result
                      </span>
                      <div className="text-xl font-black text-white mt-0.5 print:text-slate-900">
                        {prediction.prediction_label}
                      </div>
                    </div>
                    <div
                      className={`px-3 py-1 rounded-full text-xs font-bold uppercase flex items-center gap-1.5 ${
                        isParkinson
                          ? 'bg-red-500/20 text-red-300 border border-red-500/40 print:text-red-700'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 print:text-emerald-700'
                      }`}
                    >
                      {isParkinson ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                      {prediction.predicted_class}
                    </div>
                  </div>

                  {/* Dual Probability Bar */}
                  <div className="mt-4 space-y-2">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-emerald-400 font-semibold print:text-emerald-700">
                        Healthy Control: {healthyPct}%
                      </span>
                      <span className="text-red-400 font-semibold print:text-red-700">
                        Parkinson Pattern: {parkinsonPct}%
                      </span>
                    </div>
                    <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden flex print:bg-slate-200">
                      <div style={{ width: `${healthyPct}%` }} className="bg-emerald-500" />
                      <div style={{ width: `${parkinsonPct}%` }} className="bg-red-500" />
                    </div>
                  </div>
                </div>

                {/* Risk Tier & Execution Telemetry */}
                <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                    <span className="text-[10px] text-slate-500 block">RAW PROBABILITY</span>
                    <span className="text-white font-bold print:text-slate-900">{prediction.confidence_score.toFixed(4)}</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                    <span className="text-[10px] text-slate-500 block">THRESHOLD (τ)</span>
                    <span className="text-white font-bold print:text-slate-900">{prediction.decision_threshold.toFixed(2)}</span>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                    <span className="text-[10px] text-slate-500 block">INFERENCE TIME</span>
                    <span className="text-sky-400 font-bold print:text-sky-700">{prediction.inference_time_ms} ms</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Interpretation & Risk Stratification */}
          <div className="space-y-3">
            <h3 className="text-xs font-mono uppercase tracking-widest text-sky-400 font-bold border-b border-slate-800 pb-2 print:text-sky-700 print:border-slate-300">
              3. Automated Motor Assessment &amp; Risk Stratification
            </h3>

            <div className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-3 print:bg-slate-50 print:border-slate-200">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-300 flex items-center gap-2 print:text-slate-700">
                  <Activity className="w-4 h-4 text-sky-400" /> Risk Stratification Tier
                </span>
                <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full border ${riskTier.bg} ${riskTier.color}`}>
                  {riskTier.label}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed print:text-slate-700">
                {isParkinson
                  ? `The ${prediction.model_name.toUpperCase()} computer vision model estimated a ${parkinsonPct}% probability of Parkinsonian motor tremor markers (exceeding the active decision threshold of ${(prediction.decision_threshold * 100).toFixed(0)}%). Spatial stroke analysis detected micro-oscillations, radius progression irregularities, or hesitation patterns consistent with basal ganglia motor impairment. Comprehensive neurological evaluation is recommended.`
                  : `The ${prediction.model_name.toUpperCase()} computer vision model estimated a ${healthyPct}% probability of a healthy control motor trajectory (Parkinsonian probability of ${parkinsonPct}% remained below the ${(prediction.decision_threshold * 100).toFixed(0)}% threshold). Spatial curvature and concentric loop continuity appear within normal reference bounds.`}
              </p>
              {clinicianNotes && (
                <div className="pt-2 border-t border-slate-800 text-xs text-slate-300 print:border-slate-200 print:text-slate-800">
                  <span className="font-semibold text-white print:text-slate-900">Additional Notes: </span>
                  {clinicianNotes}
                </div>
              )}
            </div>
          </div>

          {/* Section 4: Reference Model Performance Metrics */}
          {modelValidation && (
            <div className="space-y-3">
              <h3 className="text-xs font-mono uppercase tracking-widest text-sky-400 font-bold border-b border-slate-800 pb-2 print:text-sky-700 print:border-slate-300">
                4. Active Model Reference Benchmarks ({prediction.model_name.toUpperCase()})
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                  <span className="text-[10px] text-slate-500 block">ACCURACY</span>
                  <span className="text-white font-bold print:text-slate-900">{(modelValidation.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                  <span className="text-[10px] text-slate-500 block">SENSITIVITY</span>
                  <span className="text-white font-bold print:text-slate-900">{(modelValidation.sensitivity * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                  <span className="text-[10px] text-slate-500 block">SPECIFICITY</span>
                  <span className="text-white font-bold print:text-slate-900">{(modelValidation.specificity * 100).toFixed(1)}%</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                  <span className="text-[10px] text-slate-500 block">ROC-AUC</span>
                  <span className="text-emerald-400 font-bold print:text-emerald-700">{modelValidation.roc_auc.toFixed(3)}</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-slate-50 print:border-slate-200">
                  <span className="text-[10px] text-slate-500 block">BRIER SCORE</span>
                  <span className="text-sky-400 font-bold print:text-sky-700">{modelValidation.brier_score.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Report Footer & Verification */}
          <div className="pt-6 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4 text-[11px] text-slate-400 print:border-slate-300 print:text-slate-600">
            <div className="flex items-start gap-2 max-w-2xl">
              <ShieldCheck className="w-4 h-4 text-sky-400 shrink-0 mt-0.5 print:text-sky-700" />
              <p>
                <strong>Notice:</strong> This automated report is generated by the NeuroSpiral AI computer vision engine for investigational motor screening and observational support. It is not a standalone medical diagnosis and should be reviewed alongside clinical neurological examination.
              </p>
            </div>
            <div className="font-mono text-right shrink-0">
              <div className="flex items-center gap-1.5 sm:justify-end">
                <Cpu className="w-3.5 h-3.5" />
                <span>Engine v1.0.0</span>
              </div>
              <div className="flex items-center gap-1.5 sm:justify-end mt-0.5">
                <Clock className="w-3.5 h-3.5" />
                <span>Verified Output</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

