import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  Sparkles, 
  FileCheck2, 
  CheckCircle2,
  AlertCircle, 
  Sliders, 
  Download, 
  RotateCcw,
  Layers
} from 'lucide-react';
import { predictSpiral } from '../services/api';
import { PredictionResult, LabReportRecord } from '../types';

interface PredictorPageProps {
  onSaveReport?: (record: LabReportRecord) => void;
  onOpenReport?: () => void;
}

export const PredictorPage: React.FC<PredictorPageProps> = ({ onSaveReport, onOpenReport }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modelName, setModelName] = useState<string>('vgg16');
  const [threshold, setThreshold] = useState<number>(0.5);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
      setError('Please select a valid image file (JPEG or PNG).');
      return;
    }
    setError(null);
    setSelectedFile(file);
    setPrediction(null);

    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictSpiral(selectedFile, modelName, threshold);
      setPrediction(res);
      if (onSaveReport) {
        const randomCode = Math.floor(1000 + Math.random() * 9000);
        onSaveReport({
          reportId: `NS-${new Date().getFullYear()}-${randomCode}`,
          timestamp: new Date().toISOString(),
          fileName: selectedFile.name,
          originalPreviewUrl: previewUrl,
          prediction: res,
        });
      }
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Ensure model is trained and API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (!prediction) return;
    const reportData = {
      timestamp: new Date().toISOString(),
      prediction,
    };
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neurospiral_inference_${prediction.predicted_class}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-10 max-w-6xl mx-auto">
      {/* Title */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/30 text-xs text-sky-400 font-mono">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Interactive Computer Vision Laboratory</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Spiral Drawing Inference &amp; Diagnostics
        </h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto">
          Upload or drag-and-drop a digitized spiral or meander drawing to run deep learning classification and generate a structured lab report.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/40 text-red-200 text-xs flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-red-300">Operational Error</p>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Main Grid: Upload & Settings Left, Prediction & XAI Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Upload Card */}
          <div className="glass-panel p-6 rounded-2xl space-y-4">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center justify-between">
              <span>1. Upload Drawing</span>
              {selectedFile && (
                <button 
                  onClick={resetAll}
                  className="text-xs text-slate-400 hover:text-rose-400 flex items-center gap-1"
                >
                  <RotateCcw className="w-3 h-3" /> Reset
                </button>
              )}
            </h2>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                previewUrl 
                  ? 'border-sky-500/50 bg-sky-950/10' 
                  : 'border-slate-700 hover:border-slate-500 bg-slate-900/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />

              {previewUrl ? (
                <div className="space-y-3">
                  <div className="w-48 h-48 mx-auto rounded-lg overflow-hidden border border-slate-700 bg-white shadow-md flex items-center justify-center">
                    <img 
                      src={previewUrl} 
                      alt="Uploaded preview" 
                      className="max-w-full max-h-full object-contain"
                    />
                  </div>
                  <div className="text-xs text-slate-300 font-mono truncate px-4">
                    {selectedFile?.name}
                  </div>
                  <p className="text-[11px] text-sky-400 hover:underline">Click to change drawing</p>
                </div>
              ) : (
                <div className="space-y-3 py-4">
                  <div className="w-12 h-12 rounded-xl bg-slate-800 text-sky-400 mx-auto flex items-center justify-center">
                    <UploadCloud className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-slate-200">
                      Drag and drop spiral drawing here
                    </p>
                    <p className="text-[11px] text-slate-500 mt-1">
                      Supports JPEG, PNG up to 10MB
                    </p>
                  </div>
                  <span className="inline-block text-[11px] font-medium px-3 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    Browse Files
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Model Selection & Parameters */}
          <div className="glass-panel p-6 rounded-2xl space-y-5">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300">
              2. Inference Architecture
            </h2>

            {/* Model Radio selector */}
            <div className="space-y-2">
              <label className="text-xs text-slate-400 font-medium">Select Model Topology:</label>
              <div className="grid grid-cols-1 gap-2">
                {[
                  { id: 'custom_cnn', name: 'Custom CNN', desc: 'Trained from scratch (3 blocks)' },
                  { id: 'vgg16', name: 'VGG16 Transfer', desc: 'Pretrained + Block5 Fine-Tuning' },
                  { id: 'resnet50', name: 'ResNet50 Transfer', desc: 'Residual base + Frozen BatchNorm' }
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setModelName(m.id)}
                    className={`p-3 rounded-xl text-left border transition-all flex items-center justify-between ${
                      modelName === m.id
                        ? 'bg-sky-500/15 border-sky-500/50 text-white shadow-sm'
                        : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="text-xs font-bold text-slate-200">{m.name}</div>
                      <div className="text-[10px] text-slate-400">{m.desc}</div>
                    </div>
                    {modelName === m.id && <CheckCircle2 className="w-4 h-4 text-sky-400 shrink-0" />}
                  </button>
                ))}
              </div>
            </div>

            {/* Decision Threshold slider */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center gap-1.5 font-medium">
                  <Sliders className="w-3.5 h-3.5 text-sky-400" />
                  Decision Threshold (τ)
                </span>
                <span className="font-mono text-sky-400 font-bold">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-sky-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0.10 (High Sensitivity)</span>
                <span>0.50 (Default)</span>
                <span>0.90 (High Specificity)</span>
              </div>
            </div>

            {/* Submit Button */}
            <button
              disabled={!selectedFile || loading}
              onClick={handlePredict}
              className={`w-full py-3.5 px-4 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${
                !selectedFile || loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                  : 'bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white shadow-lg shadow-sky-500/25 active:scale-[0.99]'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Preprocessing & Inferencing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Execute Model Prediction</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Prediction Results & XAI (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {prediction ? (
            <div className="space-y-6">
              {/* Primary Prediction Result Card */}
              <div className={`glass-panel p-6 rounded-2xl border-l-4 space-y-5 ${
                prediction.predicted_class === 'parkinson' 
                  ? 'border-l-red-500 glow-red' 
                  : 'border-l-emerald-500 glow-emerald'
              }`}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400">
                      Model Classification Outcome
                    </span>
                    <h3 className="text-xl font-black text-white mt-0.5">
                      {prediction.prediction_label}
                    </h3>
                  </div>

                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border self-start ${
                    prediction.predicted_class === 'parkinson'
                      ? 'bg-red-500/20 text-red-300 border-red-500/40'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  }`}>
                    {prediction.predicted_class.toUpperCase()}
                  </span>
                </div>

                {/* Probability meters */}
                <div className="space-y-3">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-emerald-400">
                      Healthy Probability: {(prediction.probabilities.healthy * 100).toFixed(1)}%
                    </span>
                    <span className="text-red-400">
                      Parkinson Probability: {(prediction.probabilities.parkinson * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden flex">
                    <div 
                      style={{ width: `${prediction.probabilities.healthy * 100}%` }}
                      className="bg-emerald-500 transition-all duration-500" 
                    />
                    <div 
                      style={{ width: `${prediction.probabilities.parkinson * 100}%` }}
                      className="bg-red-500 transition-all duration-500" 
                    />
                  </div>
                </div>

                {/* Metadata tags */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono pt-2">
                  <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">RAW SCORE</span>
                    <span className="text-slate-200 font-bold">{prediction.confidence_score.toFixed(4)}</span>
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">THRESHOLD</span>
                    <span className="text-slate-200 font-bold">τ = {prediction.decision_threshold.toFixed(2)}</span>
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">LATENCY</span>
                    <span className="text-sky-400 font-bold">{prediction.inference_time_ms} ms</span>
                  </div>
                  <div className="bg-slate-900/60 p-2.5 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">ARCHITECTURE</span>
                    <span className="text-slate-200 font-bold uppercase">{prediction.model_name}</span>
                  </div>
                </div>

                {/* Preprocessing verification preview */}
                {prediction.letterboxed_preview_base64 && (
                  <div className="pt-2 border-t border-slate-800 flex items-center gap-3 text-xs text-slate-400">
                    <img 
                      src={prediction.letterboxed_preview_base64} 
                      alt="Letterboxed preview" 
                      className="w-10 h-10 rounded border border-slate-700 bg-white object-contain"
                    />
                    <div>
                      <span className="font-semibold text-slate-200 block">Aspect-Ratio Preserved Canvas (224×224)</span>
                      <span className="text-[11px] text-slate-400">White letterbox padding applied to prevent stroke curvature distortion.</span>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex flex-wrap items-center gap-3 pt-2">
                  {onOpenReport && (
                    <button
                      onClick={onOpenReport}
                      className="flex-1 py-2.5 px-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 transition-all"
                    >
                      <FileCheck2 className="w-4 h-4" />
                      <span>View Full Lab Report</span>
                    </button>
                  )}

                  <button
                    onClick={handleDownloadReport}
                    className="py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 font-medium text-xs flex items-center gap-2 transition-all"
                  >
                    <Download className="w-4 h-4" />
                    <span>Export JSON</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel p-12 rounded-2xl text-center space-y-4 border border-dashed border-slate-800">
              <div className="w-14 h-14 rounded-2xl bg-slate-800/80 text-slate-500 mx-auto flex items-center justify-center">
                <FileImage className="w-7 h-7" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-semibold text-slate-300">Awaiting Spiral Drawing Upload</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Select or drop a drawing exam on the left panel and click "Execute Model Prediction" to initiate analysis.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

