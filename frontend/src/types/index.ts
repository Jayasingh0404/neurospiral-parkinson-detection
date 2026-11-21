export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  models_loaded: Record<string, boolean>;
  system_device: string;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  default_model: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  description: string;
  input_shape: number[];
  is_trained: boolean;
  total_params?: number;
  trainable_params?: number;
  model_size_mb?: number;
  accuracy?: number;
  roc_auc?: number;
}

export interface PredictionResult {
  model_name: string;
  predicted_class: 'healthy' | 'parkinson';
  prediction_label: string;
  confidence_score: number;
  decision_threshold: number;
  probabilities: {
    healthy: number;
    parkinson: number;
  };
  inference_time_ms: number;
  letterboxed_preview_base64?: string;
  medical_disclaimer: string;
}

export interface ExplanationResult {
  model_name: string;
  predicted_class: 'healthy' | 'parkinson';
  confidence_score: number;
  target_layer_name: string;
  original_image_base64: string;
  heatmap_image_base64: string;
  overlay_image_base64: string;
  interpretation: string;
  limitations: string;
  medical_disclaimer: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  balanced_accuracy: number;
  roc_auc: number;
  pr_auc: number;
  mcc: number;
  brier_score: number;
  confusion_matrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    matrix: number[][];
  };
  threshold: number;
  sample_count: number;
  total_params?: number;
  trainable_params?: number;
  train_time_sec?: number;
  inference_latency_ms?: number;
  model_size_mb?: number;
  roc_curve: {
    fpr: number[];
    tpr: number[];
  };
  pr_curve: {
    recall: number[];
    precision: number[];
  };
}

export interface MetricsData {
  models: Record<string, ModelMetrics>;
  subject_independent_summary?: {
    custom_cnn?: {
      mean_accuracy: number;
      std_accuracy: number;
      mean_recall: number;
      mean_specificity: number;
      mean_roc_auc: number;
    };
  };
  dataset_audit?: any;
}

export interface LabReportRecord {
  reportId: string;
  timestamp: string;
  fileName: string;
  originalPreviewUrl: string | null;
  prediction: PredictionResult;
}
