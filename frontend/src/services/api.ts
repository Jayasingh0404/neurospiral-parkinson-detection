import { HealthResponse, ModelsListResponse, PredictionResult, ExplanationResult, MetricsData } from '../types';

const API_BASE = '/api/v1';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelsListResponse> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to load models: ${res.statusText}`);
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsData> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch evaluation metrics: ${res.statusText}`);
  return res.json();
}

export async function predictSpiral(
  file: File,
  modelName: string = 'custom_cnn',
  threshold: number = 0.5
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);
  formData.append('threshold', threshold.toString());

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
  }
  return res.json();
}

export async function explainSpiral(
  file: File,
  modelName: string = 'custom_cnn'
): Promise<ExplanationResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model_name', modelName);

  const res = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Explainability analysis failed with status ${res.status}`);
  }
  return res.json();
}

export const apiService = {
  checkHealth: fetchHealth,
  fetchHealth,
  fetchModels,
  fetchMetrics,
  predictSpiral,
  explainSpiral,
};

