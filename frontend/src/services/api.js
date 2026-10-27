/**
 * API Service for News Bias Detection System
 * Handles communication with the FastAPI backend.
 */

const API_BASE = "";

/**
 * Check backend health and model loading status
 */
export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return await res.json();
}

/**
 * Get detailed status of all 5 individual models and ensemble configuration
 */
export async function fetchModelsStatus() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`Failed to fetch models status: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Predict bias for text using all 5 models and weighted soft voting ensemble
 */
export async function predictBias(text, includeExplanation = false) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      include_explanation: includeExplanation
    })
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
  }

  return await res.json();
}

/**
 * Generate SHAP explanation for a specific text or sentence
 */
export async function fetchExplanation(text, topN = 10, sentenceIndex = null) {
  const body = {
    text,
    top_n: topN
  };
  if (sentenceIndex !== null) {
    body.sentence_index = sentenceIndex;
  }

  const res = await fetch(`${API_BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Explanation failed with status ${res.status}`);
  }

  return await res.json();
}

/**
 * Fetch recorded historical benchmark evaluation metrics
 */
export async function fetchMetrics() {
  const res = await fetch(`${API_BASE}/api/metrics`);
  if (!res.ok) {
    throw new Error(`Failed to fetch metrics: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Fetch available test confusion matrices
 */
export async function fetchConfusionMatrices() {
  const res = await fetch(`${API_BASE}/api/confusion-matrices`);
  if (!res.ok) {
    throw new Error(`Failed to fetch confusion matrices: ${res.statusText}`);
  }
  return await res.json();
}

/**
 * Get URL to download comparison CSV
 */
export function getComparisonCsvUrl() {
  return `${API_BASE}/api/comparison-csv`;
}
