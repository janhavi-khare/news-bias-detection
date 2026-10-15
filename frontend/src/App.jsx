import React, { useState, useEffect } from "react";
import {
  Newspaper,
  Send,
  AlertCircle,
  BarChart2,
  CheckCircle,
  XCircle,
  Sparkles
} from "lucide-react";
import { BiasCard } from "./components/BiasCard";
import { LimitationsNotice } from "./components/LimitationsNotice";
import { MetricsModal } from "./components/MetricsModal";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const SAMPLE_TEXTS = [
  {
    label: "Neutral Reporting",
    text: "The government announced a new policy today, according to officials."
  },
  {
    label: "Strongly Biased",
    text: "The corrupt administration recklessly betrayed citizens with its disastrous decision."
  },
  {
    label: "Policy Debate",
    text: "Government policy sparks debate among citizens over the country's economic future."
  }
];

export default function App() {
  const [inputText, setInputText] = useState("");
  const [includeExplanation, setIncludeExplanation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [apiHealth, setApiHealth] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);

  // Check backend health on initial load
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          setApiHealth({ online: true, ...data });
        } else {
          setApiHealth({ online: false, error: "Health check returned non-200" });
        }
      } catch (err) {
        setApiHealth({ online: false, error: "Cannot connect to backend" });
      }
    }
    checkHealth();
  }, []);

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    setErrorMessage("");

    const trimmed = inputText.trim();
    if (!trimmed) {
      setErrorMessage("Please enter some news text to analyze.");
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          include_explanation: includeExplanation
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setErrorMessage(err.message || "Failed to analyze text. Ensure backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRequestExplanation = async () => {
    if (!result) return;
    setIsLoadingExplanation(true);
    try {
      const response = await fetch(`${API_BASE_URL}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: result.text,
          top_n: 10
        })
      });

      if (!response.ok) {
        throw new Error("Failed to generate explanation from backend.");
      }

      const explData = await response.json();
      setResult((prev) => ({
        ...prev,
        explanation: explData
      }));
    } catch (err) {
      setResult((prev) => ({
        ...prev,
        explanation: {
          success: false,
          error: err.message,
          top_words: [],
          token_highlights: [],
          disclaimer: "Explanation request failed."
        }
      }));
    } finally {
      setIsLoadingExplanation(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setInputText("");
    setErrorMessage("");
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-badge">
          <Newspaper size={14} />
          <span>Academic ML Research Demo</span>
        </div>
        <h1 className="app-title">News Bias Detection</h1>
        <p className="app-subtitle">
          Detect linguistic and framing bias in news articles using a fine-tuned RoBERTa binary
          classifier with SHAP explainability.
        </p>

        {/* Backend status indicator */}
        <div style={{ marginTop: "1rem", display: "inline-flex", alignItems: "center", gap: "0.5rem", fontSize: "0.82rem" }}>
          {apiHealth === null ? (
            <span style={{ color: "var(--text-muted)" }}>Connecting to backend...</span>
          ) : apiHealth.online ? (
            <span style={{ color: "#34d399", display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <CheckCircle size={14} /> Backend online (Model: RoBERTa on {apiHealth.device})
            </span>
          ) : (
            <span style={{ color: "#f87171", display: "flex", alignItems: "center", gap: "0.35rem" }}>
              <XCircle size={14} /> Backend offline ({apiHealth.error})
            </span>
          )}
          <span style={{ color: "var(--text-muted)" }}>•</span>
          <button
            onClick={() => setShowMetrics(true)}
            style={{
              background: "none",
              border: "none",
              color: "#818cf8",
              cursor: "pointer",
              fontSize: "0.82rem",
              textDecoration: "underline"
            }}
          >
            View Benchmark Metrics
          </button>
        </div>
      </header>

      {/* Main Analysis Input Card */}
      <div className="card">
        <h2 className="card-title">
          <Newspaper size={20} color="#818cf8" />
          <span>Input News Text</span>
        </h2>

        {/* Sample text buttons */}
        <div className="sample-buttons">
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", alignSelf: "center", marginRight: "0.25rem" }}>
            Try a sample:
          </span>
          {SAMPLE_TEXTS.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              className="sample-btn"
              onClick={() => {
                setInputText(sample.text);
                setErrorMessage("");
              }}
            >
              {sample.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleAnalyze}>
          <div className="textarea-container">
            <textarea
              className="news-textarea"
              placeholder="Paste a news headline, paragraph, or article text here to analyze for bias..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={5}
              disabled={isLoading}
            />
            <div className="textarea-footer">
              <span>{inputText.length} characters</span>
              <span>RoBERTa tokenizer truncation limit: 512 tokens</span>
            </div>
          </div>

          {errorMessage && (
            <div className="alert alert-error">
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <div>{errorMessage}</div>
            </div>
          )}

          <div className="actions-row">
            <label className="toggle-label">
              <input
                type="checkbox"
                className="toggle-checkbox"
                checked={includeExplanation}
                onChange={(e) => setIncludeExplanation(e.target.checked)}
                disabled={isLoading}
              />
              <Sparkles size={16} color="#818cf8" />
              <span>Include SHAP Word Attributions (~3s)</span>
            </label>

            <button
              type="submit"
              className="primary-btn"
              disabled={isLoading || !inputText.trim()}
            >
              {isLoading ? (
                <>
                  <div className="spinner" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Send size={16} />
                  <span>Analyze Text</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Result Card */}
      <BiasCard
        result={result}
        onReset={handleReset}
        isLoadingExplanation={isLoadingExplanation}
        onRequestExplanation={handleRequestExplanation}
      />

      {/* Limitations note */}
      <LimitationsNotice />

      {/* Benchmark Metrics Modal */}
      <MetricsModal isOpen={showMetrics} onClose={() => setShowMetrics(false)} />
    </div>
  );
}
