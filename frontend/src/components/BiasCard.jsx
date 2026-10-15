import React from "react";
import { CheckCircle2, AlertOctagon, RotateCcw } from "lucide-react";
import { ExplanationCard } from "./ExplanationCard";

export function BiasCard({
  result,
  onReset,
  isLoadingExplanation,
  onRequestExplanation
}) {
  if (!result) return null;

  const isBiased = result.predicted_label === "Biased";
  const nonBiasedPct = (result.probabilities["Non-biased"] * 100).toFixed(1);
  const biasedPct = (result.probabilities["Biased"] * 100).toFixed(1);

  return (
    <div className="card">
      <div className="result-header">
        <div>
          <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", display: "block", marginBottom: "0.25rem" }}>
            Model Classification
          </span>
          <div className={`prediction-badge ${isBiased ? "badge-biased" : "badge-nonbiased"}`}>
            {isBiased ? <AlertOctagon size={24} /> : <CheckCircle2 size={24} />}
            <span>{result.predicted_label}</span>
          </div>
        </div>

        <button className="secondary-btn" onClick={onReset} type="button">
          <RotateCcw size={16} />
          <span>Analyze New Text</span>
        </button>
      </div>

      <div className="probabilities-container">
        <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.75rem", fontWeight: 600 }}>
          Class Probabilities & Confidence
        </div>

        {/* Non-biased row */}
        <div className="prob-row">
          <div className="prob-meta">
            <span style={{ color: "#34d399" }}>Non-biased</span>
            <span style={{ fontFamily: "var(--font-mono)" }}>{nonBiasedPct}%</span>
          </div>
          <div className="prob-bar-track">
            <div
              className="prob-bar-fill fill-nonbiased"
              style={{ width: `${nonBiasedPct}%` }}
            />
          </div>
        </div>

        {/* Biased row */}
        <div className="prob-row">
          <div className="prob-meta">
            <span style={{ color: "#f87171" }}>Biased</span>
            <span style={{ fontFamily: "var(--font-mono)" }}>{biasedPct}%</span>
          </div>
          <div className="prob-bar-track">
            <div
              className="prob-bar-fill fill-biased"
              style={{ width: `${biasedPct}%` }}
            />
          </div>
        </div>

        <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.75rem" }}>
          * Confidence scores denote the model softmax probability distribution, not a guarantee of factual correctness.
        </div>
      </div>

      <ExplanationCard
        explanation={result.explanation}
        isLoadingExplanation={isLoadingExplanation}
        onRequestExplanation={onRequestExplanation}
      />
    </div>
  );
}
