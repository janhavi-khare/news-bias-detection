import React from "react";
import { CheckCircle2, AlertOctagon, RotateCcw, Info, Layers } from "lucide-react";
import { ExplanationCard } from "./ExplanationCard";
import { SentenceBreakdown } from "./SentenceBreakdown";

export function BiasCard({
  result,
  onReset,
  isLoadingExplanation,
  onRequestExplanation,
  onExplainSentence,
  explainingSentenceIdx
}) {
  if (!result) return null;

  const isBiased = result.predicted_label === "Biased";
  const nonBiasedPct = (result.probabilities["Non-biased"] * 100).toFixed(1);
  const biasedPct = (result.probabilities["Biased"] * 100).toFixed(1);
  const summary = result.article_summary;
  const isMultiSentence = summary && summary.total_sentences > 1;

  return (
    <div className="card">
      <div className="result-header">
        <div>
          <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", display: "block", marginBottom: "0.25rem" }}>
            {isMultiSentence ? "Article-Level Evaluation" : "Model Classification"}
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

      {/* Article Summary Statistics */}
      {isMultiSentence && (
        <div
          style={{
            background: "rgba(99, 102, 241, 0.07)",
            border: "1px solid rgba(99, 102, 241, 0.25)",
            borderRadius: "0.75rem",
            padding: "1rem 1.25rem",
            marginBottom: "1.25rem"
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.6rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontWeight: 700, fontSize: "0.95rem" }}>
              <Layers size={16} color="#818cf8" />
              <span>Sentence-Level Aggregation Summary</span>
            </div>
            <span
              style={{
                fontSize: "0.78rem",
                fontWeight: 600,
                padding: "0.2rem 0.6rem",
                borderRadius: "9999px",
                background: summary.bias_intensity === "High" ? "var(--biased-bg)" : "rgba(255, 255, 255, 0.08)",
                color: summary.bias_intensity === "High" ? "#f87171" : "var(--text-secondary)",
                border: "1px solid var(--border-color)"
              }}
            >
              Bias Density: {summary.bias_intensity}
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem", marginBottom: "0.75rem" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "0.5rem 0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Total Sentences</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800 }}>{summary.total_sentences}</div>
            </div>
            <div style={{ background: "var(--bg-subtle)", padding: "0.5rem 0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Biased Sentences</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: summary.biased_sentences_count > 0 ? "#f87171" : "inherit" }}>
                {summary.biased_sentences_count} ({summary.biased_sentences_percentage}%)
              </div>
            </div>
            <div style={{ background: "var(--bg-subtle)", padding: "0.5rem 0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mean Biased Prob</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, fontFamily: "var(--font-mono)" }}>
                {(summary.mean_biased_probability * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: "0.4rem", alignItems: "flex-start", fontSize: "0.82rem", color: "var(--text-secondary)" }}>
            <Info size={15} style={{ flexShrink: 0, marginTop: "0.15rem", color: "#818cf8" }} />
            <span>{summary.aggregation_method}</span>
          </div>
        </div>
      )}

      {/* Probabilities Meter */}
      <div className="probabilities-container">
        <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.75rem", fontWeight: 600 }}>
          {isMultiSentence ? "Overall Article Probability Distribution" : "Class Probabilities & Confidence"}
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

      {/* Top-Level or Most Biased Sentence SHAP Explanation */}
      <ExplanationCard
        explanation={result.explanation}
        isLoadingExplanation={isLoadingExplanation}
        onRequestExplanation={onRequestExplanation}
      />

      {/* Sentence Breakdown for multi-sentence articles */}
      <SentenceBreakdown
        sentences={result.sentences}
        onExplainSentence={onExplainSentence}
        explainingSentenceIdx={explainingSentenceIdx}
      />
    </div>
  );
}
