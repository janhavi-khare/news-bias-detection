import React from "react";
import { CheckCircle2, AlertOctagon, RotateCcw, Info, Layers, Award, Sparkles } from "lucide-react";
import { ExplanationCard } from "./ExplanationCard";
import { SentenceBreakdown } from "./SentenceBreakdown";
import { ModelComparisonCard } from "./ModelComparisonCard";

export function BiasCard({
  result,
  onReset,
  isLoadingExplanation,
  onRequestExplanation,
  onExplainSentence,
  explainingSentenceIdx
}) {
  if (!result) return null;

  const ensemble = result.ensemble || {
    label: result.predicted_label,
    probability_biased: result.biased_probability,
    probability_non_biased: result.non_biased_probability,
    confidence: result.confidence,
    threshold: 0.40,
    weights: { linear_svm: 1, logistic_regression: 1, naive_bayes: 1, roberta: 2, deberta: 2 }
  };

  const isBiased = ensemble.label === "Biased";
  const pBiased = ensemble.probability_biased !== undefined ? ensemble.probability_biased : 0.5;
  const pNonBiased = ensemble.probability_non_biased !== undefined ? ensemble.probability_non_biased : 0.5;
  const biasedPct = (pBiased * 100).toFixed(1);
  const nonBiasedPct = (pNonBiased * 100).toFixed(1);
  const thresholdPct = ((ensemble.threshold || 0.40) * 100).toFixed(0);

  const summary = result.article_summary;
  const isMultiSentence = summary && summary.total_sentences > 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Primary Ensemble Decision Card */}
      <div className="card">
        <div className="result-header">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.25rem" }}>
              <Sparkles size={16} color="#818cf8" />
              <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontWeight: 600 }}>
                Weighted Soft-Voting Ensemble Decision (5 Models)
              </span>
            </div>
            <div className={`prediction-badge ${isBiased ? "badge-biased" : "badge-nonbiased"}`}>
              {isBiased ? <AlertOctagon size={26} /> : <CheckCircle2 size={26} />}
              <span>{ensemble.label}</span>
            </div>
          </div>

          <button className="secondary-btn" onClick={onReset} type="button">
            <RotateCcw size={16} />
            <span>Analyze New Text</span>
          </button>
        </div>

        {/* Ensemble Weighted Probability & Threshold Meter */}
        <div
          style={{
            background: "rgba(15, 23, 42, 0.5)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.75rem",
            padding: "1.25rem",
            marginBottom: "1.25rem"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
            <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--text-primary)" }}>
              Ensemble Biased Probability: <span style={{ color: isBiased ? "#f87171" : "#34d399", fontFamily: "var(--font-mono)" }}>{biasedPct}%</span>
            </span>
            <span
              style={{
                fontSize: "0.78rem",
                padding: "0.2rem 0.55rem",
                borderRadius: "0.35rem",
                background: isBiased ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
                color: isBiased ? "#fca5a5" : "#a7f3d0",
                border: "1px solid var(--border-color)",
                fontWeight: 600
              }}
            >
              Threshold: {thresholdPct}% (P ≥ 0.40 → Biased)
            </span>
          </div>

          {/* Visual Track with Threshold Marker */}
          <div style={{ position: "relative", marginBottom: "1rem" }}>
            <div
              style={{
                height: "12px",
                background: "rgba(255, 255, 255, 0.08)",
                borderRadius: "6px",
                overflow: "hidden"
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${biasedPct}%`,
                  background: isBiased
                    ? "linear-gradient(90deg, #f59e0b, #ef4444)"
                    : "linear-gradient(90deg, #10b981, #6366f1)",
                  borderRadius: "6px",
                  transition: "width 0.5s ease"
                }}
              />
            </div>

            {/* Threshold Vertical Line */}
            <div
              style={{
                position: "absolute",
                top: "-4px",
                left: `${thresholdPct}%`,
                width: "2px",
                height: "20px",
                background: "#facc15",
                boxShadow: "0 0 6px rgba(250, 204, 21, 0.7)"
              }}
              title="Decision Threshold (0.40)"
            />
          </div>

          {/* Formula and Explanatory Details */}
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
              background: "var(--bg-subtle)",
              borderRadius: "0.5rem",
              padding: "0.65rem 0.85rem",
              border: "1px solid var(--border-color)",
              lineHeight: 1.5
            }}
          >
            <div>
              <strong>Ensemble Formula:</strong> P_ensemble(biased) = (1×P_SVM + 1×P_LR + 1×P_NB + 2×P_RoBERTa + 2×P_DeBERTa) / 7.0
            </div>
            <div style={{ color: "var(--text-muted)", marginTop: "0.3rem" }}>
              {ensemble.explanation || `Classified as ${ensemble.label} because P(biased) = ${biasedPct}% is ${isBiased ? "≥" : "<"} the 40.0% decision threshold.`}
            </div>
          </div>
        </div>

        {/* Article Summary Statistics (If Multi-Sentence) */}
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
                <span>Sentence-Level Aggregation Breakdown</span>
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

      {/* Five Individual Models Comparison Section */}
      <ModelComparisonCard
        individualPredictions={result.individual_predictions}
        ensemble={result.ensemble}
      />
    </div>
  );
}
