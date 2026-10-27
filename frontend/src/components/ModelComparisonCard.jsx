import React from "react";
import { Cpu, Scale, CheckCircle2, AlertOctagon, HelpCircle } from "lucide-react";

export function ModelComparisonCard({ individualPredictions, ensemble }) {
  if (!individualPredictions || Object.keys(individualPredictions).length === 0) {
    return null;
  }

  const modelMetadata = [
    {
      key: "linear_svm",
      displayName: "Linear SVM",
      badge: "Traditional Baseline",
      type: "TF-IDF + Calibrated LinearSVC",
      weight: ensemble?.weights?.linear_svm || 1.0,
      description: "Linear support vector classifier with Sigmoid probability calibration (Platt scaling)."
    },
    {
      key: "logistic_regression",
      displayName: "Logistic Regression",
      badge: "Traditional Baseline",
      type: "TF-IDF + LogisticRegression",
      weight: ensemble?.weights?.logistic_regression || 1.0,
      description: "L2-regularized logistic regression over sublinear TF-IDF word & bigram features."
    },
    {
      key: "naive_bayes",
      displayName: "Naive Bayes",
      badge: "Traditional Baseline",
      type: "TF-IDF + MultinomialNB",
      weight: ensemble?.weights?.naive_bayes || 1.0,
      description: "Multinomial Naive Bayes classifier with uniform class priors."
    },
    {
      key: "roberta",
      displayName: "RoBERTa",
      badge: "Fine-Tuned Transformer",
      type: "FacebookAI/roberta-base",
      weight: ensemble?.weights?.roberta || 2.0,
      description: "125M parameter transformer encoder fine-tuned on BABE with classification head."
    },
    {
      key: "deberta",
      displayName: "DeBERTa",
      badge: "Fine-Tuned Transformer",
      type: "microsoft/deberta-v3-base",
      weight: ensemble?.weights?.deberta || 2.0,
      description: "Disentangled-attention transformer encoder; top individual model (82.87% F1)."
    }
  ];

  return (
    <div className="card" style={{ marginTop: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Scale size={20} color="#818cf8" />
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>
            Five Individual Classifier Outputs
          </h3>
        </div>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          Evaluated independently on input text
        </span>
      </div>

      <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1.25rem", lineHeight: 1.5 }}>
        Compare the individual predictions from all five underlying classifiers before weighted soft-voting aggregation.
        Transformers receive double weight (w=2) relative to traditional baselines (w=1).
      </p>

      {/* Grid of 5 Model Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1rem",
          marginBottom: "1.25rem"
        }}
      >
        {modelMetadata.map((meta) => {
          const pred = individualPredictions[meta.key] || {
            label: "Unavailable",
            probability_biased: 0.5,
            probability_non_biased: 0.5,
            confidence: 0.5
          };

          const isBiased = pred.label === "Biased";
          const biasedProb = pred.probability_biased !== undefined ? pred.probability_biased : 0.5;
          const biasedPct = (biasedProb * 100).toFixed(1);
          const nonBiasedPct = ((1 - biasedProb) * 100).toFixed(1);

          return (
            <div
              key={meta.key}
              style={{
                background: "var(--bg-subtle)",
                border: `1px solid ${isBiased ? "rgba(239, 68, 68, 0.35)" : "var(--border-color)"}`,
                borderRadius: "0.75rem",
                padding: "1rem",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                position: "relative",
                transition: "all 0.15s ease"
              }}
            >
              {/* Header */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.5rem" }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: "1rem", color: "var(--text-primary)" }}>
                      {meta.displayName}
                    </div>
                    <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {meta.type}
                    </div>
                  </div>
                  <span
                    style={{
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      padding: "0.15rem 0.45rem",
                      borderRadius: "0.25rem",
                      background: meta.weight === 2 ? "rgba(129, 140, 248, 0.2)" : "rgba(148, 163, 184, 0.15)",
                      color: meta.weight === 2 ? "#a5b4fc" : "#cbd5e1",
                      border: "1px solid var(--border-color)"
                    }}
                  >
                    Weight: {meta.weight}x
                  </span>
                </div>

                {/* Prediction Badge */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", margin: "0.75rem 0" }}>
                  <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>Prediction:</span>
                  <div
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.35rem",
                      padding: "0.25rem 0.65rem",
                      borderRadius: "0.4rem",
                      fontSize: "0.82rem",
                      fontWeight: 700,
                      background: isBiased ? "var(--biased-bg)" : "var(--nonbiased-bg)",
                      color: isBiased ? "#f87171" : "#34d399",
                      border: `1px solid ${isBiased ? "var(--biased-border)" : "var(--nonbiased-border)"}`
                    }}
                  >
                    {isBiased ? <AlertOctagon size={14} /> : <CheckCircle2 size={14} />}
                    <span>{pred.label}</span>
                  </div>
                </div>

                {/* Probability Bar */}
                <div style={{ marginBottom: "0.5rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: "0.3rem" }}>
                    <span style={{ color: "var(--text-secondary)" }}>P(Biased): <strong>{biasedPct}%</strong></span>
                    <span style={{ color: "var(--text-muted)" }}>P(Non-biased): {nonBiasedPct}%</span>
                  </div>
                  <div
                    style={{
                      height: "7px",
                      background: "rgba(16, 185, 129, 0.25)",
                      borderRadius: "4px",
                      overflow: "hidden",
                      position: "relative"
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${biasedPct}%`,
                        background: isBiased
                          ? "linear-gradient(90deg, #f59e0b, #ef4444)"
                          : "linear-gradient(90deg, #10b981, #3b82f6)",
                        borderRadius: "4px",
                        transition: "width 0.4s ease"
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Footer info */}
              <div
                style={{
                  fontSize: "0.74rem",
                  color: "var(--text-muted)",
                  borderTop: "1px solid rgba(51, 65, 85, 0.4)",
                  paddingTop: "0.5rem",
                  marginTop: "0.5rem",
                  lineHeight: 1.4
                }}
              >
                {meta.description}
              </div>
            </div>
          );
        })}
      </div>

      {/* Warning Notice */}
      <div
        style={{
          background: "rgba(234, 179, 8, 0.08)",
          border: "1px solid rgba(234, 179, 8, 0.25)",
          borderRadius: "0.5rem",
          padding: "0.65rem 0.85rem",
          fontSize: "0.82rem",
          color: "#fef08a",
          lineHeight: 1.45
        }}
      >
        <strong>Model Notice:</strong> All scores reflect statistical estimates from models trained on labeled news text.
        They indicate linguistic framing and emotional word choice, not objective factual truth or publisher political ideology.
      </div>
    </div>
  );
}
