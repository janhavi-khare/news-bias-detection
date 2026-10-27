import React, { useState } from "react";
import { ListFilter, ChevronDown, ChevronUp, Sparkles, AlertOctagon, CheckCircle2 } from "lucide-react";
import { ExplanationCard } from "./ExplanationCard";

export function SentenceBreakdown({
  sentences,
  onExplainSentence,
  explainingSentenceIdx
}) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  if (!sentences || sentences.length <= 1) {
    return null;
  }

  const toggleExpand = (idx) => {
    setExpandedIndex(expandedIndex === idx ? null : idx);
  };

  return (
    <div style={{ marginTop: "1.75rem", borderTop: "1px solid var(--border-color)", paddingTop: "1.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1rem" }}>
        <ListFilter size={18} color="#818cf8" />
        <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>
          Sentence-by-Sentence Breakdown ({sentences.length} sentences)
        </h3>
      </div>

      <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
        Each sentence was classified independently using DeBERTa-v3-base (matching the BABE training granularity).
        Sentences with loaded language are highlighted below.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
        {sentences.map((s) => {
          const isBiased = s.predicted_label === "Biased";
          const biasedPct = (s.biased_probability * 100).toFixed(1);
          const nonBiasedPct = (s.non_biased_probability * 100).toFixed(1);
          const isExpanded = expandedIndex === s.index;
          const isBeingExplained = explainingSentenceIdx === s.index;

          return (
            <div
              key={s.index}
              style={{
                background: isBiased ? "rgba(239, 68, 68, 0.05)" : "rgba(255, 255, 255, 0.02)",
                border: `1px solid ${isBiased ? "rgba(239, 68, 68, 0.3)" : "var(--border-color)"}`,
                borderRadius: "0.65rem",
                padding: "0.9rem 1rem",
                transition: "all 0.15s ease"
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "0.75rem",
                  flexWrap: "wrap"
                }}
              >
                <div style={{ display: "flex", alignItems: "flex-start", gap: "0.6rem", flex: 1, minWidth: "260px" }}>
                  <span
                    style={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border-color)",
                      color: "var(--text-muted)",
                      borderRadius: "0.35rem",
                      padding: "0.15rem 0.45rem",
                      fontSize: "0.75rem",
                      fontFamily: "var(--font-mono)",
                      fontWeight: 600,
                      marginTop: "0.15rem"
                    }}
                  >
                    #{s.index}
                  </span>
                  <p style={{ fontSize: "0.92rem", lineHeight: 1.5, color: "var(--text-primary)" }}>
                    {s.text}
                  </p>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span
                    style={{
                      fontSize: "0.78rem",
                      fontWeight: 700,
                      padding: "0.2rem 0.55rem",
                      borderRadius: "0.35rem",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.3rem",
                      background: isBiased ? "var(--biased-bg)" : "var(--nonbiased-bg)",
                      color: isBiased ? "#f87171" : "#34d399",
                      border: `1px solid ${isBiased ? "var(--biased-border)" : "var(--nonbiased-border)"}`
                    }}
                  >
                    {isBiased ? <AlertOctagon size={12} /> : <CheckCircle2 size={12} />}
                    {s.predicted_label} ({isBiased ? `${biasedPct}%` : `${nonBiasedPct}%`})
                  </span>

                  <button
                    type="button"
                    className="secondary-btn"
                    style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
                    onClick={() => {
                      if (!s.explanation && onExplainSentence) {
                        onExplainSentence(s.index, s.text);
                      }
                      toggleExpand(s.index);
                    }}
                  >
                    <Sparkles size={12} color="#818cf8" />
                    <span>Explain</span>
                    {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div style={{ marginTop: "0.75rem" }}>
                  <ExplanationCard
                    explanation={s.explanation}
                    isLoadingExplanation={isBeingExplained}
                    onRequestExplanation={() => onExplainSentence && onExplainSentence(s.index, s.text)}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
