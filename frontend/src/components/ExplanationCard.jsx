import React, { useState } from "react";
import { ChevronDown, ChevronUp, Sparkles, Info, AlertCircle } from "lucide-react";

export function ExplanationCard({
  explanation,
  isLoadingExplanation,
  onRequestExplanation
}) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!explanation && !isLoadingExplanation) {
    return (
      <div className="explanation-section">
        <button
          className="secondary-btn"
          onClick={onRequestExplanation}
          type="button"
        >
          <Sparkles size={16} color="#818cf8" />
          Explain with SHAP Word Attributions
        </button>
      </div>
    );
  }

  if (isLoadingExplanation) {
    return (
      <div className="explanation-section" style={{ textAlign: "center", padding: "1.5rem 0" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.75rem", color: "var(--text-secondary)" }}>
          <div className="spinner" style={{ borderColor: "#818cf8", borderTopColor: "transparent" }}></div>
          Computing SHAP feature attributions on model representations...
        </div>
      </div>
    );
  }

  if (!explanation.success) {
    return (
      <div className="explanation-section">
        <div className="alert alert-warning">
          <AlertCircle size={18} />
          <div>
            <strong>Explanation Note:</strong> {explanation.error || "Unable to compute SHAP attributions for this text."}
          </div>
        </div>
      </div>
    );
  }

  const { top_words, token_highlights, disclaimer } = explanation;

  return (
    <div className="explanation-section">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer",
          marginBottom: isExpanded ? "1rem" : "0"
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 700, fontSize: "1.1rem" }}>
          <Sparkles size={18} color="#818cf8" />
          <span>SHAP Word-Level Feature Attributions</span>
        </div>
        <button className="secondary-btn" style={{ padding: "0.3rem 0.6rem" }}>
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          <span style={{ fontSize: "0.8rem" }}>{isExpanded ? "Collapse" : "Expand"}</span>
        </button>
      </div>

      {isExpanded && (
        <>
          {token_highlights && token_highlights.length > 0 && (
            <div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                Interactive Token Attribution Map (Red = pushes toward Biased, Green = pushes toward Non-biased):
              </div>
              <div className="token-highlight-box">
                {token_highlights.map((item, idx) => {
                  let cls = "token-highlight token-inert";
                  if (item.score > 0.03) {
                    cls = "token-highlight token-biased";
                  } else if (item.score < -0.03) {
                    cls = "token-highlight token-neutral";
                  }
                  return (
                    <span
                      key={idx}
                      className={cls}
                      title={`Score: ${item.score > 0 ? "+" : ""}${item.score} (${item.direction})`}
                    >
                      {item.token}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {top_words && top_words.length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.4rem" }}>
                Top Influential Words (Stopwords filtered):
              </div>
              <table className="words-table">
                <thead>
                  <tr>
                    <th>Token / Word</th>
                    <th>Attribution Score</th>
                    <th>Model Impact Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {top_words.map((item, idx) => {
                    const isBiased = item.score > 0;
                    return (
                      <tr key={idx}>
                        <td>
                          <span className={`word-pill ${isBiased ? "pill-biased" : "pill-nonbiased"}`}>
                            {item.word}
                          </span>
                        </td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>
                          {item.score > 0 ? `+${item.score.toFixed(4)}` : item.score.toFixed(4)}
                        </td>
                        <td style={{ color: isBiased ? "#f87171" : "#34d399", fontWeight: 500 }}>
                          {item.direction}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div
            style={{
              display: "flex",
              gap: "0.5rem",
              alignItems: "flex-start",
              fontSize: "0.82rem",
              color: "var(--text-muted)",
              marginTop: "1.25rem",
              background: "rgba(255,255,255,0.02)",
              padding: "0.75rem",
              borderRadius: "0.5rem"
            }}
          >
            <Info size={16} style={{ flexShrink: 0, marginTop: "0.1rem" }} />
            <span>{disclaimer}</span>
          </div>
        </>
      )}
    </div>
  );
}
