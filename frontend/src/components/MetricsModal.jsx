import React, { useState } from "react";
import { X, Award, BarChart3, Info, AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";

export function MetricsModal({ isOpen, onClose }) {
  const [activeTooltip, setActiveTooltip] = useState(null);

  if (!isOpen) return null;

  const metricDefinitions = {
    accuracy: "Proportion of all sentences (both biased and non-biased) correctly classified out of all 508 test sentences.",
    macroF1: "Unweighted average of F1 scores across both classes (Non-biased: 82.91%, Biased: 82.84%), ensuring equal weighting regardless of sample size.",
    valMacroF1: "Best validation checkpoint score achieved during fine-tuning epoch selection on the 550 validation sentences.",
    precision: "Proportion of sentences predicted as biased that were genuinely biased (210 / 251 = 83.67%). Higher precision means fewer false alarms.",
    recall: "Proportion of actual biased sentences in the test set that were correctly flagged by the model (210 / 256 = 82.03%). Higher recall means fewer missed biases."
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        style={{ maxWidth: "680px" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Award size={22} color="#818cf8" />
            <h3 style={{ fontSize: "1.25rem", fontWeight: 700 }}>
              Model Benchmark: DeBERTa-v3-base (BABE-Only)
            </h3>
          </div>
          <button className="close-btn" onClick={onClose} aria-label="Close modal">
            <X size={20} />
          </button>
        </div>

        {/* Model Description Section */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.5)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.65rem",
            padding: "1rem",
            marginBottom: "1.25rem",
            fontSize: "0.88rem",
            lineHeight: 1.6
          }}
        >
          <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.35rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <Info size={16} color="#818cf8" />
            <span>Model Information</span>
          </div>
          <p style={{ color: "var(--text-secondary)", marginBottom: "0.75rem" }}>
            DeBERTa-v3-base is a pretrained transformer encoder fine-tuned on the BABE dataset for sentence-level news media bias detection. Given an English news sentence, it predicts whether the sentence contains biased language or is non-biased. The model is evaluated using accuracy, macro-F1, class-wise F1, precision, recall, and a confusion matrix.
          </p>
          <div
            style={{
              background: "rgba(234, 179, 8, 0.08)",
              border: "1px solid rgba(234, 179, 8, 0.25)",
              borderRadius: "0.5rem",
              padding: "0.6rem 0.75rem",
              fontSize: "0.82rem",
              color: "#fef08a",
              display: "flex",
              alignItems: "flex-start",
              gap: "0.5rem"
            }}
          >
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: "0.15rem", color: "#facc15" }} />
            <span>
              <strong>Scope Notice:</strong> The model detects potentially biased language. It does not establish whether a news claim is factually true, determine the author&apos;s intentions, or prove that an entire article or news outlet is politically biased.
            </span>
          </div>
        </div>

        {/* Dataset Splits Summary */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "0.5rem",
            marginBottom: "1.25rem",
            padding: "0.75rem 1rem",
            background: "var(--bg-subtle)",
            borderRadius: "0.5rem",
            border: "1px solid var(--border-color)",
            fontSize: "0.82rem"
          }}
        >
          <div>
            <span style={{ color: "var(--text-muted)" }}>Architecture: </span>
            <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>microsoft/deberta-v3-base</span>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>Splits: </span>
            <span style={{ color: "var(--text-secondary)" }}>
              Train: <strong>2,579</strong> | Val: <strong>550</strong> | Test: <strong>508</strong>
            </span>
          </div>
          <div>
            <span style={{ color: "var(--text-muted)" }}>Split Strategy: </span>
            <span style={{ color: "#a5b4fc", fontWeight: 600 }}>Article-Disjoint</span>
          </div>
        </div>

        {/* Primary Metrics Grid */}
        <div style={{ marginBottom: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", fontWeight: 600 }}>
              Benchmark Metrics (Test & Validation)
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Hover &apos;?&apos; for definitions
            </span>
          </div>

          <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(135px, 1fr))" }}>
            {/* Test Accuracy */}
            <div className="metric-stat" style={{ position: "relative" }}>
              <span
                style={{
                  position: "absolute",
                  top: "0.35rem",
                  right: "0.5rem",
                  fontSize: "0.65rem",
                  padding: "0.1rem 0.35rem",
                  borderRadius: "0.25rem",
                  background: "rgba(99, 102, 241, 0.2)",
                  color: "#a5b4fc",
                  fontWeight: 600
                }}
              >
                TEST
              </span>
              <div className="metric-value">82.87%</div>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem" }}>
                <span>Accuracy</span>
                <HelpCircle
                  size={12}
                  style={{ cursor: "pointer", color: "var(--text-muted)" }}
                  onMouseEnter={() => setActiveTooltip("accuracy")}
                  onMouseLeave={() => setActiveTooltip(null)}
                />
              </div>
            </div>

            {/* Test Macro-F1 */}
            <div className="metric-stat" style={{ position: "relative" }}>
              <span
                style={{
                  position: "absolute",
                  top: "0.35rem",
                  right: "0.5rem",
                  fontSize: "0.65rem",
                  padding: "0.1rem 0.35rem",
                  borderRadius: "0.25rem",
                  background: "rgba(99, 102, 241, 0.2)",
                  color: "#a5b4fc",
                  fontWeight: 600
                }}
              >
                TEST
              </span>
              <div className="metric-value">82.87%</div>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem" }}>
                <span>Macro-F1</span>
                <HelpCircle
                  size={12}
                  style={{ cursor: "pointer", color: "var(--text-muted)" }}
                  onMouseEnter={() => setActiveTooltip("macroF1")}
                  onMouseLeave={() => setActiveTooltip(null)}
                />
              </div>
            </div>

            {/* Validation Macro-F1 */}
            <div className="metric-stat" style={{ position: "relative", border: "1px solid rgba(52, 211, 153, 0.3)" }}>
              <span
                style={{
                  position: "absolute",
                  top: "0.35rem",
                  right: "0.5rem",
                  fontSize: "0.65rem",
                  padding: "0.1rem 0.35rem",
                  borderRadius: "0.25rem",
                  background: "rgba(52, 211, 153, 0.2)",
                  color: "#34d399",
                  fontWeight: 600
                }}
              >
                VALIDATION
              </span>
              <div className="metric-value" style={{ color: "#34d399" }}>83.09%</div>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem" }}>
                <span>Val Macro-F1</span>
                <HelpCircle
                  size={12}
                  style={{ cursor: "pointer", color: "var(--text-muted)" }}
                  onMouseEnter={() => setActiveTooltip("valMacroF1")}
                  onMouseLeave={() => setActiveTooltip(null)}
                />
              </div>
            </div>

            {/* Biased Precision */}
            <div className="metric-stat" style={{ position: "relative" }}>
              <span
                style={{
                  position: "absolute",
                  top: "0.35rem",
                  right: "0.5rem",
                  fontSize: "0.65rem",
                  padding: "0.1rem 0.35rem",
                  borderRadius: "0.25rem",
                  background: "rgba(99, 102, 241, 0.2)",
                  color: "#a5b4fc",
                  fontWeight: 600
                }}
              >
                TEST
              </span>
              <div className="metric-value">83.67%</div>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem" }}>
                <span>Biased Precision</span>
                <HelpCircle
                  size={12}
                  style={{ cursor: "pointer", color: "var(--text-muted)" }}
                  onMouseEnter={() => setActiveTooltip("precision")}
                  onMouseLeave={() => setActiveTooltip(null)}
                />
              </div>
            </div>

            {/* Biased Recall */}
            <div className="metric-stat" style={{ position: "relative" }}>
              <span
                style={{
                  position: "absolute",
                  top: "0.35rem",
                  right: "0.5rem",
                  fontSize: "0.65rem",
                  padding: "0.1rem 0.35rem",
                  borderRadius: "0.25rem",
                  background: "rgba(99, 102, 241, 0.2)",
                  color: "#a5b4fc",
                  fontWeight: 600
                }}
              >
                TEST
              </span>
              <div className="metric-value">82.03%</div>
              <div className="metric-label" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.25rem" }}>
                <span>Biased Recall</span>
                <HelpCircle
                  size={12}
                  style={{ cursor: "pointer", color: "var(--text-muted)" }}
                  onMouseEnter={() => setActiveTooltip("recall")}
                  onMouseLeave={() => setActiveTooltip(null)}
                />
              </div>
            </div>
          </div>

          {/* Active Tooltip Box */}
          {activeTooltip && (
            <div
              style={{
                marginTop: "0.5rem",
                padding: "0.5rem 0.75rem",
                background: "rgba(99, 102, 241, 0.1)",
                border: "1px solid rgba(99, 102, 241, 0.3)",
                borderRadius: "0.4rem",
                fontSize: "0.8rem",
                color: "#c7d2fe"
              }}
            >
              {metricDefinitions[activeTooltip]}
            </div>
          )}
        </div>

        {/* Class-Level Breakdown Table */}
        <div style={{ marginBottom: "1.25rem" }}>
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem", fontWeight: 600 }}>
            Class-Level Breakdown (Test Split: 508 Sentences)
          </div>
          <table className="words-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ color: "#34d399", fontWeight: 600 }}>Non-biased</td>
                <td>82.10%</td>
                <td>83.73%</td>
                <td style={{ fontWeight: 700, color: "#34d399" }}>82.91%</td>
                <td>252</td>
              </tr>
              <tr>
                <td style={{ color: "#f87171", fontWeight: 600 }}>Biased</td>
                <td>83.67%</td>
                <td>82.03%</td>
                <td style={{ fontWeight: 700, color: "#f87171" }}>82.84%</td>
                <td>256</td>
              </tr>
              <tr style={{ borderTop: "1.5px solid var(--border-color)", fontWeight: 600 }}>
                <td>Macro Average</td>
                <td>82.88%</td>
                <td>82.88%</td>
                <td style={{ color: "var(--accent-primary)" }}>82.87%</td>
                <td>508</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Metric Definitions Guide */}
        <div
          style={{
            background: "rgba(15, 23, 42, 0.4)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.5rem",
            padding: "0.75rem 1rem",
            marginBottom: "1.25rem",
            fontSize: "0.8rem",
            lineHeight: 1.5
          }}
        >
          <div style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.4rem" }}>
            Metric Definitions:
          </div>
          <ul style={{ paddingLeft: "1.2rem", margin: 0, color: "var(--text-muted)" }}>
            <li><strong style={{ color: "var(--text-secondary)" }}>Accuracy:</strong> Overall percentage of test sentences classified correctly.</li>
            <li><strong style={{ color: "var(--text-secondary)" }}>Macro-F1:</strong> Unweighted mean of class F1-scores, treating both classes with equal importance.</li>
            <li><strong style={{ color: "var(--text-secondary)" }}>Precision:</strong> Proportion of sentences flagged as biased that were genuinely biased (minimizes false alarms).</li>
            <li><strong style={{ color: "var(--text-secondary)" }}>Recall:</strong> Proportion of actual biased sentences that were detected by the model (minimizes missed bias).</li>
          </ul>
        </div>

        {/* Confusion Matrix Section */}
        <div style={{ marginBottom: "1.25rem" }}>
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem", fontWeight: 600 }}>
            Test Confusion Matrix (Rows: Actual, Columns: Predicted)
          </div>
          <table className="words-table" style={{ textAlign: "center" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left" }}>Actual \ Predicted</th>
                <th style={{ textAlign: "center" }}>Pred: Non-biased</th>
                <th style={{ textAlign: "center" }}>Pred: Biased</th>
                <th style={{ textAlign: "center" }}>Total Support</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Non-biased</td>
                <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>
                  211
                </td>
                <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>
                  41
                </td>
                <td style={{ fontWeight: 600 }}>252</td>
              </tr>
              <tr>
                <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Biased</td>
                <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>
                  46
                </td>
                <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>
                  210
                </td>
                <td style={{ fontWeight: 600 }}>256</td>
              </tr>
              <tr style={{ borderTop: "1px solid var(--border-color)", fontWeight: 600 }}>
                <td style={{ textAlign: "left" }}>Total Predicted</td>
                <td style={{ color: "var(--text-secondary)" }}>257</td>
                <td style={{ color: "var(--text-secondary)" }}>251</td>
                <td style={{ color: "var(--accent-primary)" }}>508</td>
              </tr>
            </tbody>
          </table>

          {/* Plain Language Outcome Descriptions */}
          <div
            style={{
              marginTop: "0.75rem",
              background: "var(--bg-subtle)",
              borderRadius: "0.5rem",
              padding: "0.75rem 1rem",
              border: "1px solid var(--border-color)",
              fontSize: "0.82rem",
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "0.5rem"
            }}
          >
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
              <span style={{ color: "#34d399", fontWeight: 700 }}>211</span>
              <span style={{ color: "var(--text-secondary)" }}>
                non-biased sentences correctly identified as non-biased
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
              <span style={{ color: "#f87171", fontWeight: 700 }}>41</span>
              <span style={{ color: "var(--text-secondary)" }}>
                non-biased sentences mistakenly flagged as biased
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
              <span style={{ color: "#f87171", fontWeight: 700 }}>46</span>
              <span style={{ color: "var(--text-secondary)" }}>
                biased sentences missed by the model
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
              <span style={{ color: "#34d399", fontWeight: 700 }}>210</span>
              <span style={{ color: "var(--text-secondary)" }}>
                biased sentences correctly detected as biased
              </span>
            </div>
          </div>
        </div>

        {/* Mandatory Evaluation Caveat */}
        <div
          style={{
            background: "rgba(99, 102, 241, 0.08)",
            border: "1px solid rgba(99, 102, 241, 0.25)",
            borderRadius: "0.5rem",
            padding: "0.75rem 1rem",
            fontSize: "0.82rem",
            color: "#c7d2fe",
            lineHeight: 1.5
          }}
        >
          <div style={{ fontWeight: 600, color: "#e0e7ff", marginBottom: "0.25rem", display: "flex", alignItems: "center", gap: "0.35rem" }}>
            <Info size={14} color="#818cf8" />
            <span>Evaluation Note:</span>
          </div>
          The reported test set has been examined across multiple experiments during development. These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. Performance on new data may differ.
        </div>
      </div>
    </div>
  );
}
