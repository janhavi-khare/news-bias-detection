import React from "react";
import { X, Award, BarChart3 } from "lucide-react";

export function MetricsModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Award size={20} color="#818cf8" />
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Model Benchmark Evaluation</h3>
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginBottom: "1.25rem" }}>
          Reported performance metrics on the held-out test dataset split (527 news articles/sentences)
          evaluated during model fine-tuning experiments.
        </p>

        <div className="metrics-grid">
          <div className="metric-stat">
            <div className="metric-value">81.97%</div>
            <div className="metric-label">Accuracy</div>
          </div>
          <div className="metric-stat">
            <div className="metric-value">0.8184</div>
            <div className="metric-label">Macro F1</div>
          </div>
          <div className="metric-stat">
            <div className="metric-value">0.8196</div>
            <div className="metric-label">Macro Precision</div>
          </div>
          <div className="metric-stat">
            <div className="metric-value">0.8176</div>
            <div className="metric-label">Macro Recall</div>
          </div>
        </div>

        <div style={{ marginBottom: "1.25rem" }}>
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem", fontWeight: 600 }}>
            Class-Level Breakdown
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
                <td>0.8213</td>
                <td>0.8475</td>
                <td>0.8342</td>
                <td>282</td>
              </tr>
              <tr>
                <td style={{ color: "#f87171", fontWeight: 600 }}>Biased</td>
                <td>0.8178</td>
                <td>0.7878</td>
                <td>0.8025</td>
                <td>245</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.5rem", fontWeight: 600 }}>
            Confusion Matrix (Rows: Actual, Columns: Predicted)
          </div>
          <table className="words-table" style={{ textAlign: "center" }}>
            <thead>
              <tr>
                <th>Actual \ Predicted</th>
                <th>Pred: Non-biased</th>
                <th>Pred: Biased</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 600 }}>Actual: Non-biased</td>
                <td style={{ color: "#34d399", fontWeight: 700 }}>239 (True Neg)</td>
                <td style={{ color: "#f87171" }}>43 (False Pos)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Actual: Biased</td>
                <td style={{ color: "#f87171" }}>52 (False Neg)</td>
                <td style={{ color: "#34d399", fontWeight: 700 }}>193 (True Pos)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
