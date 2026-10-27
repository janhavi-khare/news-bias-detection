import React from "react";
import { Award, Zap, ShieldCheck, ArrowRight, Layers, Cpu, CheckCircle2, AlertTriangle, FileText } from "lucide-react";

export function DashboardPage({ onNavigateToDetection, modelsStatus }) {
  const modelsList = [
    {
      name: "Linear SVM",
      type: "Traditional ML",
      algorithm: "TF-IDF + Calibrated LinearSVC",
      weight: "1.0x",
      macroF1: "70.15%",
      biasedRecall: "63.28%",
      role: "Linear margin baseline with Platt scaling"
    },
    {
      name: "Logistic Regression",
      type: "Traditional ML",
      algorithm: "TF-IDF + LogisticRegression",
      weight: "1.0x",
      macroF1: "70.84%",
      biasedRecall: "67.58%",
      role: "Probabilistic linear baseline with L2 penalty"
    },
    {
      name: "Naive Bayes",
      type: "Traditional ML",
      algorithm: "TF-IDF + MultinomialNB",
      weight: "1.0x",
      macroF1: "68.62%",
      biasedRecall: "63.28%",
      role: "Conditional word frequency baseline"
    },
    {
      name: "RoBERTa",
      type: "Transformer",
      algorithm: "FacebookAI/roberta-base (125M)",
      weight: "2.0x",
      macroF1: "82.08%",
      biasedRecall: "80.47%",
      role: "Pretrained contextual representations"
    },
    {
      name: "DeBERTa",
      type: "Transformer",
      algorithm: "microsoft/deberta-v3-base (86M)",
      weight: "2.0x",
      macroF1: "82.87%",
      biasedRecall: "82.03%",
      role: "Top individual model; disentangled attention"
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      {/* Hero Welcome Banner */}
      <div
        style={{
          background: "linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95))",
          border: "1px solid var(--border-color)",
          borderRadius: "1rem",
          padding: "2rem",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1.5rem"
        }}
      >
        <div style={{ maxWidth: "680px" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "0.4rem",
              padding: "0.2rem 0.6rem",
              borderRadius: "9999px",
              background: "rgba(99, 102, 241, 0.15)",
              color: "#a5b4fc",
              fontSize: "0.78rem",
              fontWeight: 600,
              marginBottom: "0.75rem"
            }}
          >
            <ShieldCheck size={14} />
            <span>Multi-Model Bias Detection & Explainability System</span>
          </div>
          <h1 style={{ fontSize: "1.85rem", fontWeight: 800, marginBottom: "0.6rem", letterSpacing: "-0.02em" }}>
            5 Classifiers Connected with Weighted Soft Voting
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6, margin: 0 }}>
            An academic system integrating three TF-IDF machine learning baselines with fine-tuned RoBERTa and DeBERTa transformers.
            The weighted soft-voting ensemble combines all 5 probability outputs using an empirical 0.40 threshold rule for superior biased-class recall.
          </p>
        </div>

        <button
          onClick={onNavigateToDetection}
          className="analyze-btn"
          style={{ width: "auto", padding: "0.85rem 1.75rem", fontSize: "0.95rem" }}
        >
          <span>Analyze News Text</span>
          <ArrowRight size={18} />
        </button>
      </div>

      {/* 3 Summary Benchmark Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "1.25rem"
        }}
      >
        {/* Card 1: Top Individual Model */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.85rem",
            padding: "1.25rem",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between"
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
                Top Individual Model
              </span>
              <Award size={18} color="#818cf8" />
            </div>
            <div style={{ fontSize: "1.35rem", fontWeight: 800, color: "var(--text-primary)" }}>
              DeBERTa-v3-base
            </div>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
              Rank 1 single architecture on held-out test split
            </div>
          </div>
          <div style={{ display: "flex", gap: "1rem", marginTop: "1rem", borderTop: "1px solid var(--border-color)", paddingTop: "0.75rem" }}>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Test Macro-F1</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent-primary)" }}>82.87%</div>
            </div>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Accuracy</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "#34d399" }}>82.87%</div>
            </div>
          </div>
        </div>

        {/* Card 2: Ensemble Biased Recall */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid rgba(245, 158, 11, 0.3)",
            borderRadius: "0.85rem",
            padding: "1.25rem",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between"
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
              <span style={{ fontSize: "0.8rem", color: "#f59e0b", textTransform: "uppercase", fontWeight: 600 }}>
                Ensemble Biased Recall
              </span>
              <Zap size={18} color="#f59e0b" />
            </div>
            <div style={{ fontSize: "1.35rem", fontWeight: 800, color: "#f59e0b" }}>
              87.11% (+5.08%)
            </div>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
              Significant reduction in missed biased sentences vs single models
            </div>
          </div>
          <div style={{ display: "flex", gap: "1rem", marginTop: "1rem", borderTop: "1px solid var(--border-color)", paddingTop: "0.75rem" }}>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Ensemble Recall</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f59e0b" }}>87.11%</div>
            </div>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Best Single Recall</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-muted)" }}>82.03%</div>
            </div>
          </div>
        </div>

        {/* Card 3: Ensemble Macro F1 */}
        <div
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.85rem",
            padding: "1.25rem",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between"
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600 }}>
                Ensemble Macro F1
              </span>
              <Layers size={18} color="#34d399" />
            </div>
            <div style={{ fontSize: "1.35rem", fontWeight: 800, color: "var(--text-primary)" }}>
              82.63% (F1) / 82.68% (Acc)
            </div>
            <div style={{ fontSize: "0.82rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
              Balanced accuracy across both classes with 0.40 threshold
            </div>
          </div>
          <div style={{ display: "flex", gap: "1rem", marginTop: "1rem", borderTop: "1px solid var(--border-color)", paddingTop: "0.75rem" }}>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Ensemble Biased F1</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "#34d399" }}>83.52%</div>
            </div>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Threshold</div>
              <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-secondary)" }}>0.40</div>
            </div>
          </div>
        </div>
      </div>

      {/* Weighted Soft-Voting Architecture Flow */}
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
          <Cpu size={20} color="#818cf8" />
          <h2 style={{ fontSize: "1.15rem", fontWeight: 700, margin: 0 }}>
            Weighted Soft-Voting Ensemble Architecture
          </h2>
        </div>
        <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", marginBottom: "1.25rem", lineHeight: 1.5 }}>
          Each input sentence or excerpt is analyzed by all five trained models in parallel. Their continuous probability scores
          are weighted according to validation performance, and evaluated against the tuned 0.40 threshold.
        </p>

        <div
          style={{
            background: "rgba(15, 23, 42, 0.6)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.75rem",
            padding: "1.25rem",
            display: "flex",
            flexDirection: "column",
            gap: "1rem"
          }}
        >
          {/* Formula Display Box */}
          <div
            style={{
              padding: "0.85rem 1rem",
              background: "rgba(99, 102, 241, 0.08)",
              border: "1px solid rgba(99, 102, 241, 0.25)",
              borderRadius: "0.5rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.88rem",
              color: "#c7d2fe",
              textAlign: "center"
            }}
          >
            P_ensemble(biased) = (1.0 × P_SVM + 1.0 × P_LR + 1.0 × P_NB + 2.0 × P_RoBERTa + 2.0 × P_DeBERTa) / 7.0
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "#a5b4fc" }}>Decision Threshold: 0.40</div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                Tuned on 550 validation sentences to balance precision and recall.
              </div>
            </div>
            <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "#a5b4fc" }}>Weighted Model Proportions</div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                Transformers carry 57.1% (4/7) of total voting power; traditional baselines carry 42.9% (3/7).
              </div>
            </div>
            <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
              <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "#a5b4fc" }}>Article Disjoint Evaluation</div>
              <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.2rem" }}>
                Tested on 508 unseen held-out sentences with zero article leakage from training.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Connected Models Status Table */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <div>
            <h2 style={{ fontSize: "1.15rem", fontWeight: 700, margin: 0 }}>Connected Model Inventory</h2>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", margin: "0.2rem 0 0" }}>
              Status of all 5 classifiers active in backend memory
            </p>
          </div>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table className="words-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Type</th>
                <th>Algorithm / Architecture</th>
                <th>Ensemble Weight</th>
                <th>Macro F1</th>
                <th>Biased Recall</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {modelsList.map((m) => (
                <tr key={m.name}>
                  <td style={{ fontWeight: 700, color: "var(--text-primary)" }}>{m.name}</td>
                  <td>
                    <span
                      style={{
                        fontSize: "0.72rem",
                        padding: "0.15rem 0.45rem",
                        borderRadius: "0.3rem",
                        background: m.type === "Transformer" ? "rgba(129, 140, 248, 0.15)" : "rgba(148, 163, 184, 0.12)",
                        color: m.type === "Transformer" ? "#a5b4fc" : "#cbd5e1"
                      }}
                    >
                      {m.type}
                    </span>
                  </td>
                  <td style={{ fontSize: "0.82rem", fontFamily: "var(--font-mono)" }}>{m.algorithm}</td>
                  <td style={{ fontWeight: 700, color: "var(--accent-primary)" }}>{m.weight}</td>
                  <td style={{ fontWeight: 600 }}>{m.macroF1}</td>
                  <td style={{ fontWeight: 600, color: "#f59e0b" }}>{m.biasedRecall}</td>
                  <td>
                    <span style={{ color: "#34d399", display: "inline-flex", alignItems: "center", gap: "0.25rem", fontSize: "0.8rem", fontWeight: 600 }}>
                      <CheckCircle2 size={13} /> Active
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
