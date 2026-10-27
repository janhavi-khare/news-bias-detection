import React from "react";
import { BookOpen, ShieldAlert, Cpu, Database, CheckCircle2, AlertTriangle, Layers } from "lucide-react";

export function AboutPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem", maxWidth: "900px", margin: "0 auto" }}>
      {/* Title */}
      <div>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", color: "#818cf8", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.5rem" }}>
          <BookOpen size={16} />
          <span>System Documentation & Methodology</span>
        </div>
        <h1 style={{ fontSize: "1.85rem", fontWeight: 800, margin: 0, letterSpacing: "-0.01em" }}>
          About the News Bias Detection Project
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6, marginTop: "0.5rem" }}>
          An academic machine learning system developed to detect sentence-level media bias,
          combining classical NLP baselines with state-of-the-art transformer encoders via weighted soft voting.
        </p>
      </div>

      {/* Ensemble Methodology Section */}
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
          <Cpu size={20} color="#818cf8" />
          <h2 style={{ fontSize: "1.15rem", fontWeight: 700, margin: 0 }}>
            Weighted Soft Voting Ensemble Methodology
          </h2>
        </div>

        <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          Rather than relying on a single model architecture, this system combines predictions across five distinct classifiers.
          Each model provides a continuous probability estimate $P(\text{biased}) \in [0, 1]$:
        </p>

        <div
          style={{
            background: "rgba(15, 23, 42, 0.6)",
            border: "1px solid var(--border-color)",
            borderRadius: "0.5rem",
            padding: "1rem",
            fontFamily: "var(--font-mono)",
            fontSize: "0.85rem",
            color: "#c7d2fe",
            margin: "0.75rem 0",
            textAlign: "center"
          }}
        >
          P_ensemble(biased) = (1.0 × P_SVM + 1.0 × P_LR + 1.0 × P_NB + 2.0 × P_RoBERTa + 2.0 × P_DeBERTa) / 7.0
        </div>

        <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginTop: "1rem", marginBottom: "0.5rem" }}>
          Why the 0.40 Decision Threshold?
        </h3>
        <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          In media bias analysis, false negatives (missing actual bias) are often more damaging to analytical rigor than slight false positives.
          Empirical validation tuning on 550 held-out sentences demonstrated that a 0.40 threshold boosts biased-class recall from 82.03% (DeBERTa) to <strong>87.11%</strong> (Ensemble),
          while maintaining strong macro-F1 (82.63%) and overall accuracy (82.68%).
        </p>
      </div>

      {/* Dataset & Preprocessing Section */}
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
          <Database size={20} color="#818cf8" />
          <h2 style={{ fontSize: "1.15rem", fontWeight: 700, margin: 0 }}>
            Training Dataset: BABE
          </h2>
        </div>

        <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          The models are fine-tuned on the <strong>BABE (Media Bias Annotations by Experts)</strong> corpus.
          BABE contains expert sentence-level annotations characterizing whether sentences from major news outlets exhibit biased language or neutral reporting.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
          <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Training Examples</div>
            <div style={{ fontSize: "1.2rem", fontWeight: 800 }}>2,579 sentences</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Validation Split</div>
            <div style={{ fontSize: "1.2rem", fontWeight: 800 }}>550 sentences</div>
          </div>
          <div style={{ background: "var(--bg-subtle)", padding: "0.75rem", borderRadius: "0.5rem", border: "1px solid var(--border-color)" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Held-Out Test Split</div>
            <div style={{ fontSize: "1.2rem", fontWeight: 800 }}>508 sentences</div>
          </div>
        </div>
      </div>

      {/* Ethical Boundaries & Scope Notice */}
      <div
        style={{
          background: "rgba(239, 68, 68, 0.05)",
          border: "1px solid rgba(239, 68, 68, 0.25)",
          borderRadius: "0.75rem",
          padding: "1.25rem"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#f87171", fontWeight: 700, marginBottom: "0.5rem" }}>
          <ShieldAlert size={18} />
          <span>Important Academic Boundaries & Limitations</span>
        </div>
        <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
          <li>
            <strong>Lexical Framing vs Factual Truth:</strong> The classifiers detect emotional adjectives, dramatic framing, and loaded word choices.
            A sentence labeled &quot;Non-biased&quot; is not verified as factually true; it merely adheres to neutral journalistic tone.
          </li>
          <li>
            <strong>No Political Ideology:</strong> The models perform binary classification (Biased vs Non-biased) and do not classify articles as Left, Center, or Right.
          </li>
          <li>
            <strong>Publisher Scope:</strong> High or low bias scores for an excerpt do not establish that a publishing outlet or journalist is biased overall.
          </li>
          <li>
            <strong>Exploratory Benchmark Notice:</strong> The reported test set has been examined across multiple experiments during development.
            These results should therefore be interpreted as exploratory rather than as an untouched final evaluation. Performance on new data may differ.
          </li>
        </ul>
      </div>
    </div>
  );
}
