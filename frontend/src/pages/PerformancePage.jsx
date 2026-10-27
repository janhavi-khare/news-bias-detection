import React, { useState, useEffect } from "react";
import {
  BarChart3,
  Download,
  Award,
  ArrowUpDown,
  Info,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ChevronDown,
  ChevronUp
} from "lucide-react";
import { fetchMetrics, fetchConfusionMatrices, getComparisonCsvUrl } from "../services/api";

export function PerformancePage() {
  const [metricsData, setMetricsData] = useState(null);
  const [confusionData, setConfusionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Sorting state
  const [sortField, setSortField] = useState("macro_f1");
  const [sortAsc, setSortAsc] = useState(false);

  // Selected metric for chart visualization
  const [selectedChartMetric, setSelectedChartMetric] = useState("macro_f1");

  // Selected model for detailed drill-down
  const [selectedModelName, setSelectedModelName] = useState("Weighted Soft Voting");

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [mRes, cRes] = await Promise.all([
          fetchMetrics(),
          fetchConfusionMatrices()
        ]);
        setMetricsData(mRes);
        setConfusionData(cRes);
      } catch (err) {
        console.error("Failed to load benchmark data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Static authoritative fallback table matching prompt Section 6
  const models = metricsData?.models || [
    {
      name: "DeBERTa",
      accuracy: 0.8287,
      macro_precision: 0.8288,
      macro_recall: 0.8288,
      macro_f1: 0.8287,
      biased_precision: 0.8367,
      biased_recall: 0.8203,
      biased_f1: 0.8284,
      validation_macro_f1: 0.8309,
      rank: 1,
      type: "Transformer Encoder"
    },
    {
      name: "Weighted Soft Voting",
      accuracy: 0.8268,
      macro_precision: 0.8293,
      macro_recall: 0.8264,
      macro_f1: 0.8263,
      biased_precision: 0.8022,
      biased_recall: 0.8711,
      biased_f1: 0.8352,
      validation_macro_f1: 0.8284,
      rank: 2,
      type: "Ensemble (Threshold 0.40)"
    },
    {
      name: "RoBERTa",
      accuracy: 0.8209,
      macro_precision: 0.8212,
      macro_recall: 0.8210,
      macro_f1: 0.8208,
      biased_precision: 0.8340,
      biased_recall: 0.8047,
      biased_f1: 0.8191,
      validation_macro_f1: 0.8215,
      rank: 3,
      type: "Transformer Encoder"
    },
    {
      name: "Logistic Regression",
      accuracy: 0.7087,
      macro_precision: 0.7097,
      macro_recall: 0.7089,
      macro_f1: 0.7084,
      biased_precision: 0.7269,
      biased_recall: 0.6758,
      biased_f1: 0.7004,
      validation_macro_f1: 0.7120,
      rank: 4,
      type: "Linear Classifier"
    },
    {
      name: "Linear SVM",
      accuracy: 0.7028,
      macro_precision: 0.7072,
      macro_recall: 0.7033,
      macro_f1: 0.7015,
      biased_precision: 0.7397,
      biased_recall: 0.6328,
      biased_f1: 0.6821,
      validation_macro_f1: 0.7085,
      rank: 5,
      type: "Support Vector Machine"
    },
    {
      name: "Naive Bayes",
      accuracy: 0.6870,
      macro_precision: 0.6896,
      macro_recall: 0.6874,
      macro_f1: 0.6862,
      biased_precision: 0.7137,
      biased_recall: 0.6328,
      biased_f1: 0.6708,
      validation_macro_f1: 0.6950,
      rank: 6,
      type: "Probabilistic Baseline"
    }
  ];

  // Sorting handler
  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedModels = [...models].sort((a, b) => {
    const valA = a[sortField] !== undefined ? a[sortField] : 0;
    const valB = b[sortField] !== undefined ? b[sortField] : 0;
    if (typeof valA === "string") {
      return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    }
    return sortAsc ? valA - valB : valB - valA;
  });

  const chartMetrics = [
    { key: "macro_f1", label: "Macro F1" },
    { key: "accuracy", label: "Accuracy" },
    { key: "biased_recall", label: "Biased Recall" },
    { key: "biased_precision", label: "Biased Precision" },
    { key: "biased_f1", label: "Biased F1" },
    { key: "macro_precision", label: "Macro Precision" },
    { key: "macro_recall", label: "Macro Recall" }
  ];

  const selectedModelObj = models.find((m) => m.name === selectedModelName) || models[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      {/* Header and Download Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 style={{ fontSize: "1.75rem", fontWeight: 800, margin: 0, letterSpacing: "-0.01em" }}>
            Model Performance Comparison
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", margin: "0.35rem 0 0" }}>
            Benchmark results evaluated on the BABE held-out article-disjoint test split (508 sentences).
          </p>
        </div>

        <a
          href={getComparisonCsvUrl()}
          download="model_performance_comparison.csv"
          className="secondary-btn"
          style={{ textDecoration: "none" }}
        >
          <Download size={16} />
          <span>Download Table as CSV</span>
        </a>
      </div>

      {/* Dataset & Experiment Metadata Banner */}
      <div
        style={{
          background: "rgba(15, 23, 42, 0.6)",
          border: "1px solid var(--border-color)",
          borderRadius: "0.75rem",
          padding: "1rem 1.25rem",
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "1rem",
          fontSize: "0.85rem"
        }}
      >
        <div>
          <span style={{ color: "var(--text-muted)" }}>Dataset: </span>
          <strong style={{ color: "var(--text-primary)" }}>BABE (Media Bias Annotations by Experts)</strong>
        </div>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Split Strategy: </span>
          <strong style={{ color: "#a5b4fc" }}>Article-Disjoint (Zero overlap)</strong>
        </div>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Data Splits: </span>
          <span>Train: <strong>2,579</strong> | Val: <strong>550</strong> | Test: <strong>508</strong></span>
        </div>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Decision Rule: </span>
          <span>Ensemble Threshold: <strong>0.40</strong> (Single models: 0.50)</span>
        </div>
      </div>

      {/* Interactive Bar Chart Visualization */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.75rem", marginBottom: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <BarChart3 size={20} color="#818cf8" />
            <h2 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>
              Performance Metric Comparison Bar Chart
            </h2>
          </div>

          {/* Metric Selector Buttons */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
            {chartMetrics.map((cm) => (
              <button
                key={cm.key}
                onClick={() => setSelectedChartMetric(cm.key)}
                style={{
                  padding: "0.3rem 0.65rem",
                  borderRadius: "0.4rem",
                  border: "1px solid var(--border-color)",
                  background: selectedChartMetric === cm.key ? "var(--accent-primary)" : "var(--bg-subtle)",
                  color: selectedChartMetric === cm.key ? "#fff" : "var(--text-secondary)",
                  fontSize: "0.78rem",
                  fontWeight: selectedChartMetric === cm.key ? 700 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease"
                }}
              >
                {cm.label}
              </button>
            ))}
          </div>
        </div>

        {/* SVG Horizontal Bar Chart */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem", padding: "0.5rem 0" }}>
          {models.map((m) => {
            const val = m[selectedChartMetric] || 0;
            const pct = (val * 100).toFixed(2);
            const isEnsemble = m.name.includes("Ensemble") || m.name.includes("Voting");
            const isTopModel = m.name === "DeBERTa";

            return (
              <div key={m.name} style={{ display: "grid", gridTemplateColumns: "180px 1fr 75px", alignItems: "center", gap: "1rem" }}>
                <div style={{ textAlign: "right", fontSize: "0.85rem", fontWeight: 600, color: isEnsemble ? "#f59e0b" : isTopModel ? "#818cf8" : "var(--text-secondary)" }}>
                  {m.name}
                </div>
                <div style={{ background: "rgba(255, 255, 255, 0.05)", height: "24px", borderRadius: "6px", overflow: "hidden", position: "relative" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${pct}%`,
                      background: isEnsemble
                        ? "linear-gradient(90deg, #d97706, #f59e0b)"
                        : isTopModel
                        ? "linear-gradient(90deg, #4f46e5, #818cf8)"
                        : "linear-gradient(90deg, #334155, #64748b)",
                      borderRadius: "6px",
                      transition: "width 0.4s ease"
                    }}
                  />
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)" }}>
                  {pct}%
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Sortable Comparison Table */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>
            Authoritative Held-Out Test Split Comparison
          </h2>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Click column headers to sort
          </span>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table className="words-table">
            <thead>
              <tr>
                <th onClick={() => handleSort("name")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Model</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("accuracy")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Accuracy</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("macro_precision")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Macro Prec</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("macro_recall")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Macro Rec</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("macro_f1")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem", color: "var(--accent-primary)" }}>
                    <span>Macro F1</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("biased_precision")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Biased Prec</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("biased_recall")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem", color: "#f59e0b" }}>
                    <span>Biased Rec</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th onClick={() => handleSort("biased_f1")} style={{ cursor: "pointer" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
                    <span>Biased F1</span>
                    <ArrowUpDown size={12} />
                  </div>
                </th>
                <th>Rank</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((m) => {
                const isEns = m.name.includes("Voting") || m.name.includes("Ensemble");
                return (
                  <tr
                    key={m.name}
                    onClick={() => setSelectedModelName(m.name)}
                    style={{
                      cursor: "pointer",
                      background: selectedModelName === m.name ? "rgba(99, 102, 241, 0.08)" : "transparent"
                    }}
                  >
                    <td>
                      <div style={{ fontWeight: 700, color: isEns ? "#f59e0b" : "var(--text-primary)" }}>
                        {m.name}
                      </div>
                      <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{m.type}</div>
                    </td>
                    <td>{(m.accuracy * 100).toFixed(2)}%</td>
                    <td>{(m.macro_precision * 100).toFixed(2)}%</td>
                    <td>{(m.macro_recall * 100).toFixed(2)}%</td>
                    <td style={{ fontWeight: 800, color: "var(--accent-primary)" }}>
                      {(m.macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td>{(m.biased_precision * 100).toFixed(2)}%</td>
                    <td style={{ fontWeight: 700, color: "#f59e0b" }}>
                      {(m.biased_recall * 100).toFixed(2)}%
                    </td>
                    <td style={{ fontWeight: 700 }}>{(m.biased_f1 * 100).toFixed(2)}%</td>
                    <td>
                      <span
                        style={{
                          fontSize: "0.75rem",
                          fontWeight: 700,
                          padding: "0.15rem 0.5rem",
                          borderRadius: "9999px",
                          background: m.rank === 1 ? "rgba(99, 102, 241, 0.2)" : m.rank === 2 ? "rgba(245, 158, 11, 0.2)" : "rgba(255, 255, 255, 0.05)",
                          color: m.rank === 1 ? "#a5b4fc" : m.rank === 2 ? "#fcd34d" : "var(--text-secondary)",
                          border: "1px solid var(--border-color)"
                        }}
                      >
                        #{m.rank}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrices Section */}
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.75rem" }}>
          <Layers size={20} color="#818cf8" />
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, margin: 0 }}>
            Test Confusion Matrices & Plain-Language Outcomes
          </h2>
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1.25rem" }}>
          Examining true vs. predicted classifications across the 508 test examples (252 Non-biased, 256 Biased).
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.25rem" }}>
          {/* Matrix 1: Weighted Soft Voting Ensemble */}
          <div style={{ background: "var(--bg-subtle)", border: "1px solid var(--border-color)", borderRadius: "0.75rem", padding: "1.25rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.95rem", color: "#f59e0b" }}>Weighted Soft Voting (Ensemble)</strong>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Threshold: 0.40</span>
            </div>

            <table className="words-table" style={{ textAlign: "center", marginBottom: "0.75rem" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Actual \ Pred</th>
                  <th style={{ textAlign: "center" }}>Pred: Non-biased</th>
                  <th style={{ textAlign: "center" }}>Pred: Biased</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Non-biased</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>197</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>55</td>
                </tr>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Biased</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>33</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>223</td>
                </tr>
              </tbody>
            </table>

            <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              <li><strong>197</strong> non-biased sentences correctly identified</li>
              <li><strong>55</strong> non-biased sentences mistakenly flagged as biased</li>
              <li><strong>33</strong> biased sentences missed by the ensemble</li>
              <li><strong>223</strong> biased sentences correctly detected as biased</li>
            </ul>
          </div>

          {/* Matrix 2: DeBERTa */}
          <div style={{ background: "var(--bg-subtle)", border: "1px solid var(--border-color)", borderRadius: "0.75rem", padding: "1.25rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.95rem", color: "#818cf8" }}>DeBERTa-v3-base</strong>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Threshold: 0.50</span>
            </div>

            <table className="words-table" style={{ textAlign: "center", marginBottom: "0.75rem" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Actual \ Pred</th>
                  <th style={{ textAlign: "center" }}>Pred: Non-biased</th>
                  <th style={{ textAlign: "center" }}>Pred: Biased</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Non-biased</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>211</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>41</td>
                </tr>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Biased</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>46</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>210</td>
                </tr>
              </tbody>
            </table>

            <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              <li><strong>211</strong> non-biased sentences correctly identified</li>
              <li><strong>41</strong> non-biased sentences mistakenly flagged as biased</li>
              <li><strong>46</strong> biased sentences missed by the model</li>
              <li><strong>210</strong> biased sentences correctly detected as biased</li>
            </ul>
          </div>

          {/* Matrix 3: RoBERTa */}
          <div style={{ background: "var(--bg-subtle)", border: "1px solid var(--border-color)", borderRadius: "0.75rem", padding: "1.25rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <strong style={{ fontSize: "0.95rem", color: "#a5b4fc" }}>RoBERTa-base</strong>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Threshold: 0.50</span>
            </div>

            <table className="words-table" style={{ textAlign: "center", marginBottom: "0.75rem" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Actual \ Pred</th>
                  <th style={{ textAlign: "center" }}>Pred: Non-biased</th>
                  <th style={{ textAlign: "center" }}>Pred: Biased</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Non-biased</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>211</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>41</td>
                </tr>
                <tr>
                  <td style={{ textAlign: "left", fontWeight: 600 }}>Actual: Biased</td>
                  <td style={{ color: "#f87171", background: "rgba(239, 68, 68, 0.08)" }}>50</td>
                  <td style={{ color: "#34d399", fontWeight: 700, background: "rgba(52, 211, 153, 0.08)" }}>206</td>
                </tr>
              </tbody>
            </table>

            <ul style={{ paddingLeft: "1.2rem", margin: 0, fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              <li><strong>211</strong> non-biased sentences correctly identified</li>
              <li><strong>41</strong> non-biased sentences mistakenly flagged as biased</li>
              <li><strong>50</strong> biased sentences missed by the model</li>
              <li><strong>206</strong> biased sentences correctly detected as biased</li>
            </ul>
          </div>

          {/* Traditional Baselines Note */}
          <div style={{ background: "rgba(15, 23, 42, 0.4)", border: "1px dashed var(--border-color)", borderRadius: "0.75rem", padding: "1.25rem", display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
              Traditional Baselines (Linear SVM, Logistic Regression, Naive Bayes)
            </div>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: 1.5, margin: 0 }}>
              Full 2x2 confusion matrices for individual traditional baselines were not persisted in the initial experiment logs.
              Aggregate precision, recall, and F1 metrics on the 508 test examples are verified and reported in the comparison table above.
            </p>
          </div>
        </div>
      </div>

      {/* Mandatory Evaluation Caveat */}
      <div
        style={{
          background: "rgba(99, 102, 241, 0.08)",
          border: "1px solid rgba(99, 102, 241, 0.25)",
          borderRadius: "0.75rem",
          padding: "1rem 1.25rem",
          fontSize: "0.84rem",
          color: "#c7d2fe",
          lineHeight: 1.5
        }}
      >
        <div style={{ fontWeight: 700, color: "#e0e7ff", marginBottom: "0.3rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <Info size={16} color="#818cf8" />
          <span>Evaluation Note & Interpretation Caveat:</span>
        </div>
        The reported test set has been examined across multiple experiments during development. These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. Performance on new data may differ.
      </div>
    </div>
  );
}
