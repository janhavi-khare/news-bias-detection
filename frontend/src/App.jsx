import React, { useState, useEffect } from "react";
import { Navigation } from "./components/Navigation";
import { DashboardPage } from "./pages/DashboardPage";
import { BiasDetectionPage } from "./pages/BiasDetectionPage";
import { PerformancePage } from "./pages/PerformancePage";
import { AboutPage } from "./pages/AboutPage";
import { LimitationsNotice } from "./components/LimitationsNotice";
import { MetricsModal } from "./components/MetricsModal";
import { fetchHealth, fetchModelsStatus } from "./services/api";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [apiHealth, setApiHealth] = useState(null);
  const [modelsStatus, setModelsStatus] = useState(null);
  const [showMetricsModal, setShowMetricsModal] = useState(false);

  useEffect(() => {
    async function loadSystemStatus() {
      try {
        const health = await fetchHealth();
        setApiHealth({ online: true, ...health });
      } catch (err) {
        console.warn("Backend health check failed:", err);
        setApiHealth({ online: false, error: err.message });
      }

      try {
        const status = await fetchModelsStatus();
        setModelsStatus(status);
      } catch (err) {
        console.warn("Failed to fetch models status:", err);
      }
    }

    loadSystemStatus();
  }, []);

  return (
    <div className="app-container" style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top Navigation */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiHealth={apiHealth}
        modelsStatus={modelsStatus}
      />

      {/* Main Content Area */}
      <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 1.5rem", flex: 1, width: "100%" }}>
        {activeTab === "dashboard" && (
          <DashboardPage
            onNavigateToDetection={() => setActiveTab("detection")}
            modelsStatus={modelsStatus}
          />
        )}

        {activeTab === "detection" && (
          <BiasDetectionPage apiHealth={apiHealth} />
        )}

        {activeTab === "performance" && (
          <PerformancePage />
        )}

        {activeTab === "about" && (
          <AboutPage />
        )}

        <div style={{ marginTop: "3rem", marginBottom: "2rem" }}>
          <LimitationsNotice />
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid var(--border-color)",
          padding: "1.5rem",
          textAlign: "center",
          fontSize: "0.82rem",
          color: "var(--text-muted)",
          background: "rgba(15, 23, 42, 0.4)"
        }}
      >
        <div style={{ maxWidth: "1200px", margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
          <span>
            News Bias Detection — 5 Classifiers + Weighted Soft Voting (Threshold 0.40)
          </span>
          <div style={{ display: "flex", gap: "1rem" }}>
            <button
              onClick={() => setActiveTab("about")}
              style={{ background: "none", border: "none", color: "#818cf8", cursor: "pointer", fontSize: "0.82rem" }}
            >
              Methodology
            </button>
            <button
              onClick={() => setShowMetricsModal(true)}
              style={{ background: "none", border: "none", color: "#818cf8", cursor: "pointer", fontSize: "0.82rem" }}
            >
              Benchmark Modal
            </button>
          </div>
        </div>
      </footer>

      {/* Benchmark Metrics Modal */}
      <MetricsModal
        isOpen={showMetricsModal}
        onClose={() => setShowMetricsModal(false)}
      />
    </div>
  );
}
