import React from "react";
import { LayoutDashboard, Newspaper, BarChart3, Info, CheckCircle2, AlertCircle } from "lucide-react";

export function Navigation({ activeTab, setActiveTab, apiHealth, modelsStatus }) {
  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "detection", label: "Bias Detection", icon: Newspaper },
    { id: "performance", label: "Model Performance", icon: BarChart3 },
    { id: "about", label: "About & Methodology", icon: Info }
  ];

  const loadedCount = modelsStatus?.loaded_models_count ?? (apiHealth?.online ? 5 : 0);
  const totalCount = modelsStatus?.total_models ?? 5;

  return (
    <nav
      style={{
        background: "rgba(15, 23, 42, 0.85)",
        backdropFilter: "blur(8px)",
        borderBottom: "1px solid var(--border-color)",
        position: "sticky",
        top: 0,
        zIndex: 40,
        padding: "0.75rem 1.5rem",
        marginBottom: "2rem"
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1rem"
        }}
      >
        {/* Brand / Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "8px",
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: "1rem",
              color: "#fff"
            }}
          >
            NB
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: "1.05rem", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
              NewsBias AI
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
              5-Model Weighted Soft-Voting Ensemble
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            background: "rgba(30, 41, 59, 0.6)",
            padding: "0.25rem",
            borderRadius: "0.65rem",
            border: "1px solid var(--border-color)",
            gap: "0.25rem"
          }}
        >
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.45rem",
                  padding: "0.45rem 0.85rem",
                  borderRadius: "0.5rem",
                  border: "none",
                  background: isActive ? "var(--accent-primary)" : "transparent",
                  color: isActive ? "#ffffff" : "var(--text-secondary)",
                  fontWeight: isActive ? 600 : 500,
                  fontSize: "0.85rem",
                  cursor: "pointer",
                  transition: "all 0.15s ease"
                }}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* System Health Status */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.8rem" }}>
          {apiHealth?.online ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.35rem",
                padding: "0.25rem 0.65rem",
                borderRadius: "9999px",
                background: "rgba(16, 185, 129, 0.1)",
                border: "1px solid rgba(16, 185, 129, 0.3)",
                color: "#34d399",
                fontWeight: 600
              }}
            >
              <CheckCircle2 size={13} />
              <span>{loadedCount}/{totalCount} Models Active</span>
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.35rem",
                padding: "0.25rem 0.65rem",
                borderRadius: "9999px",
                background: "rgba(239, 68, 68, 0.1)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#f87171",
                fontWeight: 600
              }}
            >
              <AlertCircle size={13} />
              <span>Backend Offline</span>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
