import React from "react";
import { AlertTriangle } from "lucide-react";

export function LimitationsNotice() {
  return (
    <div className="limitations-box">
      <div className="limitations-title">
        <AlertTriangle size={16} color="#f59e0b" />
        Academic Scope & Model Limitations
      </div>
      <p>
        This model was fine-tuned to detect linguistic and framing bias patterns (Biased vs. Non-biased)
        learned from its training dataset. It does <strong>not</strong> determine the factual truth or veracity
        of any statement, nor does it classify publishers into left-wing, center, or right-wing political ideologies.
        Predictions reflect statistical token correlations and may make errors on nuanced, satirical, or domain-specific text.
      </p>
    </div>
  );
}
