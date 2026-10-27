import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.config import ENSEMBLE_CONFIG_PATH

logger = logging.getLogger(__name__)

# Default ensemble configuration fallback (matching recorded experiments)
DEFAULT_WEIGHTS = {
    "linear_svm": 1.0,
    "logistic_regression": 1.0,
    "naive_bayes": 1.0,
    "roberta": 2.0,
    "deberta": 2.0
}
DEFAULT_THRESHOLD = 0.40


class WeightedSoftVotingEnsemble:
    """Computes weighted soft voting predictions across the five classifiers.

    Formula:
        P_ensemble(biased) = sum(w_i * P_i(biased)) / sum(w_i)
        Predicted Label = 'Biased' if P_ensemble(biased) >= threshold else 'Non-biased'
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or ENSEMBLE_CONFIG_PATH
        self.weights = DEFAULT_WEIGHTS.copy()
        self.threshold = DEFAULT_THRESHOLD
        self.metadata: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load ensemble weights and decision threshold from configuration file."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                raw_weights = cfg.get("weights", {})
                # Normalize keys to lowercase underscore (e.g. 'Linear SVM' -> 'linear_svm')
                normalized_weights = {}
                for k, v in raw_weights.items():
                    norm_k = k.lower().replace(" ", "_")
                    normalized_weights[norm_k] = float(v)

                if normalized_weights:
                    self.weights = normalized_weights

                self.threshold = float(cfg.get("threshold", DEFAULT_THRESHOLD))
                self.metadata = {
                    "validation_macro_f1": cfg.get("validation_macro_f1"),
                    "validation_accuracy": cfg.get("validation_accuracy"),
                    "model_order": cfg.get("model_order"),
                    "selection_metric": cfg.get("selection_metric")
                }
                logger.info(
                    f"Loaded ensemble config from {self.config_path}: "
                    f"weights={self.weights}, threshold={self.threshold}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to parse ensemble config at {self.config_path} ({e}). "
                    f"Falling back to default weights."
                )
        else:
            logger.info(f"Ensemble config not found at {self.config_path}. Using defaults.")

    def combine_probabilities(
        self,
        probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine individual model biased-class probabilities using weighted soft voting.

        Args:
            probabilities: Dict mapping model key ('linear_svm', 'roberta', etc.)
                           to its predicted P(biased) score in [0.0, 1.0].

        Returns:
            Dict containing:
                - probability_biased: float in [0.0, 1.0]
                - probability_non_biased: float in [0.0, 1.0]
                - label: 'Biased' or 'Non-biased'
                - threshold: float
                - weights_used: Dict of weights applied
                - denominator: total sum of weights applied
                - explanation: plain language formula representation
        """
        numerator = 0.0
        denominator = 0.0
        weights_used = {}

        for model_key, prob in probabilities.items():
            norm_key = model_key.lower().replace(" ", "_")
            weight = self.weights.get(norm_key, 1.0)
            numerator += weight * float(prob)
            denominator += weight
            weights_used[norm_key] = weight

        if denominator == 0.0:
            raise ValueError("No valid model weights found to compute ensemble probability.")

        p_biased = round(numerator / denominator, 4)
        p_non_biased = round(1.0 - p_biased, 4)
        predicted_label = "Biased" if p_biased >= self.threshold else "Non-biased"
        confidence = p_biased if predicted_label == "Biased" else p_non_biased

        explanation = (
            f"Weighted Soft Voting: P(biased) = {p_biased:.4f} "
            f"({'≥' if p_biased >= self.threshold else '<'} threshold {self.threshold:.2f} → {predicted_label}). "
            f"Weights: {weights_used}. Total weight sum: {denominator}."
        )

        return {
            "label": predicted_label,
            "predicted_label": predicted_label,
            "predicted_id": 1 if predicted_label == "Biased" else 0,
            "probability_biased": p_biased,
            "biased_probability": p_biased,
            "probability_non_biased": p_non_biased,
            "non_biased_probability": p_non_biased,
            "confidence": round(confidence, 4),
            "threshold": self.threshold,
            "weights": weights_used,
            "denominator": denominator,
            "explanation": explanation,
            "caveat": (
                "The ensemble score is a weighted model combination optimized for high biased-class recall. "
                "It represents estimated linguistic framing probability, not a calibrated real-world certainty."
            )
        }
