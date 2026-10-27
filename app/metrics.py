import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.utils.config import METRICS_PATH, COMPARISON_CSV_PATH, CONFUSION_MATRICES_PATH

logger = logging.getLogger(__name__)


class MetricsManager:
    """Manages loading and serving the authoritative benchmark evaluation metrics,
    model comparison results, and confusion matrices.
    """

    def __init__(
        self,
        metrics_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        cm_path: Optional[Path] = None
    ):
        self.metrics_path = Path(metrics_path or METRICS_PATH)
        self.csv_path = Path(csv_path or COMPARISON_CSV_PATH)
        self.cm_path = Path(cm_path or CONFUSION_MATRICES_PATH)

        self._metrics_cache: Optional[Dict[str, Any]] = None
        self._cm_cache: Optional[Dict[str, Any]] = None

    def get_metrics_data(self) -> Dict[str, Any]:
        """Return full benchmark metadata and per-model metrics."""
        if self._metrics_cache is not None:
            return self._metrics_cache

        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self._metrics_cache = json.load(f)
                return self._metrics_cache
            except Exception as e:
                logger.error(f"Error reading metrics from {self.metrics_path}: {e}")

        # Fallback dictionary matching prompt Section 6 authoritative table
        return {
            "experiment_metadata": {
                "dataset": "BABE (sentence-level)",
                "test_set_size": 508,
                "validation_set_size": 550,
                "train_set_size": 2579,
                "evaluation_note": (
                    "The reported test set has been examined across multiple experiments during development. "
                    "These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. "
                    "Performance on new data may differ."
                )
            },
            "models": [
                {
                    "name": "DeBERTa",
                    "accuracy": 0.8287,
                    "macro_precision": 0.8288,
                    "macro_recall": 0.8288,
                    "macro_f1": 0.8287,
                    "biased_precision": 0.8367,
                    "biased_recall": 0.8203,
                    "biased_f1": 0.8284,
                    "rank": 1
                },
                {
                    "name": "Weighted Soft Voting",
                    "accuracy": 0.8268,
                    "macro_precision": 0.8293,
                    "macro_recall": 0.8264,
                    "macro_f1": 0.8263,
                    "biased_precision": 0.8022,
                    "biased_recall": 0.8711,
                    "biased_f1": 0.8352,
                    "rank": 2
                },
                {
                    "name": "RoBERTa",
                    "accuracy": 0.8209,
                    "macro_precision": 0.8212,
                    "macro_recall": 0.8210,
                    "macro_f1": 0.8208,
                    "biased_precision": 0.8340,
                    "biased_recall": 0.8047,
                    "biased_f1": 0.8191,
                    "rank": 3
                },
                {
                    "name": "Logistic Regression",
                    "accuracy": 0.7087,
                    "macro_precision": 0.7097,
                    "macro_recall": 0.7089,
                    "macro_f1": 0.7084,
                    "biased_precision": 0.7269,
                    "biased_recall": 0.6758,
                    "biased_f1": 0.7004,
                    "rank": 4
                },
                {
                    "name": "Linear SVM",
                    "accuracy": 0.7028,
                    "macro_precision": 0.7072,
                    "macro_recall": 0.7033,
                    "macro_f1": 0.7015,
                    "biased_precision": 0.7397,
                    "biased_recall": 0.6328,
                    "biased_f1": 0.6821,
                    "rank": 5
                },
                {
                    "name": "Naive Bayes",
                    "accuracy": 0.6870,
                    "macro_precision": 0.6896,
                    "macro_recall": 0.6874,
                    "macro_f1": 0.6862,
                    "biased_precision": 0.7137,
                    "biased_recall": 0.6328,
                    "biased_f1": 0.6708,
                    "rank": 6
                }
            ]
        }

    def get_confusion_matrices(self) -> Dict[str, Any]:
        """Return available test confusion matrices."""
        if self._cm_cache is not None:
            return self._cm_cache

        if self.cm_path.exists():
            try:
                with open(self.cm_path, "r", encoding="utf-8") as f:
                    self._cm_cache = json.load(f)
                return self._cm_cache
            except Exception as e:
                logger.error(f"Error reading confusion matrices from {self.cm_path}: {e}")

        # Fallback dictionary matching prompt requirements
        return {
            "classes": ["Non-biased", "Biased"],
            "matrices": {
                "weighted_soft_voting": {
                    "model_name": "Weighted Soft Voting",
                    "available": true,
                    "matrix": [[197, 55], [33, 223]]
                },
                "deberta": {
                    "model_name": "DeBERTa",
                    "available": true,
                    "matrix": [[211, 41], [46, 210]]
                },
                "roberta": {
                    "model_name": "RoBERTa",
                    "available": true,
                    "matrix": [[211, 41], [50, 206]]
                },
                "linear_svm": {"model_name": "Linear SVM", "available": False},
                "logistic_regression": {"model_name": "Logistic Regression", "available": False},
                "naive_bayes": {"model_name": "Naive Bayes", "available": False}
            }
        }

    def get_comparison_csv_content(self) -> str:
        """Return CSV formatted text for download."""
        if self.csv_path.exists():
            try:
                return self.csv_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading CSV file {self.csv_path}: {e}")

        # Construct from metrics data
        data = self.get_metrics_data()
        df = pd.DataFrame(data.get("models", []))
        cols = [
            "name", "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "biased_precision", "biased_recall", "biased_f1"
        ]
        available_cols = [c for c in cols if c in df.columns]
        df_sub = df[available_cols].rename(columns={
            "name": "Model",
            "accuracy": "Accuracy",
            "macro_precision": "Macro Precision",
            "macro_recall": "Macro Recall",
            "macro_f1": "Macro F1",
            "biased_precision": "Biased Precision",
            "biased_recall": "Biased Recall",
            "biased_f1": "Biased F1"
        })
        return df_sub.to_csv(index=False)
