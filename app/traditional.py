import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import joblib

from src.utils.config import SVM_MODEL_PATH, LR_MODEL_PATH, NB_MODEL_PATH

logger = logging.getLogger(__name__)


class TraditionalModelsManager:
    """Manages the lifecycle and inference for the three TF-IDF based baseline models:
    1. Linear SVM (wrapped with CalibratedClassifierCV for probability output)
    2. Logistic Regression (Pipeline with TfidfVectorizer)
    3. Multinomial Naive Bayes (Pipeline with TfidfVectorizer)
    """

    def __init__(
        self,
        svm_path: Optional[Path] = None,
        lr_path: Optional[Path] = None,
        nb_path: Optional[Path] = None
    ):
        self.svm_path = Path(svm_path or SVM_MODEL_PATH)
        self.lr_path = Path(lr_path or LR_MODEL_PATH)
        self.nb_path = Path(nb_path or NB_MODEL_PATH)

        self.svm_model = None
        self.lr_model = None
        self.nb_model = None

        self.loaded_status = {
            "linear_svm": False,
            "logistic_regression": False,
            "naive_bayes": False
        }
        self.load_errors = {}

    def load_models(self) -> None:
        """Load the three joblib models into memory."""
        # 1. Linear SVM
        try:
            if self.svm_path.exists():
                self.svm_model = joblib.load(self.svm_path)
                self.loaded_status["linear_svm"] = True
                logger.info(f"Loaded Linear SVM model from {self.svm_path}")
            else:
                self.load_errors["linear_svm"] = f"File not found: {self.svm_path}"
                logger.warning(f"Linear SVM model not found at {self.svm_path}")
        except Exception as e:
            self.load_errors["linear_svm"] = str(e)
            logger.error(f"Failed to load Linear SVM: {e}", exc_info=True)

        # 2. Logistic Regression
        try:
            if self.lr_path.exists():
                self.lr_model = joblib.load(self.lr_path)
                self.loaded_status["logistic_regression"] = True
                logger.info(f"Loaded Logistic Regression model from {self.lr_path}")
            else:
                self.load_errors["logistic_regression"] = f"File not found: {self.lr_path}"
                logger.warning(f"Logistic Regression model not found at {self.lr_path}")
        except Exception as e:
            self.load_errors["logistic_regression"] = str(e)
            logger.error(f"Failed to load Logistic Regression: {e}", exc_info=True)

        # 3. Multinomial Naive Bayes
        try:
            if self.nb_path.exists():
                self.nb_model = joblib.load(self.nb_path)
                self.loaded_status["naive_bayes"] = True
                logger.info(f"Loaded Naive Bayes model from {self.nb_path}")
            else:
                self.load_errors["naive_bayes"] = f"File not found: {self.nb_path}"
                logger.warning(f"Naive Bayes model not found at {self.nb_path}")
        except Exception as e:
            self.load_errors["naive_bayes"] = str(e)
            logger.error(f"Failed to load Naive Bayes: {e}", exc_info=True)

    def is_loaded(self, model_key: str) -> bool:
        return self.loaded_status.get(model_key, False)

    def predict_single(self, model_key: str, text: str) -> Dict[str, Any]:
        """Predict bias probability and label using a specific traditional classifier.

        Returns:
            Dict containing:
                - label: 'Biased' or 'Non-biased'
                - probability_biased: float
                - probability_non_biased: float
                - confidence: float
                - model_key: str
        """
        model = None
        if model_key == "linear_svm":
            model = self.svm_model
        elif model_key == "logistic_regression":
            model = self.lr_model
        elif model_key == "naive_bayes":
            model = self.nb_model
        else:
            raise ValueError(f"Unknown traditional model key: {model_key}")

        if model is None:
            raise RuntimeError(
                f"Model '{model_key}' is not loaded. Error: {self.load_errors.get(model_key, 'Model unavailable')}"
            )

        # Models expect an iterable of text strings
        probs = model.predict_proba([text])[0]
        # Classes are [0, 1] -> [Non-biased, Biased]
        p_non_biased = round(float(probs[0]), 4)
        p_biased = round(float(probs[1]), 4)

        # Standard decision threshold 0.5 for individual classifiers
        label = "Biased" if p_biased >= 0.5 else "Non-biased"
        confidence = p_biased if label == "Biased" else p_non_biased

        return {
            "label": label,
            "predicted_label": label,
            "probability_biased": p_biased,
            "biased_probability": p_biased,
            "probability_non_biased": p_non_biased,
            "non_biased_probability": p_non_biased,
            "confidence": round(confidence, 4)
        }

    def predict_all(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Predict with all available traditional models."""
        results = {}
        for key in ["linear_svm", "logistic_regression", "naive_bayes"]:
            if self.is_loaded(key):
                try:
                    results[key] = self.predict_single(key, text)
                except Exception as e:
                    logger.error(f"Inference failed for {key}: {e}")
                    results[key] = {
                        "error": str(e),
                        "label": "Unavailable",
                        "probability_biased": 0.5,
                        "probability_non_biased": 0.5,
                        "confidence": 0.5
                    }
            else:
                results[key] = {
                    "error": self.load_errors.get(key, "Model not loaded"),
                    "label": "Unavailable",
                    "probability_biased": 0.5,
                    "probability_non_biased": 0.5,
                    "confidence": 0.5
                }
        return results
