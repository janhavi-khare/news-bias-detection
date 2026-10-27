import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.models.inference import NewsBiasClassifier
from src.models.explainer import NewsBiasExplainer
from src.utils.config import (
    ROBERTA_MODEL_DIR,
    DEBERTA_MODEL_DIR,
    SVM_MODEL_PATH,
    LR_MODEL_PATH,
    NB_MODEL_PATH,
    ENSEMBLE_CONFIG_PATH
)
from app.traditional import TraditionalModelsManager
from app.ensemble import WeightedSoftVotingEnsemble

logger = logging.getLogger(__name__)


class MultiModelPredictor:
    """Orchestrates loading, inference, and weighted soft voting across all 5 models:
    1. Linear SVM (TF-IDF + CalibratedClassifierCV)
    2. Logistic Regression (TF-IDF + LogisticRegression)
    3. Naive Bayes (TF-IDF + MultinomialNB)
    4. RoBERTa (Fine-tuned transformer encoder)
    5. DeBERTa (Fine-tuned transformer encoder, primary sentence analyzer)
    + Weighted Soft-Voting Ensemble (Threshold 0.40)
    """

    def __init__(self):
        self.traditional_manager = TraditionalModelsManager(
            svm_path=SVM_MODEL_PATH,
            lr_path=LR_MODEL_PATH,
            nb_path=NB_MODEL_PATH
        )
        self.roberta_classifier: Optional[NewsBiasClassifier] = None
        self.deberta_classifier: Optional[NewsBiasClassifier] = None
        self.explainer: Optional[NewsBiasExplainer] = None
        self.ensemble = WeightedSoftVotingEnsemble(config_path=ENSEMBLE_CONFIG_PATH)

        self.model_status: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    def load_all_models(self) -> None:
        """Load all five models into memory once at application startup."""
        if self.is_initialized:
            return

        logger.info("Starting MultiModelPredictor initialization...")
        t0 = time.time()

        # 1-3. Load Traditional Models
        self.traditional_manager.load_models()
        self.model_status["linear_svm"] = {
            "name": "Linear SVM",
            "type": "TF-IDF + Linear SVM (Calibrated)",
            "loaded": self.traditional_manager.is_loaded("linear_svm"),
            "path": str(SVM_MODEL_PATH),
            "weight": self.ensemble.weights.get("linear_svm", 1.0),
            "error": self.traditional_manager.load_errors.get("linear_svm")
        }
        self.model_status["logistic_regression"] = {
            "name": "Logistic Regression",
            "type": "TF-IDF + Logistic Regression",
            "loaded": self.traditional_manager.is_loaded("logistic_regression"),
            "path": str(LR_MODEL_PATH),
            "weight": self.ensemble.weights.get("logistic_regression", 1.0),
            "error": self.traditional_manager.load_errors.get("logistic_regression")
        }
        self.model_status["naive_bayes"] = {
            "name": "Naive Bayes",
            "type": "TF-IDF + Multinomial Naive Bayes",
            "loaded": self.traditional_manager.is_loaded("naive_bayes"),
            "path": str(NB_MODEL_PATH),
            "weight": self.ensemble.weights.get("naive_bayes", 1.0),
            "error": self.traditional_manager.load_errors.get("naive_bayes")
        }

        # 4. Load RoBERTa
        try:
            if ROBERTA_MODEL_DIR.exists():
                logger.info(f"Loading RoBERTa from {ROBERTA_MODEL_DIR}...")
                self.roberta_classifier = NewsBiasClassifier(model_path=ROBERTA_MODEL_DIR)
                self.roberta_classifier.load_model()
                self.model_status["roberta"] = {
                    "name": "RoBERTa",
                    "type": "Fine-tuned Transformer (FacebookAI/roberta-base)",
                    "loaded": True,
                    "path": str(ROBERTA_MODEL_DIR),
                    "weight": self.ensemble.weights.get("roberta", 2.0),
                    "device": str(self.roberta_classifier.device),
                    "error": None
                }
            else:
                self.model_status["roberta"] = {
                    "name": "RoBERTa",
                    "type": "Fine-tuned Transformer",
                    "loaded": False,
                    "path": str(ROBERTA_MODEL_DIR),
                    "weight": self.ensemble.weights.get("roberta", 2.0),
                    "error": f"Directory not found: {ROBERTA_MODEL_DIR}"
                }
        except Exception as e:
            logger.error(f"Failed to load RoBERTa: {e}", exc_info=True)
            self.model_status["roberta"] = {
                "name": "RoBERTa",
                "type": "Fine-tuned Transformer",
                "loaded": False,
                "path": str(ROBERTA_MODEL_DIR),
                "weight": self.ensemble.weights.get("roberta", 2.0),
                "error": str(e)
            }

        # 5. Load DeBERTa
        try:
            if DEBERTA_MODEL_DIR.exists():
                logger.info(f"Loading DeBERTa from {DEBERTA_MODEL_DIR}...")
                self.deberta_classifier = NewsBiasClassifier(model_path=DEBERTA_MODEL_DIR)
                self.deberta_classifier.load_model()
                self.model_status["deberta"] = {
                    "name": "DeBERTa",
                    "type": "Fine-tuned Transformer (microsoft/deberta-v3-base)",
                    "loaded": True,
                    "path": str(DEBERTA_MODEL_DIR),
                    "weight": self.ensemble.weights.get("deberta", 2.0),
                    "device": str(self.deberta_classifier.device),
                    "error": None
                }
            else:
                self.model_status["deberta"] = {
                    "name": "DeBERTa",
                    "type": "Fine-tuned Transformer",
                    "loaded": False,
                    "path": str(DEBERTA_MODEL_DIR),
                    "weight": self.ensemble.weights.get("deberta", 2.0),
                    "error": f"Directory not found: {DEBERTA_MODEL_DIR}"
                }
        except Exception as e:
            logger.error(f"Failed to load DeBERTa: {e}", exc_info=True)
            self.model_status["deberta"] = {
                "name": "DeBERTa",
                "type": "Fine-tuned Transformer",
                "loaded": False,
                "path": str(DEBERTA_MODEL_DIR),
                "weight": self.ensemble.weights.get("deberta", 2.0),
                "error": str(e)
            }

        # Initialize SHAP explainer on primary transformer (DeBERTa if available, else RoBERTa)
        primary_clf = self.deberta_classifier or self.roberta_classifier
        if primary_clf and primary_clf.is_loaded():
            try:
                logger.info("Initializing NewsBiasExplainer...")
                self.explainer = NewsBiasExplainer(classifier=primary_clf)
            except Exception as e:
                logger.warning(f"Could not initialize NewsBiasExplainer: {e}")

        # Ensemble metadata
        self.model_status["weighted_soft_voting"] = {
            "name": "Weighted Soft Voting",
            "type": "Ensemble",
            "loaded": True,
            "threshold": self.ensemble.threshold,
            "weights": self.ensemble.weights
        }

        self.is_initialized = True
        logger.info(f"MultiModelPredictor initialization finished in {time.time()-t0:.2f}s.")

    def get_models_status(self) -> Dict[str, Any]:
        """Return operational readiness and configuration for each model."""
        loaded_count = sum(
            1 for k, v in self.model_status.items()
            if k != "weighted_soft_voting" and v.get("loaded", False)
        )
        return {
            "total_models": 5,
            "loaded_models_count": loaded_count,
            "all_loaded": loaded_count == 5,
            "models": self.model_status,
            "ensemble_config": {
                "threshold": self.ensemble.threshold,
                "weights": self.ensemble.weights,
                "formula": "P(biased) = (1*SVM + 1*LR + 1*NB + 2*RoBERTa + 2*DeBERTa) / 7.0"
            }
        }

    def predict(
        self,
        text: str,
        include_explanation: bool = False,
        top_n: int = 10,
        max_evals: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform unified multi-model inference and weighted soft voting."""
        if not self.is_initialized:
            self.load_all_models()

        if not isinstance(text, str):
            raise TypeError("Input 'text' must be a valid string.")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input 'text' cannot be empty or whitespace-only.")

        individual_predictions: Dict[str, Dict[str, Any]] = {}
        probabilities_for_ensemble: Dict[str, float] = {}

        # 1. Linear SVM
        if self.traditional_manager.is_loaded("linear_svm"):
            res_svm = self.traditional_manager.predict_single("linear_svm", cleaned)
            res_svm["model_name"] = "Linear SVM"
            res_svm["model_type"] = "TF-IDF + Linear SVM (Calibrated)"
            individual_predictions["linear_svm"] = res_svm
            probabilities_for_ensemble["linear_svm"] = res_svm["probability_biased"]
        else:
            individual_predictions["linear_svm"] = {
                "error": "Model not loaded",
                "label": "Unavailable",
                "probability_biased": 0.5,
                "probability_non_biased": 0.5,
                "confidence": 0.5
            }

        # 2. Logistic Regression
        if self.traditional_manager.is_loaded("logistic_regression"):
            res_lr = self.traditional_manager.predict_single("logistic_regression", cleaned)
            res_lr["model_name"] = "Logistic Regression"
            res_lr["model_type"] = "TF-IDF + Logistic Regression"
            individual_predictions["logistic_regression"] = res_lr
            probabilities_for_ensemble["logistic_regression"] = res_lr["probability_biased"]
        else:
            individual_predictions["logistic_regression"] = {
                "error": "Model not loaded",
                "label": "Unavailable",
                "probability_biased": 0.5,
                "probability_non_biased": 0.5,
                "confidence": 0.5
            }

        # 3. Naive Bayes
        if self.traditional_manager.is_loaded("naive_bayes"):
            res_nb = self.traditional_manager.predict_single("naive_bayes", cleaned)
            res_nb["model_name"] = "Naive Bayes"
            res_nb["model_type"] = "TF-IDF + Multinomial Naive Bayes"
            individual_predictions["naive_bayes"] = res_nb
            probabilities_for_ensemble["naive_bayes"] = res_nb["probability_biased"]
        else:
            individual_predictions["naive_bayes"] = {
                "error": "Model not loaded",
                "label": "Unavailable",
                "probability_biased": 0.5,
                "probability_non_biased": 0.5,
                "confidence": 0.5
            }

        # 4. RoBERTa
        if self.roberta_classifier and self.roberta_classifier.is_loaded():
            res_rob_full = self.roberta_classifier.predict(cleaned)
            p_rob = float(res_rob_full["biased_probability"])
            individual_predictions["roberta"] = {
                "label": res_rob_full["predicted_label"],
                "predicted_label": res_rob_full["predicted_label"],
                "probability_biased": round(p_rob, 4),
                "biased_probability": round(p_rob, 4),
                "probability_non_biased": round(1.0 - p_rob, 4),
                "non_biased_probability": round(1.0 - p_rob, 4),
                "confidence": round(res_rob_full["confidence"], 4),
                "model_name": "RoBERTa",
                "model_type": "Fine-tuned Transformer (FacebookAI/roberta-base)"
            }
            probabilities_for_ensemble["roberta"] = p_rob
        else:
            individual_predictions["roberta"] = {
                "error": "Model not loaded",
                "label": "Unavailable",
                "probability_biased": 0.5,
                "probability_non_biased": 0.5,
                "confidence": 0.5
            }

        # 5. DeBERTa (Also serves primary sentence-level breakdown)
        deberta_full_result = None
        if self.deberta_classifier and self.deberta_classifier.is_loaded():
            deberta_full_result = self.deberta_classifier.predict(cleaned)
            p_deb = float(deberta_full_result["biased_probability"])
            individual_predictions["deberta"] = {
                "label": deberta_full_result["predicted_label"],
                "predicted_label": deberta_full_result["predicted_label"],
                "probability_biased": round(p_deb, 4),
                "biased_probability": round(p_deb, 4),
                "probability_non_biased": round(1.0 - p_deb, 4),
                "non_biased_probability": round(1.0 - p_deb, 4),
                "confidence": round(deberta_full_result["confidence"], 4),
                "model_name": "DeBERTa",
                "model_type": "Fine-tuned Transformer (microsoft/deberta-v3-base)"
            }
            probabilities_for_ensemble["deberta"] = p_deb
        else:
            individual_predictions["deberta"] = {
                "error": "Model not loaded",
                "label": "Unavailable",
                "probability_biased": 0.5,
                "probability_non_biased": 0.5,
                "confidence": 0.5
            }

        # Compute Weighted Soft Voting Ensemble
        ensemble_result = self.ensemble.combine_probabilities(probabilities_for_ensemble)

        # Generate SHAP explanation if requested
        explanation_data = None
        if include_explanation and self.explainer is not None:
            try:
                # If multi-sentence, explain the sentence with highest biased probability
                target_idx = None
                if deberta_full_result and deberta_full_result.get("sentences"):
                    sents = deberta_full_result["sentences"]
                    if len(sents) > 1:
                        most_biased = max(sents, key=lambda s: s.get("biased_probability", 0.0))
                        target_idx = most_biased["index"]

                explanation_data = self.explainer.explain(
                    text=cleaned,
                    top_n=top_n,
                    max_evals=max_evals,
                    sentence_index=target_idx
                )
            except Exception as e:
                logger.error(f"SHAP explanation generation error: {e}", exc_info=True)

        # Build response merging both requested format & backward-compatible structure
        response = {
            # Requested format
            "input_text": cleaned,
            "individual_predictions": individual_predictions,
            "ensemble": ensemble_result,

            # Backward-compatible fields
            "text": cleaned,
            "predicted_label": ensemble_result["label"],
            "predicted_id": ensemble_result["predicted_id"],
            "confidence": ensemble_result["confidence"],
            "probabilities": {
                "Non-biased": ensemble_result["probability_non_biased"],
                "Biased": ensemble_result["probability_biased"]
            },
            "biased_probability": ensemble_result["probability_biased"],
            "non_biased_probability": ensemble_result["probability_non_biased"],
            "article_summary": deberta_full_result["article_summary"] if deberta_full_result else None,
            "sentences": deberta_full_result["sentences"] if deberta_full_result else [],
            "explanation": explanation_data
        }

        return response
