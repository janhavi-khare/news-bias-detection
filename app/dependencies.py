import logging
from typing import Optional

from src.models.inference import NewsBiasClassifier
from src.models.explainer import NewsBiasExplainer
from app.predictor import MultiModelPredictor
from app.metrics import MetricsManager

logger = logging.getLogger(__name__)

_predictor: Optional[MultiModelPredictor] = None
_metrics_manager: Optional[MetricsManager] = None


def init_models() -> None:
    """Initialize and pre-load all models into memory once during application startup."""
    global _predictor, _metrics_manager
    if _predictor is None:
        logger.info("Initializing MultiModelPredictor...")
        _predictor = MultiModelPredictor()
        _predictor.load_all_models()

    if _metrics_manager is None:
        _metrics_manager = MetricsManager()


def get_predictor() -> MultiModelPredictor:
    """Dependency provider for the singleton MultiModelPredictor."""
    global _predictor
    if _predictor is None or not _predictor.is_initialized:
        init_models()
    return _predictor


def get_metrics_manager() -> MetricsManager:
    """Dependency provider for the benchmark metrics manager."""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


def get_classifier() -> NewsBiasClassifier:
    """Dependency provider for primary inference classifier (DeBERTa)."""
    predictor = get_predictor()
    if predictor.deberta_classifier and predictor.deberta_classifier.is_loaded():
        return predictor.deberta_classifier
    elif predictor.roberta_classifier and predictor.roberta_classifier.is_loaded():
        return predictor.roberta_classifier
    raise RuntimeError("No transformer classifier is loaded.")


def get_explainer() -> NewsBiasExplainer:
    """Dependency provider for the singleton SHAP explainer."""
    predictor = get_predictor()
    if predictor.explainer is not None:
        return predictor.explainer
    classifier = get_classifier()
    predictor.explainer = NewsBiasExplainer(classifier=classifier)
    return predictor.explainer
