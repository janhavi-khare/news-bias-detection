import logging
from typing import Optional

from src.models.inference import NewsBiasClassifier
from src.models.explainer import NewsBiasExplainer

logger = logging.getLogger(__name__)

_classifier: Optional[NewsBiasClassifier] = None
_explainer: Optional[NewsBiasExplainer] = None


def init_models() -> None:
    """Initialize and pre-load model into memory once during application startup."""
    global _classifier, _explainer
    if _classifier is None:
        logger.info("Initializing global NewsBiasClassifier...")
        _classifier = NewsBiasClassifier()
        _classifier.load_model()

    if _explainer is None:
        logger.info("Initializing global NewsBiasExplainer...")
        _explainer = NewsBiasExplainer(classifier=_classifier)


def get_classifier() -> NewsBiasClassifier:
    """Dependency provider for the singleton inference classifier."""
    global _classifier
    if _classifier is None or not _classifier.is_loaded():
        init_models()
    return _classifier


def get_explainer() -> NewsBiasExplainer:
    """Dependency provider for the singleton SHAP explainer."""
    global _explainer
    if _explainer is None:
        init_models()
    return _explainer
