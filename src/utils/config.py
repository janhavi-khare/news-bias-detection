import os
from pathlib import Path
from typing import List

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Models root directory
MODELS_DIR = BASE_DIR / "models"

# 1. Linear SVM
_default_svm = MODELS_DIR / "linear_svm.joblib"
SVM_MODEL_PATH = Path(os.getenv("SVM_MODEL_PATH", str(_default_svm)))
if not SVM_MODEL_PATH.is_absolute():
    SVM_MODEL_PATH = BASE_DIR / SVM_MODEL_PATH

# 2. Logistic Regression
_default_lr = MODELS_DIR / "logistic_regression.joblib"
LR_MODEL_PATH = Path(os.getenv("LR_MODEL_PATH", str(_default_lr)))
if not LR_MODEL_PATH.is_absolute():
    LR_MODEL_PATH = BASE_DIR / LR_MODEL_PATH

# 3. Multinomial Naive Bayes
_default_nb = MODELS_DIR / "naive_bayes.joblib"
NB_MODEL_PATH = Path(os.getenv("NB_MODEL_PATH", str(_default_nb)))
if not NB_MODEL_PATH.is_absolute():
    NB_MODEL_PATH = BASE_DIR / NB_MODEL_PATH

# 4. RoBERTa (check models/roberta first, then newsbias_roberta_best)
_default_roberta = MODELS_DIR / "roberta"
if not _default_roberta.exists() and (BASE_DIR / "newsbias_roberta_best").exists():
    _default_roberta = BASE_DIR / "newsbias_roberta_best"
ROBERTA_MODEL_DIR = Path(os.getenv("ROBERTA_MODEL_DIR", str(_default_roberta)))
if not ROBERTA_MODEL_DIR.is_absolute():
    ROBERTA_MODEL_DIR = BASE_DIR / ROBERTA_MODEL_DIR

# 5. DeBERTa (check models/deberta first, then model_deberta)
_default_deberta = MODELS_DIR / "deberta"
if not _default_deberta.exists() and (BASE_DIR / "model_deberta").exists():
    _default_deberta = BASE_DIR / "model_deberta"
DEBERTA_MODEL_DIR = Path(os.getenv("DEBERTA_MODEL_DIR", str(_default_deberta)))
if not DEBERTA_MODEL_DIR.is_absolute():
    DEBERTA_MODEL_DIR = BASE_DIR / DEBERTA_MODEL_DIR

# Primary model default (DeBERTa) for single-model backward compatibility
DEFAULT_MODEL_DIR = DEBERTA_MODEL_DIR
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = BASE_DIR / MODEL_DIR

# Ensemble configuration
_default_ensemble_cfg = MODELS_DIR / "ensemble_config.json"
ENSEMBLE_CONFIG_PATH = Path(os.getenv("ENSEMBLE_CONFIG_PATH", str(_default_ensemble_cfg)))
if not ENSEMBLE_CONFIG_PATH.is_absolute():
    ENSEMBLE_CONFIG_PATH = BASE_DIR / ENSEMBLE_CONFIG_PATH

# Results directory and files
RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "comparison_results.csv"
CONFUSION_MATRICES_PATH = RESULTS_DIR / "confusion_matrices.json"

# Hardware device
DEVICE_SETTING = os.getenv("DEVICE", "auto").lower()

# Preprocessing parameters
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
SHAP_MAX_LENGTH = int(os.getenv("SHAP_MAX_LENGTH", "256"))
SHAP_MAX_EVALS = int(os.getenv("SHAP_MAX_EVALS", "50"))

# Server configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
)
CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]
