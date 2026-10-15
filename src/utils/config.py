import os
from pathlib import Path
from typing import List

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Model configuration
DEFAULT_MODEL_DIR = BASE_DIR / "newsbias_roberta_best"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))

# Ensure relative path resolution from base directory if not absolute
if not MODEL_DIR.is_absolute():
    MODEL_DIR = BASE_DIR / MODEL_DIR

DEVICE_SETTING = os.getenv("DEVICE", "auto").lower()

# Preprocessing parameters (matching training and evaluation notebooks)
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
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
