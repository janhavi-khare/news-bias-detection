import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.dependencies import (
    init_models,
    get_predictor,
    get_metrics_manager,
    get_classifier,
    get_explainer,
)
from app.predictor import MultiModelPredictor
from app.metrics import MetricsManager
from app.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelMetricsResponse,
    ModelsStatusResponse,
    ConfusionMatricesResponse,
    PredictRequest,
    PredictResponse,
)
from src import __version__
from src.models.explainer import NewsBiasExplainer
from src.utils.config import CORS_ORIGINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("newsbias.api")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load all 5 models and ensemble configuration once during startup."""
    logger.info("Application startup: loading all five models into memory...")
    try:
        init_models()
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}", exc_info=True)
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="News Bias Detection API",
    description=(
        "Multi-model news bias detection system powered by 5 classifiers "
        "(Linear SVM, Logistic Regression, Naive Bayes, RoBERTa, DeBERTa-v3) "
        "and a Weighted Soft-Voting Ensemble with word-level SHAP explainability."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount frontend assets if compiled dist is present
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "ValueError"}
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "error_type": "TypeError"}
    )


# -----------------------------------------------------------------------------
# Root & Health Endpoints
# -----------------------------------------------------------------------------
@app.get("/", summary="App / API Index")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return {
        "message": "News Bias Detection API is operational",
        "version": __version__,
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain": "/explain",
        "endpoints": {
            "health": ["/health", "/api/health"],
            "models": ["/models", "/api/models"],
            "predict": ["/predict", "/api/predict"],
            "explain": ["/explain", "/api/explain"],
            "metrics": ["/metrics", "/api/metrics"],
            "confusion_matrices": ["/confusion-matrices", "/api/confusion-matrices"],
            "comparison_csv": ["/api/comparison-csv"]
        }
    }


@app.get("/health", response_model=HealthResponse, summary="Server & Model Health Check")
@app.get("/api/health", response_model=HealthResponse, summary="API Health Check")
async def health_check(predictor: MultiModelPredictor = Depends(get_predictor)):
    status_info = predictor.get_models_status()
    primary_device = "cpu"
    if predictor.deberta_classifier and predictor.deberta_classifier.model:
        primary_device = str(predictor.deberta_classifier.device)

    return HealthResponse(
        status="ok",
        model_loaded=status_info["loaded_models_count"] > 0,
        device=primary_device,
        model_path=str(predictor.model_status.get("deberta", {}).get("path", "models/deberta")),
        id2label={"0": "Non-biased", "1": "Biased"},
        total_models=status_info["total_models"],
        loaded_models_count=status_info["loaded_models_count"],
        version=__version__
    )


@app.get("/models", response_model=ModelsStatusResponse, summary="Get Status of All 5 Models")
@app.get("/api/models", response_model=ModelsStatusResponse, summary="API Models Status")
async def get_models(predictor: MultiModelPredictor = Depends(get_predictor)):
    """Return status and configuration of all individual classifiers and ensemble."""
    return predictor.get_models_status()


# -----------------------------------------------------------------------------
# Prediction Endpoints
# -----------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, summary="Predict Bias with All 5 Models + Ensemble")
@app.post("/api/predict", response_model=PredictResponse, summary="API Predict Bias")
async def predict_bias(
    request: PredictRequest,
    predictor: MultiModelPredictor = Depends(get_predictor),
):
    """Predict bias for text using all 5 trained models and the weighted soft voting ensemble."""
    try:
        result = predictor.predict(
            text=request.text,
            include_explanation=request.include_explanation
        )
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


# -----------------------------------------------------------------------------
# SHAP Explanation Endpoints
# -----------------------------------------------------------------------------
@app.post("/explain", response_model=ExplainResponse, summary="Generate Word-Level SHAP Explanation")
@app.post("/api/explain", response_model=ExplainResponse, summary="API Word-Level SHAP Explanation")
async def explain_text(
    request: ExplainRequest,
    explainer: NewsBiasExplainer = Depends(get_explainer)
):
    """Generate SHAP token feature attributions towards the Biased class."""
    try:
        result = explainer.explain(
            text=request.text,
            top_n=request.top_n,
            max_evals=request.max_evals,
            sentence_index=request.sentence_index
        )
        return ExplainResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation calculation error: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Benchmark Metrics & Performance Endpoints
# -----------------------------------------------------------------------------
@app.get("/metrics", response_model=ModelMetricsResponse, summary="Recorded Model Performance Metrics")
@app.get("/api/metrics", response_model=ModelMetricsResponse, summary="API Performance Metrics")
async def get_metrics(metrics_mgr: MetricsManager = Depends(get_metrics_manager)):
    """Return historical recorded performance metrics for all models and ensemble."""
    data = metrics_mgr.get_metrics_data()
    return ModelMetricsResponse(
        model_architecture="DeBERTa-v3-base",
        dataset="BABE (sentence-level news media bias detection)",
        train_set_size=2579,
        validation_set_size=550,
        test_set_size=508,
        accuracy=0.8287,
        macro_f1=0.8287,
        validation_macro_f1=0.8309,
        macro_precision=0.8288,
        macro_recall=0.8288,
        classes={
            "Non-biased": {"precision": 0.8210, "recall": 0.8373, "f1": 0.8291, "support": 252},
            "Biased": {"precision": 0.8367, "recall": 0.8203, "f1": 0.8284, "support": 256}
        },
        confusion_matrix=[[211, 41], [46, 210]],
        notes=data.get("experiment_metadata", {}).get("evaluation_note", ""),
        experiment_metadata=data.get("experiment_metadata"),
        models=data.get("models")
    )


@app.get("/confusion-matrices", response_model=ConfusionMatricesResponse, summary="Available Confusion Matrices")
@app.get("/api/confusion-matrices", response_model=ConfusionMatricesResponse, summary="API Confusion Matrices")
async def get_confusion_matrices(metrics_mgr: MetricsManager = Depends(get_metrics_manager)):
    """Return confusion matrices from actual evaluation experiments."""
    return metrics_mgr.get_confusion_matrices()


@app.get("/api/comparison-csv", summary="Download Model Comparison as CSV")
@app.get("/comparison-csv", summary="Download Model Comparison as CSV")
async def download_comparison_csv(metrics_mgr: MetricsManager = Depends(get_metrics_manager)):
    """Download the benchmark model comparison table as a CSV file."""
    csv_content = metrics_mgr.get_comparison_csv_content()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="model_performance_comparison.csv"'}
    )
