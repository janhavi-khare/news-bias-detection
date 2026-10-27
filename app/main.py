import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.dependencies import get_classifier, get_explainer, init_models
from app.schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelMetricsResponse,
    PredictRequest,
    PredictResponse,
)
from src import __version__
from src.models.explainer import NewsBiasExplainer
from src.models.inference import NewsBiasClassifier
from src.utils.config import CORS_ORIGINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("newsbias.api")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load model during application startup to avoid per-request latency."""
    logger.info("Application startup: loading model and tokenizer into memory...")
    try:
        init_models()
        logger.info("Model initialization complete.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="News Bias Detection API",
    description=(
        "Binary text classification API detecting bias in news text "
        "using fine-tuned DeBERTa-v3-base with word-level SHAP explainability."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS for local frontend communication
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
        "explain": "/explain"
    }


@app.get("/health", response_model=HealthResponse, summary="API & Model Health Check")
async def health_check(classifier: NewsBiasClassifier = Depends(get_classifier)):
    info = classifier.get_model_info()
    return HealthResponse(
        status="ok",
        model_loaded=info["is_loaded"],
        device=info["device"],
        model_path=info["model_path"],
        id2label={str(k): v for k, v in info["id2label"].items()},
        version=__version__
    )


@app.post("/predict", response_model=PredictResponse, summary="Predict Bias for Input Text")
async def predict_bias(
    request: PredictRequest,
    classifier: NewsBiasClassifier = Depends(get_classifier),
    explainer: NewsBiasExplainer = Depends(get_explainer),
):
    try:
        result = classifier.predict(
            text=request.text,
            include_explanation=request.include_explanation,
            explainer=explainer if request.include_explanation else None
        )
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/explain", response_model=ExplainResponse, summary="Generate SHAP Explanation for Text")
async def explain_text(
    request: ExplainRequest,
    explainer: NewsBiasExplainer = Depends(get_explainer)
):
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


@app.get("/metrics", response_model=ModelMetricsResponse, summary="Reported Model Performance Metrics")
async def get_metrics():
    """Return the previously reported evaluation benchmark metrics on the test split."""
    return ModelMetricsResponse()
