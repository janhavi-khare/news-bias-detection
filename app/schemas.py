from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
    device: str = Field(..., examples=["cpu"])
    model_path: str = Field(..., examples=["newsbias_roberta_best"])
    id2label: Dict[str, str] = Field(default_factory=dict)
    version: str = Field(..., examples=["0.1.0"])


class WordAttribution(BaseModel):
    word: str = Field(..., description="Cleaned token or word segment")
    score: float = Field(..., description="SHAP attribution value towards Biased class")
    absolute_importance: float = Field(..., description="Magnitude of attribution score")
    direction: str = Field(..., description="'Pushes toward Biased' or 'Pushes away from Biased'")


class TokenHighlight(BaseModel):
    token: str = Field(..., description="Individual token chunk from text masker")
    score: float = Field(..., description="Attribution value for token")
    direction: str = Field(..., description="Attribution direction")


class ExplainResponse(BaseModel):
    success: bool = Field(..., description="Whether SHAP explanation succeeded")
    base_value: Optional[float] = Field(None, description="Expected baseline bias probability")
    top_words: List[WordAttribution] = Field(default_factory=list, description="Top influential words")
    token_highlights: List[TokenHighlight] = Field(default_factory=list, description="Tokens for visual highlighting")
    disclaimer: str = Field(..., description="Model interpretability caveat")
    error: Optional[str] = Field(None, description="Error message if explanation failed")


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="News headline, paragraph, or article text to evaluate for bias",
        examples=["The corrupt government recklessly betrayed ordinary citizens with its disastrous decision."]
    )
    include_explanation: bool = Field(
        False,
        description="Whether to also calculate and return SHAP token attributions"
    )


class PredictResponse(BaseModel):
    text: str = Field(..., description="Sanitized input text analyzed")
    predicted_label: str = Field(..., description="'Biased' or 'Non-biased'")
    predicted_id: int = Field(..., description="Class integer ID (0=Non-biased, 1=Biased)")
    confidence: float = Field(..., description="Probability of the predicted class (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Class name to probability mapping")
    biased_probability: float = Field(..., description="Probability score for the Biased class")
    non_biased_probability: float = Field(..., description="Probability score for the Non-biased class")
    explanation: Optional[ExplainResponse] = Field(None, description="SHAP explanation if requested")


class ExplainRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="News text to explain using SHAP attributions",
        examples=["The corrupt administration recklessly betrayed citizens."]
    )
    top_n: int = Field(10, ge=1, le=50, description="Maximum number of top words to rank")
    max_evals: Optional[int] = Field(None, ge=10, le=200, description="Evaluation budget for SHAP")


class ModelMetricsResponse(BaseModel):
    test_set_size: int = 527
    accuracy: float = 0.8197
    macro_precision: float = 0.8196
    macro_recall: float = 0.8176
    macro_f1: float = 0.8184
    classes: Dict[str, Dict[str, float]] = {
        "Non-biased": {"precision": 0.8213, "recall": 0.8475, "f1": 0.8342, "support": 282},
        "Biased": {"precision": 0.8178, "recall": 0.7878, "f1": 0.8025, "support": 245}
    }
    confusion_matrix: List[List[int]] = [[239, 43], [52, 193]]
    notes: str = "Previously reported evaluation results on the held-out test split."
