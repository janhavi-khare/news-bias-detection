from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
    device: str = Field(..., examples=["cpu"])
    model_path: str = Field(..., examples=["model_deberta"])
    id2label: Dict[str, str] = Field(default_factory=dict)
    version: str = Field(..., examples=["0.1.0"])


class WordAttribution(BaseModel):
    word: str = Field(..., description="Cleaned token or word segment")
    score: float = Field(..., description="SHAP attribution value towards Biased class (class index 1)")
    absolute_importance: float = Field(..., description="Magnitude of attribution score")
    direction: str = Field(..., description="'Pushes toward Biased' or 'Pushes away from Biased'")


class TokenHighlight(BaseModel):
    token: str = Field(..., description="Individual token chunk from text masker")
    score: float = Field(..., description="Attribution value for token towards Biased class")
    direction: str = Field(..., description="Attribution direction")


class ExplainResponse(BaseModel):
    success: bool = Field(..., description="Whether SHAP explanation succeeded")
    target_sentence: Optional[str] = Field(None, description="The specific sentence explained")
    base_value: Optional[float] = Field(None, description="Expected baseline bias probability")
    top_words: List[WordAttribution] = Field(default_factory=list, description="Top influential words")
    token_highlights: List[TokenHighlight] = Field(default_factory=list, description="Tokens for visual highlighting")
    disclaimer: str = Field(..., description="Model interpretability caveat")
    error: Optional[str] = Field(None, description="Error message if explanation failed")


class SentencePrediction(BaseModel):
    index: int = Field(..., description="1-based index of sentence within the article")
    text: str = Field(..., description="Sentence text evaluated")
    predicted_label: str = Field(..., description="'Biased' or 'Non-biased'")
    predicted_id: int = Field(..., description="0=Non-biased, 1=Biased")
    confidence: float = Field(..., description="Confidence score for this sentence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities for this sentence")
    biased_probability: float = Field(..., description="Probability of Biased class")
    non_biased_probability: float = Field(..., description="Probability of Non-biased class")
    explanation: Optional[ExplainResponse] = Field(None, description="SHAP explanation for this sentence if requested")


class ArticleSummary(BaseModel):
    total_sentences: int = Field(..., description="Total count of sentences in article")
    biased_sentences_count: int = Field(..., description="Number of sentences classified as Biased")
    non_biased_sentences_count: int = Field(..., description="Number of sentences classified as Non-biased")
    biased_sentences_percentage: float = Field(..., description="Percentage of sentences classified as Biased (0-100)")
    mean_biased_probability: float = Field(..., description="Average Biased probability across all sentences")
    max_biased_probability: float = Field(..., description="Maximum Biased probability in any single sentence")
    bias_intensity: str = Field(..., description="Intensity tier: 'None / Minimal', 'Low / Isolated', 'Moderate', or 'High'")
    aggregation_method: str = Field(..., description="Detailed description of the article-level decision rule")


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
    predicted_label: str = Field(..., description="Article-level prediction: 'Biased' or 'Non-biased'")
    predicted_id: int = Field(..., description="Class integer ID (0=Non-biased, 1=Biased)")
    confidence: float = Field(..., description="Confidence score of the article-level prediction")
    probabilities: Dict[str, float] = Field(..., description="Article-level class probabilities")
    biased_probability: float = Field(..., description="Article-level probability score for Biased class")
    non_biased_probability: float = Field(..., description="Article-level probability score for Non-biased class")
    article_summary: Optional[ArticleSummary] = Field(None, description="Detailed sentence breakdown and aggregation metrics")
    sentences: List[SentencePrediction] = Field(default_factory=list, description="Independent sentence-level classifications")
    explanation: Optional[ExplainResponse] = Field(None, description="SHAP explanation for the most biased sentence if requested")


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
    sentence_index: Optional[int] = Field(None, ge=1, description="Specific sentence index to explain (1-based)")


class ModelMetricsResponse(BaseModel):
    model_architecture: str = "DeBERTa-v3-base"
    dataset: str = "BABE (sentence-level news media bias detection)"
    train_set_size: int = 2579
    validation_set_size: int = 550
    test_set_size: int = 508
    accuracy: float = 0.8287
    macro_f1: float = 0.8287
    validation_macro_f1: float = 0.8309
    macro_precision: float = 0.8288
    macro_recall: float = 0.8288
    classes: Dict[str, Dict[str, float]] = {
        "Non-biased": {"precision": 0.8210, "recall": 0.8373, "f1": 0.8291, "support": 252},
        "Biased": {"precision": 0.8367, "recall": 0.8203, "f1": 0.8284, "support": 256}
    }
    confusion_matrix: List[List[int]] = [[211, 41], [46, 210]]
    notes: str = (
        "Reported evaluation results on the BABE article-disjoint split. "
        "The reported test set has been examined across multiple experiments during development. "
        "These results should therefore be interpreted as exploratory rather than as a completely untouched final evaluation. "
        "Performance on new data may differ."
    )
