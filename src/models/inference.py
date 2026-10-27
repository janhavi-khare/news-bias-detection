import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import (
    DEVICE_SETTING,
    MAX_SEQUENCE_LENGTH,
    MODEL_DIR as DEFAULT_MODEL_DIR,
)
from src.utils.text import normalize_whitespace, split_into_sentences

logger = logging.getLogger(__name__)


class NewsBiasClassifier:
    """Reusable inference engine for the fine-tuned DeBERTa-v3-base News Bias classifier.

    Features sentence-level analysis matching the BABE training granularity,
    along with article-level aggregation and bias density scoring.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = 256,  # 256 matches the BABE training sequence length
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR
        self.max_length = max_length

        if device:
            self.device = torch.device(device)
        elif DEVICE_SETTING == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif DEVICE_SETTING == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.biased_index: int = 1

    def _verify_checkpoint(self) -> None:
        """Verify that all required checkpoint files exist locally."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint directory not found at '{self.model_path}'. "
                "Ensure model_deberta is present in the project."
            )

        config_file = self.model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model config file missing at '{config_file}'."
            )

        safetensors_file = self.model_path / "model.safetensors"
        bin_file = self.model_path / "pytorch_model.bin"
        if not safetensors_file.exists() and not bin_file.exists():
            raise FileNotFoundError(
                f"Model weights missing in '{self.model_path}'. "
                "Expected 'model.safetensors' or 'pytorch_model.bin'."
            )

    def load_model(self) -> None:
        """Load tokenizer and model from local checkpoint into memory."""
        if self.is_loaded():
            logger.info("Model is already loaded.")
            return

        self._verify_checkpoint()
        logger.info(f"Loading model and tokenizer from '{self.model_path}' onto {self.device}...")

        # local_files_only=True guarantees we do NOT download any external model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True
        ).to(self.device)

        self.model.eval()

        # Extract verified label mappings from config
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.label2id = {str(k): int(v) for k, v in self.model.config.label2id.items()}

        # Verify binary classification labels
        for idx, name in self.id2label.items():
            if "non" not in name.lower() and "bias" in name.lower():
                self.biased_index = idx
                break

        logger.info(f"Model loaded successfully on {self.device}. Labels: {self.id2label}")

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded and ready."""
        return self.model is not None and self.tokenizer is not None

    def validate_input(self, text: Union[str, any]) -> str:
        """Validate and clean input text, raising clear errors on invalid inputs."""
        if text is None:
            raise ValueError("Input text cannot be null or None.")

        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}.")

        cleaned_text = normalize_whitespace(text)
        if not cleaned_text:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(cleaned_text) > 50000:
            raise ValueError(
                f"Input text length ({len(cleaned_text)} characters) exceeds the maximum allowed limit of 50,000 characters."
            )

        return cleaned_text

    def predict_sentence(self, sentence: str) -> Dict[str, Any]:
        """Classify a single sentence matching BABE sentence-level training granularity."""
        if not self.is_loaded():
            self.load_model()

        cleaned_sentence = normalize_whitespace(sentence)

        inputs = self.tokenizer(
            cleaned_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities_tensor = torch.softmax(outputs.logits, dim=-1)[0]

        probabilities = probabilities_tensor.detach().cpu().numpy().tolist()
        pred_id = int(torch.argmax(probabilities_tensor).item())
        pred_label = self.id2label[pred_id]
        pred_confidence = float(probabilities[pred_id])

        class_probabilities = {
            self.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        return {
            "text": cleaned_sentence,
            "predicted_label": pred_label,
            "predicted_id": pred_id,
            "confidence": pred_confidence,
            "probabilities": class_probabilities,
            "biased_probability": float(probabilities[self.biased_index]),
            "non_biased_probability": float(probabilities[1 - self.biased_index])
        }

    def predict_sentences_batch(
        self,
        sentences: List[str],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """Classify multiple sentences in batches for high throughput."""
        if not self.is_loaded():
            self.load_model()

        results: List[Dict[str, Any]] = []

        for start_idx in range(0, len(sentences), batch_size):
            batch = sentences[start_idx : start_idx + batch_size]
            cleaned_batch = [normalize_whitespace(s) for s in batch]

            inputs = self.tokenizer(
                cleaned_batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

            for i, (text, prob) in enumerate(zip(cleaned_batch, probs)):
                pred_id = int(prob.argmax())
                pred_label = self.id2label[pred_id]
                results.append({
                    "index": start_idx + i + 1,
                    "text": text,
                    "predicted_label": pred_label,
                    "predicted_id": pred_id,
                    "confidence": float(prob[pred_id]),
                    "probabilities": {
                        self.id2label[j]: float(prob[j]) for j in range(len(prob))
                    },
                    "biased_probability": float(prob[self.biased_index]),
                    "non_biased_probability": float(prob[1 - self.biased_index])
                })

        return results

    def predict_article(
        self,
        text: str,
        include_explanation: bool = False,
        explainer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Perform sentence-level inference and article-level aggregation.

        Splits article into sentences, classifies each individually, and calculates
        aggregate article bias metrics with documented decision criteria.
        """
        if not self.is_loaded():
            self.load_model()

        validated_text = self.validate_input(text)
        sentences = split_into_sentences(validated_text)

        if not sentences:
            sentences = [validated_text]

        # Classify all sentences independently
        sentence_predictions = self.predict_sentences_batch(sentences)
        total_sentences = len(sentence_predictions)

        # Count biased and non-biased sentences
        biased_sentences = [
            s for s in sentence_predictions if s["predicted_label"] == "Biased"
        ]
        biased_count = len(biased_sentences)
        non_biased_count = total_sentences - biased_count
        biased_ratio = biased_count / total_sentences if total_sentences > 0 else 0.0
        biased_percentage = round(biased_ratio * 100, 2)

        biased_probs = [s["biased_probability"] for s in sentence_predictions]
        mean_biased_prob = float(np.mean(biased_probs)) if biased_probs else 0.0
        max_biased_prob = float(np.max(biased_probs)) if biased_probs else 0.0

        # Documented Article-Level Decision Logic:
        # An article is classified as Biased if:
        # 1. At least 25% of its sentences are classified as Biased, OR
        # 2. Mean biased probability across all sentences is >= 50%, OR
        # 3. For short text (<= 3 sentences), at least 1 sentence is Biased with ratio >= 33%.
        # An isolated biased sentence in a long article (< 25% ratio) does not arbitrarily label the whole piece as biased.
        if total_sentences == 1:
            article_label = sentence_predictions[0]["predicted_label"]
            article_id = sentence_predictions[0]["predicted_id"]
            article_prob_biased = sentence_predictions[0]["biased_probability"]
            article_prob_non_biased = sentence_predictions[0]["non_biased_probability"]
            aggregation_note = "Single sentence input evaluated directly."
        else:
            is_article_biased = (
                (biased_ratio >= 0.25)
                or (mean_biased_prob >= 0.50)
                or (total_sentences <= 3 and biased_count >= 1)
            )

            if is_article_biased:
                article_label = "Biased"
                article_id = self.biased_index
                aggregation_note = (
                    f"Classified as Biased: {biased_count} of {total_sentences} sentences "
                    f"({biased_percentage}%) show biased framing (exceeding the 25% article threshold), "
                    f"with mean biased probability {mean_biased_prob:.2%}."
                )
            else:
                article_label = "Non-biased"
                article_id = 1 - self.biased_index
                if biased_count > 0:
                    aggregation_note = (
                        f"Classified as Non-biased: although {biased_count} of {total_sentences} sentences "
                        f"({biased_percentage}%) showed biased phrasing, the article overall remains predominantly "
                        f"non-biased reporting (below the 25% threshold)."
                    )
                else:
                    aggregation_note = (
                        f"Classified as Non-biased: 0 of {total_sentences} sentences showed biased framing."
                    )

            # Article-level probability distribution reflects the average across sentences
            article_prob_biased = round(mean_biased_prob, 4)
            article_prob_non_biased = round(1.0 - article_prob_biased, 4)

        # Intensity scoring
        if biased_count == 0:
            bias_intensity = "None / Minimal"
        elif biased_count == 1 and biased_ratio < 0.25:
            bias_intensity = "Low / Isolated"
        elif biased_ratio <= 0.50:
            bias_intensity = "Moderate"
        else:
            bias_intensity = "High"

        # Optional SHAP Explanations
        top_explanation = None
        if include_explanation and explainer is not None:
            # Sort sentences by biased probability to explain the most significant sentences first
            sorted_by_bias = sorted(
                sentence_predictions,
                key=lambda s: s["biased_probability"],
                reverse=True
            )
            # Explain up to the top 2 highest biased sentences to maintain responsive execution
            sentences_to_explain = sorted_by_bias[:2] if sorted_by_bias else []
            for s in sentences_to_explain:
                try:
                    s_expl = explainer.explain(s["text"])
                    s["explanation"] = s_expl
                    if top_explanation is None and s_expl.get("success"):
                        top_explanation = s_expl
                except Exception as e:
                    logger.warning(f"Failed to explain sentence {s['index']}: {e}")

            # Fallback if no individual sentence explanation was set
            if top_explanation is None and sentences_to_explain:
                top_explanation = sentences_to_explain[0].get("explanation")

        confidence = (
            article_prob_biased if article_label == "Biased" else article_prob_non_biased
        )

        return {
            "text": validated_text,
            "predicted_label": article_label,
            "predicted_id": article_id,
            "confidence": float(confidence),
            "probabilities": {
                "Non-biased": float(article_prob_non_biased),
                "Biased": float(article_prob_biased)
            },
            "biased_probability": float(article_prob_biased),
            "non_biased_probability": float(article_prob_non_biased),
            "article_summary": {
                "total_sentences": total_sentences,
                "biased_sentences_count": biased_count,
                "non_biased_sentences_count": non_biased_count,
                "biased_sentences_percentage": biased_percentage,
                "mean_biased_probability": round(mean_biased_prob, 4),
                "max_biased_probability": round(max_biased_prob, 4),
                "bias_intensity": bias_intensity,
                "aggregation_method": aggregation_note
            },
            "sentences": sentence_predictions,
            "explanation": top_explanation
        }

    def predict(
        self,
        text: str,
        include_explanation: bool = False,
        explainer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Main inference entry point.

        Performs sentence-aware article classification and returns full backward-compatible
        prediction attributes along with sentence-level breakdown.
        """
        return self.predict_article(
            text=text,
            include_explanation=include_explanation,
            explainer=explainer
        )

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """Perform batch inference over multiple independent texts."""
        if not self.is_loaded():
            self.load_model()

        return [self.predict(t) for t in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "is_loaded": self.is_loaded(),
            "max_sequence_length": self.max_length,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "architecture": (
                self.model.config.architectures[0]
                if self.model and hasattr(self.model.config, "architectures") and self.model.config.architectures
                else "DebertaV2ForSequenceClassification"
            )
        }
