import logging
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
import torch

from src.models.inference import NewsBiasClassifier
from src.utils.config import SHAP_MAX_EVALS, SHAP_MAX_LENGTH
from src.utils.text import split_into_sentences, normalize_whitespace

logger = logging.getLogger(__name__)

# Stopwords excluded specifically from summary tables (matching explainability notebook)
DISPLAY_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at",
    "and", "or", "but", "is", "are", "was", "were",
    "with", "for", "by", "from", "according", "that", "this",
    "it", "as", "be", "has", "have", "had"
}


class NewsBiasExplainer:
    """SHAP-based explainability module for News Bias Detection.

    Reuses the exact word-level aggregation and attribution logic
    established in the project's explainability notebook.
    Targets class index 1 (Biased) consistently.
    """

    def __init__(
        self,
        classifier: NewsBiasClassifier,
        max_evals: int = SHAP_MAX_EVALS,
        max_length: int = SHAP_MAX_LENGTH,
    ):
        self.classifier = classifier
        self.max_evals = max_evals
        self.max_length = max_length
        self._explainer = None
        self._masker = None

    def _init_explainer(self) -> None:
        """Initialize SHAP Text masker and Explainer."""
        if not self.classifier.is_loaded():
            self.classifier.load_model()

        if self._explainer is not None:
            return

        logger.info("Initializing SHAP Text Explainer...")
        self._masker = shap.maskers.Text(self.classifier.tokenizer)
        self._explainer = shap.Explainer(
            self._predict_biased_probability,
            self._masker
        )
        logger.info("SHAP Explainer initialized successfully.")

    def _predict_biased_probability(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Prediction wrapper for SHAP, outputting Biased class probability (class index 1)."""
        if isinstance(texts, str):
            texts = [texts]

        results = []
        batch_size = 8
        device = self.classifier.device
        biased_idx = self.classifier.biased_index

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.classifier.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(device)

            with torch.no_grad():
                logits = self.classifier.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, biased_idx]

            results.extend(probs.detach().cpu().numpy().tolist())

        return np.array(results)

    def explain(
        self,
        text: str,
        top_n: int = 10,
        max_evals: Optional[int] = None,
        sentence_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate SHAP word-level explanation for text or a specific sentence.

        Returns:
        - target_sentence: the specific sentence text explained
        - base_value: expected biased probability baseline
        - predicted_biased_prob: model's biased probability for input
        - top_words: list of dicts with word, score, direction
        - token_highlights: full text tokenized segments with individual scores for UI rendering
        - disclaimer: technical caveat noting this is a model attribution, not objective truth
        """
        cleaned_text = self.classifier.validate_input(text)
        self._init_explainer()

        # Handle sentence targeting if multi-sentence article is provided
        sentences = split_into_sentences(cleaned_text)
        if not sentences:
            sentences = [cleaned_text]

        if sentence_index is not None and 1 <= sentence_index <= len(sentences):
            target_text = sentences[sentence_index - 1]
        elif len(sentences) > 1:
            # Pick the sentence with highest biased probability to explain
            preds = [self.classifier.predict_sentence(s) for s in sentences]
            highest_idx = int(np.argmax([p["biased_probability"] for p in preds]))
            target_text = sentences[highest_idx]
        else:
            target_text = sentences[0]

        evals = max_evals or self.max_evals

        try:
            shap_values = self._explainer([target_text], max_evals=evals)
            values_entry = shap_values[0]

            segments = [str(s) for s in values_entry.data]
            contributions = np.asarray(values_entry.values).reshape(-1).tolist()
            base_val = float(np.asarray(values_entry.base_values).ravel()[0]) if hasattr(values_entry, "base_values") else 0.5

            # Word-level importance matching notebook
            top_words = self._aggregate_word_importance(segments, contributions, top_n=top_n)

            # Build token highlights for inline frontend visualizer
            token_highlights = []
            for seg, score in zip(segments, contributions):
                if seg:
                    token_highlights.append({
                        "token": seg,
                        "score": round(float(score), 4),
                        "direction": "Pushes toward Biased" if score > 0 else "Pushes away from Biased"
                    })

            return {
                "success": True,
                "target_sentence": target_text,
                "base_value": round(base_val, 4),
                "top_words": top_words,
                "token_highlights": token_highlights,
                "disclaimer": (
                    "SHAP feature attributions reflect which words shifted this specific model's "
                    "prediction score toward or away from 'Biased'. They do not prove objective truth "
                    "or publisher ideology."
                )
            }

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}", exc_info=True)
            return {
                "success": False,
                "target_sentence": target_text,
                "error": str(e),
                "top_words": [],
                "token_highlights": [],
                "disclaimer": (
                    "Explanation could not be generated for this input due to an evaluation error."
                )
            }

    def _aggregate_word_importance(
        self,
        segments: List[str],
        contributions: List[float],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Merge subword tokens into complete words, filter stopwords, and rank by absolute impact."""
        words = []
        current_word = ""
        current_score = 0.0

        for segment, score in zip(segments, contributions):
            if not segment.strip():
                if current_word:
                    words.append((current_word, current_score))
                    current_word = ""
                    current_score = 0.0
                continue

            piece = segment.strip()
            current_word += piece
            current_score += float(score)

            if segment.endswith(" ") or (len(segment) > 1 and segment[-1].isspace()):
                words.append((current_word, current_score))
                current_word = ""
                current_score = 0.0

        if current_word:
            words.append((current_word, current_score))

        if not words:
            return []

        df = pd.DataFrame(words, columns=["token", "shap_value"])

        # Filter only for display; alphanumeric check
        df = df[df["token"].str.contains(r"[A-Za-z0-9]", regex=True)].copy()

        # Stopword filtering
        df = df[
            ~df["token"].str.lower().str.strip(".,!?;:'\"()[]{}").isin(DISPLAY_STOPWORDS)
        ].copy()

        if df.empty:
            return []

        df["direction"] = np.where(
            df["shap_value"] > 0,
            "Pushes toward Biased",
            "Pushes away from Biased"
        )
        df["absolute_importance"] = df["shap_value"].abs()

        ranked = (
            df.sort_values("absolute_importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        result = []
        for _, row in ranked.iterrows():
            result.append({
                "word": str(row["token"]),
                "score": round(float(row["shap_value"]), 4),
                "absolute_importance": round(float(row["absolute_importance"]), 4),
                "direction": str(row["direction"])
            })

        return result
