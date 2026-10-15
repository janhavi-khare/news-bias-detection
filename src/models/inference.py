import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import (
    DEVICE_SETTING,
    MAX_SEQUENCE_LENGTH,
    MODEL_DIR as DEFAULT_MODEL_DIR,
)

logger = logging.getLogger(__name__)


class NewsBiasClassifier:
    """Reusable inference engine for the fine-tuned RoBERTa News Bias classifier."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
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
                "Ensure newsbias_roberta_best is present in the project."
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

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        if len(cleaned_text) > 50000:
            raise ValueError(
                f"Input text length ({len(cleaned_text)} characters) exceeds the maximum allowed limit of 50,000 characters."
            )

        return cleaned_text

    def predict(self, text: str) -> Dict[str, any]:
        """Perform inference on a single text string.

        Returns a dictionary with:
        - text: cleaned input string
        - predicted_label: Biased or Non-biased
        - predicted_id: class integer (0 or 1)
        - confidence: probability score for predicted class
        - probabilities: mapping of class name -> probability (summing to 1.0)
        """
        if not self.is_loaded():
            self.load_model()

        cleaned_text = self.validate_input(text)

        inputs = self.tokenizer(
            cleaned_text,
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
            "text": cleaned_text,
            "predicted_label": pred_label,
            "predicted_id": pred_id,
            "confidence": pred_confidence,
            "probabilities": class_probabilities,
            "biased_probability": float(probabilities[self.biased_index]),
            "non_biased_probability": float(probabilities[1 - self.biased_index])
        }

    def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, any]]:
        """Perform batch inference over multiple texts."""
        if not self.is_loaded():
            self.load_model()

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            cleaned_batch = [self.validate_input(t) for t in batch_texts]

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

            for text, prob in zip(cleaned_batch, probs):
                pred_id = int(prob.argmax())
                pred_label = self.id2label[pred_id]
                results.append({
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

    def get_model_info(self) -> Dict[str, any]:
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
                else "RobertaForSequenceClassification"
            )
        }
