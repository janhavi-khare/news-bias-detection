import pytest
from src.models.inference import NewsBiasClassifier


@pytest.fixture(scope="module")
def classifier():
    clf = NewsBiasClassifier()
    clf.load_model()
    return clf


def test_model_loading(classifier):
    assert classifier.is_loaded()
    assert classifier.tokenizer is not None
    assert classifier.model is not None
    assert 0 in classifier.id2label
    assert 1 in classifier.id2label
    assert classifier.id2label[0] == "Non-biased"
    assert classifier.id2label[1] == "Biased"


def test_predict_neutral_text(classifier):
    text = "The government announced a new policy today, according to officials."
    result = classifier.predict(text)

    assert result["text"] == text
    assert result["predicted_label"] in ["Non-biased", "Biased"]
    assert result["predicted_id"] in [0, 1]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "Non-biased" in result["probabilities"]
    assert "Biased" in result["probabilities"]

    # Probabilities should sum to approximately 1.0
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-4

    # Neutral text should be classified as Non-biased
    assert result["predicted_label"] == "Non-biased"
    assert result["non_biased_probability"] > 0.5


def test_predict_biased_text(classifier):
    text = "The corrupt administration recklessly betrayed citizens with its disastrous decision."
    result = classifier.predict(text)

    assert result["predicted_label"] == "Biased"
    assert result["biased_probability"] > 0.5
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_empty_and_whitespace_input(classifier):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        classifier.predict("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        classifier.predict("   \n\t  ")


def test_predict_non_string_input(classifier):
    with pytest.raises(TypeError, match="must be a string"):
        classifier.predict(12345)


def test_predict_batch(classifier):
    texts = [
        "The committee met to review the quarterly budget report.",
        "Shameless politicians stole from hardworking families."
    ]
    results = classifier.predict_batch(texts)

    assert len(results) == 2
    for r in results:
        assert "predicted_label" in r
        assert "confidence" in r
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-4
