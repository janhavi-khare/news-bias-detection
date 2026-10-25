import pytest
from src.models.inference import NewsBiasClassifier
from src.utils.text import split_into_sentences, normalize_whitespace


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


def test_predict_neutral_single_sentence(classifier):
    text = "The government announced a new policy today, according to officials."
    result = classifier.predict(text)

    print(f"\n[Single Neutral Prediction]: Label={result['predicted_label']}, "
          f"Probabilities={result['probabilities']}")

    assert result["predicted_label"] == "Non-biased"
    assert result["predicted_id"] == 0
    assert result["non_biased_probability"] > 0.5
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    assert result["article_summary"]["total_sentences"] == 1


def test_predict_biased_single_sentence(classifier):
    text = "The corrupt administration recklessly betrayed citizens with its disastrous decision."
    result = classifier.predict(text)

    print(f"\n[Single Loaded Prediction]: Label={result['predicted_label']}, "
          f"Probabilities={result['probabilities']}")

    assert result["predicted_label"] == "Biased"
    assert result["predicted_id"] == 1
    assert result["biased_probability"] > 0.5
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["article_summary"]["total_sentences"] == 1


def test_predict_neutral_article(classifier):
    article = (
        "The committee met on Thursday to review the quarterly fiscal report. "
        "Independent analysts projected moderate economic growth across key industrial sectors. "
        "Officials concluded the meeting by scheduling public hearings for next month."
    )
    result = classifier.predict(article)
    summary = result["article_summary"]

    print(f"\n[Neutral Article Prediction]: Label={result['predicted_label']}, "
          f"BiasedCount={summary['biased_sentences_count']}/{summary['total_sentences']}, "
          f"MeanBiasedProb={summary['mean_biased_probability']:.4f}")

    assert summary["total_sentences"] == 3
    assert summary["biased_sentences_count"] == 0
    assert summary["biased_sentences_percentage"] == 0.0
    assert result["predicted_label"] == "Non-biased"
    assert result["non_biased_probability"] > 0.5
    assert len(result["sentences"]) == 3
    for s in result["sentences"]:
        assert s["predicted_label"] == "Non-biased"


def test_predict_explicitly_loaded_article(classifier):
    article = (
        "The corrupt administration pushed through a disastrous policy yesterday. "
        "The reckless leadership shamelessly betrayed hardworking ordinary citizens. "
        "Their outrageous proposal threatens vulnerable families across the nation."
    )
    result = classifier.predict(article)
    summary = result["article_summary"]

    print(f"\n[Loaded Article Prediction]: Label={result['predicted_label']}, "
          f"BiasedCount={summary['biased_sentences_count']}/{summary['total_sentences']}, "
          f"MeanBiasedProb={summary['mean_biased_probability']:.4f}")

    assert summary["total_sentences"] == 3
    assert summary["biased_sentences_count"] >= 2
    assert summary["biased_sentences_percentage"] >= 66.0
    assert result["predicted_label"] == "Biased"
    assert result["biased_probability"] > 0.5


def test_predict_mixed_article_isolated_bias(classifier):
    """Verify that an isolated biased sentence in an otherwise neutral article does not arbitrarily make the entire article Biased."""
    article = (
        "The municipal water department released its annual infrastructure inspection report on Tuesday. "
        "Engineers found that three major pump stations require scheduled pump maintenance over the winter. "
        "Independent water quality tests showed full compliance with federal safety regulations. "
        "However, critics argued that the corrupt administration pushed through a disastrous contract giveaway. "
        "The city council agreed to allocate funds from the capital improvement reserve for the repairs."
    )
    result = classifier.predict(article)
    summary = result["article_summary"]

    print(f"\n[Mixed Article Prediction]: Label={result['predicted_label']}, "
          f"BiasedCount={summary['biased_sentences_count']}/{summary['total_sentences']} "
          f"({summary['biased_sentences_percentage']}%), Intensity={summary['bias_intensity']}")

    assert summary["total_sentences"] == 5
    # Sentence 4 contains 'corrupt' and 'disastrous', which is detected as Biased
    assert summary["biased_sentences_count"] == 1
    assert summary["biased_sentences_percentage"] == 20.0
    # Because 1 out of 5 (20%) is below the 25% article threshold, article remains Non-biased overall
    assert result["predicted_label"] == "Non-biased"
    assert summary["bias_intensity"] == "Low / Isolated"
    assert "below the 25% threshold" in summary["aggregation_method"]


def test_predict_long_article_batching(classifier):
    """Verify that long multi-sentence articles are safely processed in batches without dropping sentences."""
    sentences = [
        f"Paragraph sentence number {i} provides standard factual documentation of the events."
        for i in range(1, 15)
    ]
    long_article = " ".join(sentences)

    result = classifier.predict(long_article)
    summary = result["article_summary"]

    print(f"\n[Long Article Prediction]: Processed {summary['total_sentences']} sentences. "
          f"Mean Biased Prob={summary['mean_biased_probability']:.4f}")

    assert summary["total_sentences"] == 14
    assert len(result["sentences"]) == 14
    assert result["predicted_label"] == "Non-biased"
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_edge_cases(classifier):
    # Empty string
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        classifier.predict("")

    # Whitespace only
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        classifier.predict("   \n\t  ")

    # Non-string input
    with pytest.raises(TypeError, match="must be a string"):
        classifier.predict(12345)

    # Short sentence with numbers and quotes
    short_text = '"Inflation rose by 0.3% in September," officials confirmed.'
    res = classifier.predict(short_text)
    assert res["predicted_label"] in ["Non-biased", "Biased"]
    assert res["article_summary"]["total_sentences"] == 1


def test_text_splitting_and_normalization():
    raw = "  WASHINGTON — First   sentence here.   \n\nSecond sentence begins! Third?  "
    norm = normalize_whitespace(raw)
    assert "\n" not in norm
    assert "   " not in norm

    sents = split_into_sentences(raw)
    assert len(sents) >= 2
    for s in sents:
        assert s == s.strip()
