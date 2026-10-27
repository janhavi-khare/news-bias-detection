import pytest
from app.ensemble import WeightedSoftVotingEnsemble, DEFAULT_WEIGHTS, DEFAULT_THRESHOLD


def test_ensemble_weights_and_threshold():
    ens = WeightedSoftVotingEnsemble()
    assert ens.weights["linear_svm"] == 1.0
    assert ens.weights["logistic_regression"] == 1.0
    assert ens.weights["naive_bayes"] == 1.0
    assert ens.weights["roberta"] == 2.0
    assert ens.weights["deberta"] == 2.0
    assert ens.threshold == 0.40


def test_ensemble_manual_calculation_at_threshold():
    ens = WeightedSoftVotingEnsemble()
    # (1*0.2 + 1*0.3 + 1*0.1 + 2*0.5 + 2*0.6) / 7 = 2.8 / 7 = 0.4000
    probs = {
        "linear_svm": 0.2,
        "logistic_regression": 0.3,
        "naive_bayes": 0.1,
        "roberta": 0.5,
        "deberta": 0.6
    }
    res = ens.combine_probabilities(probs)
    assert res["probability_biased"] == 0.4000
    assert res["label"] == "Biased"
    assert res["predicted_id"] == 1
    assert res["probability_non_biased"] == 0.6000


def test_ensemble_manual_calculation_below_threshold():
    ens = WeightedSoftVotingEnsemble()
    # (1*0.1 + 1*0.2 + 1*0.1 + 2*0.2 + 2*0.2) / 7 = 1.2 / 7 = 0.1714
    probs = {
        "linear_svm": 0.1,
        "logistic_regression": 0.2,
        "naive_bayes": 0.1,
        "roberta": 0.2,
        "deberta": 0.2
    }
    res = ens.combine_probabilities(probs)
    assert res["probability_biased"] == 0.1714
    assert res["label"] == "Non-biased"
    assert res["predicted_id"] == 0
    assert res["probability_non_biased"] == 0.8286


def test_ensemble_manual_calculation_strongly_biased():
    ens = WeightedSoftVotingEnsemble()
    # (1*0.9 + 1*0.8 + 1*0.9 + 2*0.95 + 2*0.98) / 7 = (2.6 + 1.9 + 1.96) / 7 = 6.46 / 7 = 0.9229
    probs = {
        "linear_svm": 0.9,
        "logistic_regression": 0.8,
        "naive_bayes": 0.9,
        "roberta": 0.95,
        "deberta": 0.98
    }
    res = ens.combine_probabilities(probs)
    assert res["probability_biased"] == 0.9229
    assert res["label"] == "Biased"
    assert res["predicted_id"] == 1
