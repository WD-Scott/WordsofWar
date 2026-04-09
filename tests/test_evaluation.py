"""
Tests for model evaluation utilities.
"""

import numpy as np

from words_of_war.evaluation import evaluate_binary_model


class _FakeModel:
    """Minimal model stub that returns pre-set predictions."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


class TestEvaluateBinaryModel:
    def test_returns_required_keys(self, known_predictions):
        y_true, y_pred_prob = known_predictions
        model = _FakeModel(y_pred_prob)
        results = evaluate_binary_model(model, np.zeros((len(y_true), 1)), y_true)
        expected_keys = {
            "auc_roc", "f1", "precision", "recall",
            "confusion_matrix", "y_pred_prob", "y_pred_class",
        }
        assert expected_keys == set(results.keys())

    def test_auc_roc_range(self, known_predictions):
        y_true, y_pred_prob = known_predictions
        model = _FakeModel(y_pred_prob)
        results = evaluate_binary_model(model, np.zeros((len(y_true), 1)), y_true)
        assert 0 <= results["auc_roc"] <= 1

    def test_f1_range(self, known_predictions):
        y_true, y_pred_prob = known_predictions
        model = _FakeModel(y_pred_prob)
        results = evaluate_binary_model(model, np.zeros((len(y_true), 1)), y_true)
        assert 0 <= results["f1"] <= 1

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_prob = np.array([0.0, 0.0, 1.0, 1.0])
        model = _FakeModel(y_pred_prob)
        results = evaluate_binary_model(model, np.zeros((4, 1)), y_true)
        assert results["auc_roc"] == 1.0
        assert results["f1"] == 1.0

    def test_custom_threshold(self, known_predictions):
        y_true, y_pred_prob = known_predictions
        model = _FakeModel(y_pred_prob)
        results_low = evaluate_binary_model(
            model, np.zeros((len(y_true), 1)), y_true, threshold=0.3
        )
        results_high = evaluate_binary_model(
            model, np.zeros((len(y_true), 1)), y_true, threshold=0.8
        )
        # Lower threshold -> more positive predictions -> higher recall
        assert results_low["recall"] >= results_high["recall"]

    def test_pred_class_matches_threshold(self):
        y_true = np.array([0, 1])
        y_pred_prob = np.array([0.6, 0.6])
        model = _FakeModel(y_pred_prob)
        results = evaluate_binary_model(
            model, np.zeros((2, 1)), y_true, threshold=0.5
        )
        np.testing.assert_array_equal(results["y_pred_class"], [1, 1])
