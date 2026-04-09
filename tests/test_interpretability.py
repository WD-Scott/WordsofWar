"""
Tests for model interpretability utilities.

Tests cover attention analysis, LIME, and SHAP explainer creation.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from words_of_war.interpretability import (  # noqa: E402
    analyze_attention_by_class,
    create_lime_explainer,
    plot_attention_distribution,
)


# ── analyze_attention_by_class ──────────────────────────────────────────


class TestAnalyzeAttentionByClass:
    @pytest.fixture()
    def attention_data(self):
        rng = np.random.default_rng(42)
        weights = rng.standard_normal((20, 10)).astype(np.float32)
        labels = np.array([0] * 12 + [1] * 8)
        return weights, labels

    def test_returns_required_keys(self, attention_data):
        weights, labels = attention_data
        result = analyze_attention_by_class(weights, labels)
        assert set(result.keys()) == {
            "mean_class_0", "mean_class_1", "t_statistic", "p_value",
        }

    def test_mean_shapes(self, attention_data):
        weights, labels = attention_data
        result = analyze_attention_by_class(weights, labels)
        assert len(result["mean_class_0"]) == 12
        assert len(result["mean_class_1"]) == 8

    def test_statistic_types(self, attention_data):
        weights, labels = attention_data
        result = analyze_attention_by_class(weights, labels)
        assert isinstance(result["t_statistic"], float)
        assert isinstance(result["p_value"], float)


# ── plot_attention_distribution ─────────────────────────────────────────


class TestPlotAttentionDistribution:
    def test_returns_figure(self):
        rng = np.random.default_rng(42)
        wt_0 = rng.standard_normal(50).astype(np.float32)
        wt_1 = rng.standard_normal(30).astype(np.float32)
        fig = plot_attention_distribution(wt_0, wt_1)
        assert isinstance(fig, Figure)
        plt.close(fig)


# ── extract_attention_weights ───────────────────────────────────────────

tf = pytest.importorskip("tensorflow")


class TestExtractAttentionWeights:
    def test_output_shape(self):
        from words_of_war.interpretability import extract_attention_weights
        from words_of_war.models import build_lstm_attention

        input_dim = 20
        n_samples = 6
        model = build_lstm_attention(input_dim=input_dim)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(n_samples, input_dim).astype(np.float32)

        weights = extract_attention_weights(model, X)
        assert weights.shape[0] == n_samples


# ── create_lime_explainer ───────────────────────────────────────────────


class TestCreateLimeExplainer:
    @pytest.fixture()
    def training_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 5)).astype(np.float32)
        names = [f"feat_{i}" for i in range(5)]
        return X, names

    def test_returns_explainer(self, training_data):
        import lime.lime_tabular
        X, names = training_data
        explainer = create_lime_explainer(X, names)
        assert isinstance(explainer, lime.lime_tabular.LimeTabularExplainer)

    def test_default_class_names(self, training_data):
        X, names = training_data
        explainer = create_lime_explainer(X, names)
        assert explainer.class_names == ["Non-War", "War"]

    def test_custom_class_names(self, training_data):
        X, names = training_data
        custom = ["Peace", "Conflict"]
        explainer = create_lime_explainer(X, names, class_names=custom)
        assert explainer.class_names == custom


# ── create_shap_explainer ───────────────────────────────────────────────


class TestCreateShapExplainer:
    def test_returns_tuple(self):
        from unittest.mock import MagicMock, patch

        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 5)).astype(np.float32)

        mock_sv = rng.standard_normal((30, 5)).astype(np.float32)
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_sv

        mock_shap = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            from words_of_war.interpretability import create_shap_explainer
            model = MagicMock()
            explainer, shap_values = create_shap_explainer(model, X)
            assert shap_values.shape == (30, 5)
            mock_shap.Explainer.assert_called_once_with(model, X)
