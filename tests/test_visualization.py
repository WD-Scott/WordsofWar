"""
Tests for visualization utilities.

Verifies that plot functions return Figure objects without raising.
Uses matplotlib's non-interactive Agg backend.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from words_of_war.visualization import (  # noqa: E402
    plot_class_balance,
    plot_class_boxplots,
    plot_class_distributions,
    plot_speech_length_distribution,
    plot_top_words,
)


@pytest.fixture
def eda_df():
    """DataFrame with columns needed by visualization functions."""
    return pd.DataFrame(
        {
            "Transcript": [
                "the president spoke about war and peace",
                "congress debated the new policy on trade",
                "the nation united under strong leadership",
                "war efforts continued across the country",
                "peace negotiations began in the capital",
                "the government passed new legislation today",
            ],
            "War": [0, 0, 0, 1, 1, 0],
            "Text Length": [38, 41, 42, 40, 39, 45],
        }
    )


class TestPlotSpeechLengthDistribution:
    def test_returns_figure(self, eda_df):
        fig = plot_speech_length_distribution(eda_df)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotClassDistributions:
    def test_returns_figure(self, eda_df):
        fig = plot_class_distributions(eda_df)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotClassBoxplots:
    def test_returns_figure(self, eda_df):
        fig = plot_class_boxplots(eda_df)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotTopWords:
    def test_returns_figure(self, eda_df):
        fig = plot_top_words(eda_df)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotClassBalance:
    def test_returns_figure(self):
        old = pd.Series({0: 880, 1: 84})
        new = pd.Series({0: 661, 1: 529})
        fig = plot_class_balance(old, new)
        assert isinstance(fig, Figure)
        plt.close(fig)
