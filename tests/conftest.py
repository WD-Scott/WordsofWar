"""
Shared fixtures for the WordsofWar test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_speech_df():
    """
    Small synthetic DataFrame mimicking the presidential speeches dataset.
    """
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "1810-01-01",
                    "1812-03-01",
                    "1850-06-15",
                    "1917-01-01",
                    "1990-05-01",
                ]
            ),
            "President": [
                "James Madison",
                "James Madison",
                "Millard Fillmore",
                "Woodrow Wilson",
                "George Bush",
            ],
            "Party": [
                "Democratic-Republican",
                "Democratic-Republican",
                "Whig",
                "Democratic",
                "Republican",
            ],
            "Transcript": [
                "Fellow citizens, the state of our union is strong.",
                "James Madison declares war on the British empire 1812.",
                "The republic endures. Millard Fillmore addresses Congress.",
                "Woodrow Wilson asks for a declaration of war in 1917.",
                "George Bush addresses the nation on foreign policy.",
            ],
        }
    )


@pytest.fixture
def tiny_features():
    """Feature array small enough for fast model tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 20)).astype(np.float32)


@pytest.fixture
def tiny_labels():
    """Binary labels matching tiny_features."""
    return np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])


@pytest.fixture
def known_predictions():
    """Fixed y_true and y_pred_prob for deterministic metric tests."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred_prob = np.array([0.1, 0.4, 0.8, 0.9, 0.7, 0.2, 0.6, 0.3])
    return y_true, y_pred_prob
