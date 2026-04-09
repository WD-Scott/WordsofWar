"""
Tests for data loading, labeling, and resampling utilities.
"""

import numpy as np
import pandas as pd

from words_of_war.data_utils import (
    build_resampling_pipeline,
    create_war_dates_df,
    export_splits,
    label_wars,
    load_split_data,
)


class TestCreateWarDates:
    def test_returns_dataframe(self):
        df = create_war_dates_df()
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = create_war_dates_df()
        assert set(df.columns) == {"War_Name", "Start_Date", "End_Date"}

    def test_dates_are_datetime(self):
        df = create_war_dates_df()
        assert pd.api.types.is_datetime64_any_dtype(df["Start_Date"])
        assert pd.api.types.is_datetime64_any_dtype(df["End_Date"])

    def test_has_known_wars(self):
        df = create_war_dates_df()
        names = df["War_Name"].tolist()
        assert "World War 1" in names
        assert "World War 2" in names
        assert "Korean War" in names


class TestLabelWars:
    def test_adds_war_column(self, sample_speech_df):
        war_dates = create_war_dates_df()
        result = label_wars(sample_speech_df, war_dates)
        assert "War" in result.columns

    def test_labels_are_binary(self, sample_speech_df):
        war_dates = create_war_dates_df()
        result = label_wars(sample_speech_df, war_dates)
        assert set(result["War"].unique()).issubset({0, 1})

    def test_does_not_modify_input(self, sample_speech_df):
        war_dates = create_war_dates_df()
        original_cols = set(sample_speech_df.columns)
        label_wars(sample_speech_df, war_dates)
        assert set(sample_speech_df.columns) == original_cols

    def test_pre_war_speech_labeled_1(self):
        """A speech 6 months before War of 1812 (June 1812) should be labeled 1."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["1812-01-01"]),
                "President": ["James Madison"],
            }
        )
        war_dates = create_war_dates_df()
        result = label_wars(df, war_dates)
        assert result["War"].iloc[0] == 1

    def test_peacetime_speech_labeled_0(self):
        """A speech in 1830 (no nearby wars) should be labeled 0."""
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["1830-06-01"]),
                "President": ["Andrew Jackson"],
            }
        )
        war_dates = create_war_dates_df()
        result = label_wars(df, war_dates)
        assert result["War"].iloc[0] == 0


class TestBuildResamplingPipeline:
    def test_returns_pipeline(self):
        pipeline = build_resampling_pipeline()
        assert hasattr(pipeline, "fit_resample")

    def test_resamples_imbalanced_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = np.array([0] * 90 + [1] * 10)
        pipeline = build_resampling_pipeline()
        X_res, y_res = pipeline.fit_resample(X, y)
        # After SMOTE + undersampling, class ratio should be closer to balanced
        counts = np.bincount(y_res)
        assert counts[1] > 10  # Minority class grew


class TestExportAndLoad:
    def test_round_trip(self, tmp_path):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((20, 5))
        X_val = rng.standard_normal((5, 5))
        X_test = rng.standard_normal((5, 5))
        y_train = np.array([0] * 15 + [1] * 5)
        y_val = np.array([0, 0, 1, 0, 1])
        y_test = np.array([1, 0, 0, 1, 0])

        export_splits(X_train, X_val, X_test, y_train, y_val, y_test, str(tmp_path))
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_split_data(str(tmp_path))

        np.testing.assert_array_almost_equal(X_tr, X_train)
        np.testing.assert_array_almost_equal(X_v, X_val)
        np.testing.assert_array_almost_equal(X_te, X_test)
        np.testing.assert_array_equal(y_tr, y_train)
        np.testing.assert_array_equal(y_v, y_val)
        np.testing.assert_array_equal(y_te, y_test)
