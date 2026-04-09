"""
Data loading, labeling, resampling, and export utilities.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from words_of_war.config import (
    RANDOM_STATE,
    SMOTE_RATIO,
    UNDERSAMPLE_RATIO,
)


def create_war_dates_df() -> pd.DataFrame:
    """
    Return a DataFrame of major US wars with start and end dates.

    Returns:
        DataFrame with columns ``War_Name``, ``Start_Date``, and
        ``End_Date`` (both as ``datetime64``).
    """
    df_wars = pd.DataFrame(
        {
            "War_Name": [
                "First Barbary Wars",
                "War of 1812",
                "Indian Wars",
                "Mexican-American War",
                "Spanish-American War",
                "Mexican Border Wars",
                "World War 1",
                "World War 2",
                "Korean War",
                "Vietnam War",
                "Persian Gulf War",
                "OEF",
                "OFS",
                "OES",
                "OIF",
                "OND",
                "OIR",
            ],
            "Start_Date": [
                "1801-05-01",
                "1812-06-18",
                "1817-01-01",
                "1846-04-25",
                "1898-04-21",
                "1916-05-09",
                "1917-04-06",
                "1941-12-07",
                "1950-06-25",
                "1964-08-05",
                "1990-08-02",
                "2001-10-07",
                "2015-01-01",
                "2021-10-01",
                "2003-03-17",
                "2010-11-01",
                "2014-10-15",
            ],
            "End_Date": [
                "1805-06-10",
                "1815-02-18",
                "1898-12-31",
                "1848-02-02",
                "1903-07-15",
                "1917-04-05",
                "1918-11-11",
                "1946-12-31",
                "1955-01-31",
                "1975-05-07",
                "1991-04-06",
                "2014-12-28",
                "2021-08-31",
                "2024-02-20",
                "2010-08-31",
                "2011-12-15",
                "2024-02-20",
            ],
        }
    )
    df_wars["Start_Date"] = pd.to_datetime(df_wars["Start_Date"])
    df_wars["End_Date"] = pd.to_datetime(df_wars["End_Date"])
    return df_wars


def label_wars(
    df: pd.DataFrame,
    war_dates_df: pd.DataFrame,
    lookback_years: int = 1,
) -> pd.DataFrame:
    """
    Add a binary ``War`` column indicating proximity to a war start date.

    A speech is labeled ``1`` if its date falls within *lookback_years*
    before any war's start date (inclusive of the start date itself).

    Args:
        df:
            DataFrame with a ``Date`` column (datetime).
        war_dates_df:
            DataFrame returned by :func:`create_war_dates_df`.
        lookback_years:
            Number of years before a war start date to label as ``1``.

    Returns:
        A copy of *df* with the ``War`` column added.
    """
    df = df.copy()
    df["War"] = 0

    for _, row in df.iterrows():
        current_date = row["Date"]
        for _, war_row in war_dates_df.iterrows():
            start_date = war_row["Start_Date"]
            window_start = start_date - pd.DateOffset(years=lookback_years)
            if window_start <= current_date <= start_date:
                df.at[row.name, "War"] = 1
                break

    return df


def build_resampling_pipeline(
    smote_ratio: float = SMOTE_RATIO,
    undersample_ratio: float = UNDERSAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Build an imbalanced-learn pipeline that applies SMOTE then undersampling.

    Args:
        smote_ratio:
            SMOTE ``sampling_strategy`` (minority/majority target ratio).
        undersample_ratio:
            ``RandomUnderSampler`` ``sampling_strategy``.
        random_state:
            Random seed for reproducibility.

    Returns:
        An imblearn ``Pipeline`` ready for ``.fit_resample()``.
    """
    over = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
    under = RandomUnderSampler(
        sampling_strategy=undersample_ratio, random_state=random_state
    )
    return Pipeline(steps=[("smote", over), ("undersample", under)])


def load_split_data(
    data_dir: str = "data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-split feature and label CSVs into numpy arrays.

    Expects ``X_train.csv``, ``X_val.csv``, ``X_test.csv``,
    ``y_train.csv``, ``y_val.csv``, ``y_test.csv`` in *data_dir*.

    Args:
        data_dir:
            Path to the directory containing the CSV files.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)`` as numpy
        arrays.
    """
    d = Path(data_dir)
    X_train = pd.read_csv(d / "X_train.csv").values
    X_val = pd.read_csv(d / "X_val.csv").values
    X_test = pd.read_csv(d / "X_test.csv").values
    y_train = pd.read_csv(d / "y_train.csv")["War"].values
    y_val = pd.read_csv(d / "y_val.csv")["War"].values
    y_test = pd.read_csv(d / "y_test.csv")["War"].values
    return X_train, X_val, X_test, y_train, y_val, y_test


def export_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    data_dir: str = "data",
) -> None:
    """
    Save train/val/test arrays as CSV files.

    Args:
        X_train, X_val, X_test:
            Feature arrays.
        y_train, y_val, y_test:
            Label arrays.
        data_dir:
            Output directory path.
    """
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)

    n_features = X_train.shape[1]
    col_names = [f"feature_{i}" for i in range(n_features)]

    for name, arr in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        pd.DataFrame(arr, columns=col_names).to_csv(d / f"{name}.csv", index=False)

    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        pd.DataFrame(arr, columns=["War"]).to_csv(d / f"{name}.csv", index=False)
