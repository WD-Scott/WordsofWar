"""
Model evaluation utilities for binary classification.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a binary classifier and return key metrics.

    Args:
        model:
            A fitted model with a ``.predict()`` method that returns
            probabilities for the positive class.
        X_test:
            Test feature array.
        y_test:
            True binary labels.
        threshold:
            Decision threshold for converting probabilities to class
            labels.

    Returns:
        Dictionary with keys ``auc_roc``, ``f1``, ``precision``,
        ``recall``, ``confusion_matrix``, ``y_pred_prob``, and
        ``y_pred_class``.
    """
    y_pred_prob = model.predict(X_test)

    # Flatten in case model returns shape (n, 1)
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    y_pred_class = (y_pred_prob > threshold).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_test, y_pred_prob)),
        "f1": float(f1_score(y_test, y_pred_class)),
        "precision": float(precision_score(y_test, y_pred_class)),
        "recall": float(recall_score(y_test, y_pred_class)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_class),
        "y_pred_prob": y_pred_prob,
        "y_pred_class": y_pred_class,
    }
