"""
Model interpretability utilities: attention analysis, LIME, and SHAP.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import ttest_ind

from words_of_war.config import NAVY, ORANGE


def extract_attention_weights(
    model: Any,
    X_data: np.ndarray,
    attention_layer_index: int = 4,
) -> np.ndarray:
    """
    Extract attention weights from an LSTM-Attention model.

    Builds a sub-model that outputs the attention layer's activations,
    then predicts on the provided data.

    Args:
        model:
            A trained Keras model containing an attention layer.
        X_data:
            Input feature array.
        attention_layer_index:
            Index of the attention layer in the model's layer list.

    Returns:
        Array of attention weight outputs with shape
        ``(n_samples, ...)``.
    """
    from tensorflow.keras.models import Model

    attention_model = Model(
        inputs=model.input,
        outputs=model.layers[attention_layer_index].output,
    )
    return np.asarray(attention_model.predict(X_data))


def analyze_attention_by_class(
    attention_weights: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare mean attention weights between classes using a t-test.

    Args:
        attention_weights:
            Attention layer outputs, shape ``(n_samples, features)``.
        y:
            Binary labels aligned with *attention_weights*.

    Returns:
        Dictionary with keys ``mean_class_0``, ``mean_class_1``,
        ``t_statistic``, and ``p_value``.
    """
    mean_wt_0 = np.mean(attention_weights[y == 0], axis=1)
    mean_wt_1 = np.mean(attention_weights[y == 1], axis=1)
    t_stat, p_value = ttest_ind(mean_wt_0, mean_wt_1)

    return {
        "mean_class_0": mean_wt_0,
        "mean_class_1": mean_wt_1,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def plot_attention_distribution(
    mean_wt_class_0: np.ndarray,
    mean_wt_class_1: np.ndarray,
    color_0: str = NAVY,
    color_1: str = ORANGE,
) -> Figure:
    """
    Plot overlaid histograms of mean attention weights by class.

    Args:
        mean_wt_class_0:
            Mean attention weights for class 0 (peacetime).
        mean_wt_class_1:
            Mean attention weights for class 1 (war).
        color_0:
            Color for class 0.
        color_1:
            Color for class 1.

    Returns:
        The generated figure.
    """
    with plt.style.context("fivethirtyeight"):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(mean_wt_class_0, color=color_0, label="0", kde=True, ax=ax)
        sns.histplot(mean_wt_class_1, color=color_1, label="1", kde=True, ax=ax)
        ax.set_title("Distribution of Mean Attention Weights")
        ax.set_xlabel("Mean Attention Weight")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    return fig


def create_lime_explainer(
    X_train: np.ndarray,
    feature_names: list,
    class_names: Optional[list] = None,
) -> Any:
    """
    Create a LIME tabular explainer for the training data.

    Args:
        X_train:
            Training feature array.
        feature_names:
            List of feature column names.
        class_names:
            Human-readable class labels (default: ``['Non-War', 'War']``).

    Returns:
        A ``lime.lime_tabular.LimeTabularExplainer`` instance.
    """
    import lime.lime_tabular

    if class_names is None:
        class_names = ["Non-War", "War"]

    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )


def create_shap_explainer(model: Any, X_train: np.ndarray) -> Tuple[Any, np.ndarray]:
    """
    Create a SHAP explainer and compute SHAP values.

    Args:
        model:
            A trained model compatible with ``shap.Explainer``.
        X_train:
            Training feature array used as background data.

    Returns:
        Tuple of ``(explainer, shap_values)``.
    """
    import shap

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    return explainer, shap_values
