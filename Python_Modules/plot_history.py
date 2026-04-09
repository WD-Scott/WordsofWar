"""
Training history visualization for model accuracy and loss curves.
"""

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_training_history(
    acc: Sequence[float],
    val_acc: Sequence[float],
    loss: Sequence[float],
    val_loss: Sequence[float],
    epochs_range: Sequence[int],
    train_color: str = "#232D4B",
    val_color: str = "#E57200",
) -> Figure:
    """
    Plot training and validation accuracy/loss curves side by side.

    Args:
        acc: Training accuracy per epoch.
        val_acc: Validation accuracy per epoch.
        loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        epochs_range: Epoch numbers for the x-axis.
        train_color: Hex color for training curves (default: UVA navy).
        val_color: Hex color for validation curves (default: UVA orange).

    Returns:
        The generated figure. Call ``plt.show()`` to display interactively.
    """
    with plt.style.context("fivethirtyeight"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy
        ax1.plot(epochs_range, acc, label="Training", color=train_color)
        ax1.plot(epochs_range, val_acc, label="Validation", color=val_color)
        ax1.set_xlabel("Epoch", weight="bold", size=15)
        ax1.set_ylabel("Accuracy", weight="bold", size=15)
        ax1.set_title("Training and Validation Accuracy", weight="bold", size=17)
        legend1 = ax1.legend(loc="lower right")
        plt.setp(legend1.texts, weight="bold", size=13)
        legend1.get_frame().set_edgecolor("black")

        # Loss
        ax2.plot(epochs_range, loss, label="Training", color=train_color)
        ax2.plot(epochs_range, val_loss, label="Validation", color=val_color)
        ax2.set_xlabel("Epoch", weight="bold", size=15)
        ax2.set_ylabel("Loss", weight="bold", size=15)
        ax2.set_title("Training and Validation Loss", weight="bold", size=17)
        legend2 = ax2.legend(loc="upper right")
        plt.setp(legend2.texts, weight="bold", size=13)
        legend2.get_frame().set_edgecolor("black")

        plt.tight_layout()
    return fig
