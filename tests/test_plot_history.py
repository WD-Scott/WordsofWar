"""
Tests for training history visualization.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from words_of_war.plot_history import plot_training_history  # noqa: E402


class TestPlotTrainingHistory:
    def _make_history(self):
        epochs = range(1, 6)
        acc = [0.5, 0.6, 0.7, 0.8, 0.85]
        val_acc = [0.45, 0.55, 0.65, 0.75, 0.80]
        loss = [0.9, 0.7, 0.5, 0.3, 0.2]
        val_loss = [0.95, 0.75, 0.55, 0.35, 0.25]
        return acc, val_acc, loss, val_loss, epochs

    def test_returns_figure(self):
        acc, val_acc, loss, val_loss, epochs = self._make_history()
        fig = plot_training_history(acc, val_acc, loss, val_loss, epochs)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_two_subplots(self):
        acc, val_acc, loss, val_loss, epochs = self._make_history()
        fig = plot_training_history(acc, val_acc, loss, val_loss, epochs)
        assert len(fig.axes) == 2
        plt.close(fig)
