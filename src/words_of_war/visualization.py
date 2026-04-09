"""
Visualization utilities for exploratory data analysis.

All plot functions return a ``matplotlib.figure.Figure`` so they can be
used both interactively (call ``plt.show()``) and programmatically.
"""

from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from words_of_war.config import NAVY, ORANGE


def _compact_thousands(x: float, _: object) -> str:
    if abs(x) >= 1000:
        return f"{int(x / 1000)}k"
    return str(int(x))


_THOUSANDS_FMT = FuncFormatter(_compact_thousands)


def plot_speech_length_distribution(
    df: pd.DataFrame,
    text_length_col: str = "Text Length",
    color: str = NAVY,
    edge_color: str = ORANGE,
) -> Figure:
    """
    Plot a histogram of overall speech lengths.

    Args:
        df:
            DataFrame with a numeric column for speech lengths.
        text_length_col:
            Name of the column containing word counts.
        color:
            Fill color for bars.
        edge_color:
            Edge color for bars.

    Returns:
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        df[text_length_col].dropna(),
        bins=20,
        color=color,
        edgecolor=edge_color,
        linewidth=1.25,
    )
    ax.set_xlabel("Speech Length (number of words)", weight="bold", size=12)
    ax.set_ylabel("Count", weight="bold", size=12)
    ax.set_title("Distribution of Speech Lengths", weight="bold", size=15)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_formatter(_THOUSANDS_FMT)
    sns.set_context("paper", font_scale=1.2)
    plt.tight_layout()
    return fig


def plot_class_distributions(
    df: pd.DataFrame,
    text_col: str = "Transcript",
    label_col: str = "War",
    color: str = NAVY,
    edge_color: str = ORANGE,
) -> Figure:
    """
    Plot side-by-side histograms of speech lengths by class.

    Args:
        df:
            DataFrame with text and label columns.
        text_col:
            Name of the column containing transcript text.
        label_col:
            Name of the binary label column.
        color:
            Fill color for bars.
        edge_color:
            Edge color for bars.

    Returns:
        The generated figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for ax, war_value in zip(axs, [1, 0]):
        lengths = df[df[label_col] == war_value][text_col].apply(len).dropna()
        ax.hist(lengths, bins=10, color=color, edgecolor=edge_color, linewidth=1.25)
        ax.set_xlabel("Speech Length (number of words)", weight="bold", size=12)
        ax.set_ylabel("Count", weight="bold", size=12)
        ax.set_title(
            f"Distribution of Speech Lengths (War = {war_value})",
            weight="bold",
            size=14,
        )
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_formatter(_THOUSANDS_FMT)

    plt.tight_layout()
    return fig


def plot_class_boxplots(
    df: pd.DataFrame,
    text_col: str = "Transcript",
    label_col: str = "War",
    box_color: str = NAVY,
    accent_color: str = ORANGE,
) -> Figure:
    """
    Plot side-by-side boxplots of speech lengths by class.

    Args:
        df:
            DataFrame with text and label columns.
        text_col:
            Name of the column containing transcript text.
        label_col:
            Name of the binary label column.
        box_color:
            Fill color for the box body.
        accent_color:
            Color for edges, median line, and whiskers.

    Returns:
        The generated figure.
    """
    sns.set_style("ticks")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for ax, war_value in zip(axs, [1, 0]):
        lengths = df[df[label_col] == war_value][text_col].apply(len)
        sns.boxplot(
            y=lengths,
            ax=ax,
            color=box_color,
            width=0.5,
            boxprops=dict(edgecolor=accent_color),
            medianprops=dict(color=accent_color),
            whiskerprops=dict(color=accent_color),
        )
        ax.set_xlabel("")
        ax.set_ylabel("Speech Length (number of words)", weight="bold", size=12)
        ax.set_title(
            f"Distribution of Speech Lengths (War = {war_value})",
            weight="bold",
            size=14,
        )
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_formatter(_THOUSANDS_FMT)

    plt.tight_layout()
    return fig


def plot_overall_top_words(
    df: pd.DataFrame,
    text_col: str = "Transcript",
    n_words: int = 25,
    bar_color: str = NAVY,
    accent_color: str = ORANGE,
) -> Figure:
    """
    Plot top word frequencies across the entire corpus.

    Args:
        df:
            DataFrame with a text column.
        text_col:
            Name of the column containing transcript text.
        n_words:
            Number of top words to display.
        bar_color:
            Fill color for bars.
        accent_color:
            Edge color for bars.

    Returns:
        The generated figure.
    """
    word_counts: Counter = Counter()
    for transcript in df[text_col]:
        if isinstance(transcript, str):
            word_counts.update(transcript.split())

    top = word_counts.most_common(n_words)
    words, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(words)), counts,
        color=bar_color, edgecolor=accent_color, linewidth=0.5,
    )
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel("Frequency", weight="bold", size=12)
    ax.set_ylabel("Words", weight="bold", size=12)
    ax.set_title(f"Top {n_words} Words in Transcript", weight="bold", size=15)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_formatter(_THOUSANDS_FMT)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_top_words(
    df: pd.DataFrame,
    text_col: str = "Transcript",
    label_col: str = "War",
    n_words: int = 25,
    bar_color: str = NAVY,
    accent_color: str = ORANGE,
    common_color: str = "green",
) -> Figure:
    """
    Plot top word frequencies for each class with shared words highlighted.

    Args:
        df:
            DataFrame with text and label columns.
        text_col:
            Name of the column containing transcript text.
        label_col:
            Name of the binary label column.
        n_words:
            Number of top words to display per class.
        bar_color:
            Fill color for bars.
        accent_color:
            Edge color for bars.
        common_color:
            Label color for words appearing in both classes.

    Returns:
        The generated figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    word_sets = []

    for ax, war_value in zip(axs, [1, 0]):
        subset = df[df[label_col] == war_value]
        word_counts: Counter = Counter()
        for transcript in subset[text_col]:
            if isinstance(transcript, str):
                word_counts.update(transcript.split())

        top = word_counts.most_common(n_words)
        words, counts = zip(*top)
        ax.barh(
            range(len(words)), counts,
            color=bar_color, edgecolor=accent_color, linewidth=0.5,
        )
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel("Frequency", weight="bold", size=12)
        ax.set_ylabel("Words", weight="bold", size=12)
        ax.set_title(
            f"Top {n_words} Words (War = {war_value})",
            weight="bold", size=14,
        )
        ax.tick_params(labelsize=11)
        ax.xaxis.set_major_formatter(_THOUSANDS_FMT)
        ax.invert_yaxis()
        word_sets.append(set(words))

    # Highlight common words
    common = word_sets[0].intersection(word_sets[1])
    for ax in axs:
        for label in ax.get_yticklabels():
            if label.get_text() in common:
                label.set_color(common_color)

    plt.tight_layout()
    return fig


def plot_class_balance(
    old_counts: pd.Series,
    new_counts: pd.Series,
    colors: Optional[list] = None,
) -> Figure:
    """
    Visualize class counts before and after resampling.

    Args:
        old_counts:
            Value counts before resampling.
        new_counts:
            Value counts after resampling.
        colors:
            Two-element list of bar colors.

    Returns:
        The generated figure.
    """
    if colors is None:
        colors = [NAVY, ORANGE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    with plt.style.context("fivethirtyeight"):
        old_counts.plot(kind="bar", color=colors, ax=ax1)
        ax1.set_title("Before SMOTE", weight="bold")
        ax1.set_xlabel("Class", weight="bold", size=12)
        ax1.set_ylabel("Count", weight="bold", size=12)
        ax1.tick_params(axis="x", rotation=0)

        new_counts.plot(kind="bar", color=colors, ax=ax2)
        ax2.set_title("After SMOTE", weight="bold")
        ax2.set_xlabel("Class", weight="bold", size=12)
        ax2.set_ylabel("Count", weight="bold", size=12)
        ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    return fig
