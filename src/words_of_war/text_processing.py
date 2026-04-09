"""
Text normalization utilities for presidential speech transcripts.
"""

import re

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

# Ensure the punkt tokenizer data is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def clean_transcript(row: pd.Series) -> str:
    """
    Clean a transcript by removing the president's name, numbers, and punctuation.

    Designed to be applied row-wise to a DataFrame containing ``'President'``
    and ``'Transcript'`` columns.

    Args:
        row:
            A row from the DataFrame.  Must contain ``'President'``
            (str) and ``'Transcript'`` (str) fields.

    Returns:
        The cleaned transcript with the president's name, numbers, and
        punctuation removed, and all text lowercased.
    """
    president = row["President"].lower()
    transcript = str(row["Transcript"])

    # Remove floating-point numbers and integers
    transcript = re.sub(r"\b\d+(?:\.\d+)?\s+", "", transcript)

    # Convert transcript to lowercase
    transcript = transcript.lower()

    # Remove president's name
    transcript = transcript.replace(president, "").strip()

    # Remove punctuation using regular expressions
    transcript = re.sub(r"[^\w\s]", "", transcript)

    # Tokenize transcript
    tokens = word_tokenize(transcript)

    # Join tokens back into a string
    return " ".join(tokens)
