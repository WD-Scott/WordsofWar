"""
Tests for text processing utilities.
"""

import pandas as pd

from words_of_war.text_processing import clean_transcript


def test_removes_president_name():
    row = pd.Series(
        {"President": "John Adams", "Transcript": "John Adams spoke to Congress."}
    )
    result = clean_transcript(row)
    assert "john adams" not in result
    assert "john" not in result.split()


def test_removes_numbers():
    row = pd.Series(
        {"President": "Test President", "Transcript": "In 1812 the war began with 50 ships."}
    )
    result = clean_transcript(row)
    assert "1812" not in result
    assert "50" not in result


def test_removes_punctuation():
    row = pd.Series(
        {"President": "Test President", "Transcript": "Hello, world! This is a test."}
    )
    result = clean_transcript(row)
    assert "," not in result
    assert "!" not in result
    assert "." not in result


def test_lowercases_text():
    row = pd.Series(
        {"President": "Test President", "Transcript": "The NATION is STRONG."}
    )
    result = clean_transcript(row)
    assert result == result.lower()


def test_returns_string():
    row = pd.Series(
        {"President": "Test", "Transcript": "Simple text here."}
    )
    assert isinstance(clean_transcript(row), str)


def test_handles_nan_transcript():
    row = pd.Series(
        {"President": "Test", "Transcript": float("nan")}
    )
    result = clean_transcript(row)
    assert isinstance(result, str)
