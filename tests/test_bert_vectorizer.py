"""
Tests for BertSequenceVectorizer with mocked HuggingFace models.

All tests mock the pretrained model downloads to keep tests fast and offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class TestBertSequenceVectorizerInit:
    @patch("words_of_war.bert_vectorizer.BertModel")
    @patch("words_of_war.bert_vectorizer.BertTokenizer")
    def test_sets_attributes(self, mock_tokenizer_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

        from words_of_war.bert_vectorizer import BertSequenceVectorizer
        vec = BertSequenceVectorizer(model_name="test-model", max_len=128)

        assert vec.model_name == "test-model"
        assert vec.max_len == 128
        assert vec.device in ("cpu", "cuda")

    @patch("words_of_war.bert_vectorizer.BertModel")
    @patch("words_of_war.bert_vectorizer.BertTokenizer")
    def test_raises_on_bad_model(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer_cls.from_pretrained.side_effect = OSError("not found")

        from words_of_war.bert_vectorizer import BertSequenceVectorizer
        with pytest.raises(RuntimeError, match="Failed to load model"):
            BertSequenceVectorizer(model_name="nonexistent-model")


class TestBertSequenceVectorizerVectorize:
    @patch("words_of_war.bert_vectorizer.BertModel")
    @patch("words_of_war.bert_vectorizer.BertTokenizer")
    def test_returns_correct_shape(self, mock_tokenizer_cls, mock_model_cls):
        hidden_size = 768

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, key: (
            torch.randn(1, 10, hidden_size)
            if key == "last_hidden_state" else None
        )
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.return_value = mock_output
        mock_model_cls.from_pretrained.return_value = mock_model

        from words_of_war.bert_vectorizer import BertSequenceVectorizer
        vec = BertSequenceVectorizer()
        result = vec.vectorize("Hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (hidden_size,)
