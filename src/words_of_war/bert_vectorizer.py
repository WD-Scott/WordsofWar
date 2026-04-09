"""
BERT sequence vectorizer for converting text into fixed-length embeddings.

Wraps HuggingFace's ``bert-base-uncased`` (by default) to produce a single
768-dimensional vector per input sentence, suitable for downstream classification.
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel


class BertSequenceVectorizer:
    """
    Vectorize text sequences using a pre-trained BERT model.

    Tokenizes input text, passes it through BERT, and returns the ``[CLS]``
    token embedding as a fixed-length representation of the sequence.

    Args:
        model_name: HuggingFace model identifier (default: ``'bert-base-uncased'``).
        max_len: Maximum token length for input sequences (default: 256).
            Longer sequences are truncated; shorter ones are padded.

    Attributes:
        device: ``'cuda'`` if a GPU is available, otherwise ``'cpu'``.
        tokenizer: Tokenizer instance for the specified model.
        bert_model: Pre-trained BERT encoder.
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_len: int = 256) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.max_len = max_len

        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert_model = BertModel.from_pretrained(self.model_name)
        except OSError as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_name}'. Ensure you have "
                f"internet access or the model is cached locally."
            ) from e

        self.bert_model = self.bert_model.to(self.device)  # type: ignore[arg-type]

    def vectorize(self, sentence: str) -> np.ndarray:
        """
        Convert a sentence into a BERT ``[CLS]`` embedding.

        Args:
            sentence: Input text to vectorize.

        Returns:
            1-D array of shape ``(768,)`` (for ``bert-base-uncased``).
        """
        inputs = self.tokenizer(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        inputs_tensor = inputs["input_ids"].to(self.device)
        masks_tensor = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            bert_out = self.bert_model(inputs_tensor, attention_mask=masks_tensor)

        seq_out = bert_out["last_hidden_state"]

        # .cpu() is a no-op when already on CPU
        return np.asarray(seq_out[0][0].cpu().detach().numpy())
