import torch
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel


class BertSequenceVectorizer:
    """
    This class represents a BERT-based sequence vectorizer, which converts input text
    into vector representations using a pre-trained BERT model.

    Attributes:
    - device (str): The device used for computation (CPU or CUDA).
    - model_name (str): The name of the pre-trained BERT model to use.
    - tokenizer (BertTokenizer): The tokenizer for tokenizing input text.
    - bert_model (BertModel): The pre-trained BERT model for generating vector representations.
    - max_len (int): The maximum length of input sequences.
    """

    def __init__(self):
        """
        Initializes the BertSequenceVectorizer object.

        This method sets up the necessary components for vectorizing input text using BERT.
        It initializes the device, loads the pre-trained BERT model and tokenizer, and
        specifies the maximum input sequence length.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        """
        Vectorizes the input text using BERT.

        This method takes a string input (a sentence or text snippet) and converts it into
        a vector representation using a pre-trained BERT model.

        Parameters:
        - sentence (str): The input text to vectorize.

        Returns:
        - np.array: A numpy array representing the vectorized form of the input text.
        """
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()