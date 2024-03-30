import torch
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel


class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 256

    def vectorize(self, sentence: str) -> np.array:
        # Tokenize input
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        inputs_tensor = inputs['input_ids'].to(self.device)
        masks_tensor = inputs['attention_mask'].to(self.device)

        # Get BERT output
        with torch.no_grad():
            bert_out = self.bert_model(inputs_tensor, attention_mask=masks_tensor)

        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()