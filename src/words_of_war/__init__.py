"""
WordsofWar: Deep learning NLP analysis of US presidential rhetoric.

This package provides reusable utilities for the WordsofWar research
pipeline — data preparation, text processing, model architectures,
evaluation, visualization, and interpretability.
"""

from words_of_war.bert_vectorizer import BertSequenceVectorizer
from words_of_war.config import (
    BATCH_SIZE,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    EMBEDDING_DIM,
    EPOCHS,
    LEARNING_RATE,
    NAVY,
    ORANGE,
    RANDOM_STATE,
    SMOTE_RATIO,
    UNDERSAMPLE_RATIO,
)
from words_of_war.data_utils import (
    build_resampling_pipeline,
    create_war_dates_df,
    export_splits,
    label_wars,
    load_split_data,
)
from words_of_war.evaluation import evaluate_binary_model
from words_of_war.models import (
    AttentionLayer,
    build_lstm,
    build_lstm_attention,
    build_mlp,
)
from words_of_war.plot_history import plot_training_history
from words_of_war.text_processing import clean_transcript

__all__ = [
    # config
    "NAVY",
    "ORANGE",
    "BERT_MODEL_NAME",
    "BERT_MAX_LEN",
    "EMBEDDING_DIM",
    "SMOTE_RATIO",
    "UNDERSAMPLE_RATIO",
    "RANDOM_STATE",
    "EPOCHS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    # data_utils
    "create_war_dates_df",
    "label_wars",
    "build_resampling_pipeline",
    "load_split_data",
    "export_splits",
    # text_processing
    "clean_transcript",
    # models
    "AttentionLayer",
    "build_mlp",
    "build_lstm",
    "build_lstm_attention",
    # evaluation
    "evaluate_binary_model",
    # bert_vectorizer
    "BertSequenceVectorizer",
    # plot_history
    "plot_training_history",
]
