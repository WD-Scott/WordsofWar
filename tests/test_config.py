"""
Tests for shared configuration constants.
"""

from words_of_war import config


def test_color_constants_are_hex_strings():
    assert isinstance(config.NAVY, str)
    assert config.NAVY.startswith("#")
    assert isinstance(config.ORANGE, str)
    assert config.ORANGE.startswith("#")


def test_bert_constants():
    assert isinstance(config.BERT_MODEL_NAME, str)
    assert config.BERT_MAX_LEN > 0
    assert config.EMBEDDING_DIM == 768


def test_resampling_constants():
    assert 0 < config.SMOTE_RATIO < 1
    assert 0 < config.UNDERSAMPLE_RATIO < 1
    assert isinstance(config.RANDOM_STATE, int)


def test_training_constants():
    assert config.EPOCHS > 0
    assert config.BATCH_SIZE > 0
    assert config.LEARNING_RATE > 0
