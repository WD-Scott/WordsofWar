"""
Shared constants for the WordsofWar project.

Centralizes colors, BERT configuration, resampling parameters, and
training defaults used across notebooks and modules.
"""

# -- Visualization colors (UVA branding) ------------------------------------
NAVY: str = "#232D4B"
ORANGE: str = "#E57200"

# -- BERT configuration -----------------------------------------------------
BERT_MODEL_NAME: str = "bert-base-uncased"
BERT_MAX_LEN: int = 256
EMBEDDING_DIM: int = 768

# -- Resampling (class imbalance) -------------------------------------------
SMOTE_RATIO: float = 0.6
UNDERSAMPLE_RATIO: float = 0.8
RANDOM_STATE: int = 28

# -- Training defaults -------------------------------------------------------
EPOCHS: int = 10
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
