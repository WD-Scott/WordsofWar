"""
Neural network architectures for presidential speech classification.

Provides factory functions for MLP, RNN-LSTM, and LSTM-Attention models,
plus the custom :class:`AttentionLayer` used by the attention variant.
All models are returned **uncompiled** so callers can choose optimizers.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LSTM,
    Layer,
    Reshape,
)
from tensorflow.keras.models import Sequential


class AttentionLayer(Layer):
    """
    Additive (Bahdanau-style) attention over temporal steps.

    Learns a weight vector ``W`` and bias ``b``, computes alignment
    scores via ``softmax(xW + b)``, and returns the weighted sum over
    the sequence dimension.

    Input shape:
        ``(batch, timesteps, features)``

    Output shape:
        ``(batch, features)``
    """

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        et = tf.matmul(x, self.W) + self.b
        at = tf.nn.softmax(et, axis=1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, int]:
        return (input_shape[0], input_shape[-1])


def build_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...] = (128, 64),
    dropout: float = 0.1,
    l2_reg: float = 0.01,
) -> Sequential:
    """
    Build a multi-layer perceptron for binary classification.

    Args:
        input_dim:
            Number of input features (e.g. 768 for BERT embeddings).
        hidden_dims:
            Sizes of hidden dense layers.
        dropout:
            Dropout rate after each hidden layer.
        l2_reg:
            L2 regularization strength.

    Returns:
        An uncompiled ``Sequential`` model.
    """
    layers = [Input(shape=(input_dim,))]
    for dim in hidden_dims:
        layers.append(
            Dense(
                dim,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            )
        )
        layers.append(Dropout(dropout))
    layers.append(
        Dense(
            1,
            activation="sigmoid",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )
    )
    return Sequential(layers)


def build_lstm(
    input_dim: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout: float = 0.1,
    l2_reg: float = 0.01,
) -> Sequential:
    """
    Build an RNN-LSTM model for binary classification.

    Reshapes flat input to ``(1, input_dim)`` to create a single-step
    sequence, then applies an LSTM followed by a dense layer.

    Args:
        input_dim:
            Number of input features.
        lstm_units:
            Number of LSTM hidden units.
        dense_units:
            Size of the post-LSTM dense layer.
        dropout:
            Dropout rate.
        l2_reg:
            L2 regularization strength.

    Returns:
        An uncompiled ``Sequential`` model.
    """
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Reshape((1, input_dim)),
            LSTM(
                lstm_units,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                dropout=dropout,
                recurrent_dropout=dropout,
            ),
            Dense(
                dense_units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            Dropout(dropout),
            Dense(
                1,
                activation="sigmoid",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
        ]
    )


def build_lstm_attention(
    input_dim: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout: float = 0.1,
    l2_reg: float = 0.01,
) -> Sequential:
    """
    Build an LSTM model with an additive attention layer.

    Same structure as :func:`build_lstm` but the LSTM returns full
    sequences, which are then weighted by :class:`AttentionLayer`.

    Args:
        input_dim:
            Number of input features.
        lstm_units:
            Number of LSTM hidden units.
        dense_units:
            Size of the post-attention dense layer.
        dropout:
            Dropout rate.
        l2_reg:
            L2 regularization strength.

    Returns:
        An uncompiled ``Sequential`` model.
    """
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Reshape((1, input_dim)),
            LSTM(
                lstm_units,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True,
            ),
            AttentionLayer(),
            Dense(
                dense_units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            Dropout(dropout),
            Dense(
                1,
                activation="sigmoid",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
        ]
    )
