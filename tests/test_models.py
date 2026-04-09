"""
Tests for neural network model architectures.

These tests verify model structure and output shapes using tiny inputs.
They run on CPU without GPU or large data.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from words_of_war.models import (  # noqa: E402
    AttentionLayer,
    build_lstm,
    build_lstm_attention,
    build_mlp,
)


INPUT_DIM = 20
BATCH_SIZE = 4


class TestBuildMLP:
    def test_output_shape(self):
        model = build_mlp(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (BATCH_SIZE, 1)

    def test_output_range(self):
        model = build_mlp(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_custom_hidden_dims(self):
        model = build_mlp(input_dim=INPUT_DIM, hidden_dims=(32,))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (BATCH_SIZE, 1)


class TestBuildLSTM:
    def test_output_shape(self):
        model = build_lstm(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (BATCH_SIZE, 1)

    def test_output_range(self):
        model = build_lstm(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestBuildLSTMAttention:
    def test_output_shape(self):
        model = build_lstm_attention(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (BATCH_SIZE, 1)

    def test_output_range(self):
        model = build_lstm_attention(input_dim=INPUT_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.rand(BATCH_SIZE, INPUT_DIM).astype(np.float32)
        preds = model.predict(X, verbose=0)
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestAttentionLayer:
    def test_output_shape(self):
        layer = AttentionLayer()
        x = tf.random.normal((BATCH_SIZE, 1, INPUT_DIM))
        out = layer(x)
        assert out.shape == (BATCH_SIZE, INPUT_DIM)
