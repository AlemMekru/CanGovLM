from __future__ import annotations

from unittest import TestCase

import numpy as np

from cangovlm.model import (
    MultiHeadSelfAttention,
    MultiHeadSelfAttentionError,
    TransformerConfig,
)


class MultiHeadSelfAttentionTests(TestCase):
    def test_initializes_projection_matrices_from_transformer_config(self) -> None:
        config = _config()
        attention = MultiHeadSelfAttention(config, random_seed=42)

        self.assertEqual(attention.query_projection.shape, (4, 4))
        self.assertEqual(attention.key_projection.shape, (4, 4))
        self.assertEqual(attention.value_projection.shape, (4, 4))
        self.assertEqual(attention.output_projection.shape, (4, 4))
        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(attention.head_dim, 2)

    def test_forward_preserves_single_sequence_shape(self) -> None:
        attention = _identity_attention()
        inputs = np.arange(12, dtype=np.float64).reshape(3, 4)

        output = attention.forward(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_forward_preserves_batched_shape(self) -> None:
        attention = _identity_attention()
        inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

        output = attention.forward(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_split_heads_and_merge_heads_round_trip(self) -> None:
        attention = _identity_attention()
        inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

        heads = attention.split_heads(inputs)
        merged = attention.merge_heads(heads)

        self.assertEqual(heads.shape, (2, 2, 3, 2))
        np.testing.assert_array_equal(merged, inputs)

    def test_causal_mask_is_lower_triangular(self) -> None:
        attention = _identity_attention()

        mask = attention.causal_mask(4)

        expected = np.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_causal_mask_prevents_attention_to_future_positions(self) -> None:
        attention = _identity_attention()
        inputs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        _, weights = attention.forward(inputs, return_attention_weights=True)

        self.assertEqual(weights.shape, (2, 3, 3))
        self.assertTrue(np.all(weights[:, 0, 1:] == 0.0))
        self.assertTrue(np.all(weights[:, 1, 2] == 0.0))
        self.assertGreater(weights[0, 2, 0], 0.0)
        self.assertGreater(weights[0, 2, 1], 0.0)
        self.assertGreater(weights[0, 2, 2], 0.0)

    def test_attention_weights_sum_to_one(self) -> None:
        attention = _identity_attention()
        inputs = np.arange(12, dtype=np.float64).reshape(3, 4)

        _, weights = attention.forward(inputs, return_attention_weights=True)

        np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones((2, 3)))

    def test_random_initialization_is_deterministic_with_seed(self) -> None:
        config = _config(weight_init_std=0.05)

        first = MultiHeadSelfAttention(config, random_seed=123)
        second = MultiHeadSelfAttention(config, random_seed=123)
        different = MultiHeadSelfAttention(config, random_seed=456)

        np.testing.assert_array_equal(first.query_projection, second.query_projection)
        np.testing.assert_array_equal(first.key_projection, second.key_projection)
        np.testing.assert_array_equal(first.value_projection, second.value_projection)
        np.testing.assert_array_equal(first.output_projection, second.output_projection)
        self.assertFalse(np.array_equal(first.query_projection, different.query_projection))

    def test_forward_pass_is_deterministic(self) -> None:
        config = _config()
        inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0
        attention = MultiHeadSelfAttention(config, random_seed=123)

        first = attention.forward(inputs)
        second = attention.forward(inputs)

        np.testing.assert_array_equal(first, second)

    def test_identity_attention_output_matches_manual_causal_attention(self) -> None:
        attention = _identity_attention()
        inputs = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 3.0]])

        output = attention.forward(inputs)

        query_heads = attention.split_heads(inputs)
        scores = query_heads @ np.swapaxes(query_heads, -1, -2)
        scores = scores / np.sqrt(attention.head_dim)
        scores = np.where(attention.causal_mask(2), scores, -np.inf)
        weights = _softmax(scores, axis=-1)
        expected = attention.merge_heads(weights @ query_heads)[0]
        np.testing.assert_allclose(output, expected)

    def test_rejects_invalid_input_dimensions(self) -> None:
        attention = _identity_attention()

        for inputs in (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.zeros((1, 2, 3, 4), dtype=np.float64),
        ):
            with self.subTest(shape=inputs.shape):
                with self.assertRaises(MultiHeadSelfAttentionError):
                    attention.forward(inputs)

    def test_rejects_wrong_embedding_dimension(self) -> None:
        attention = _identity_attention()

        with self.assertRaises(MultiHeadSelfAttentionError) as context:
            attention.forward(np.zeros((2, 3), dtype=np.float64))

        self.assertIn("last input dimension", str(context.exception))

    def test_rejects_non_float_inputs(self) -> None:
        attention = _identity_attention()

        with self.assertRaises(MultiHeadSelfAttentionError):
            attention.forward(np.zeros((2, 4), dtype=np.int64))

    def test_rejects_sequence_length_above_context_length(self) -> None:
        attention = _identity_attention()

        with self.assertRaises(MultiHeadSelfAttentionError) as context:
            attention.forward(np.zeros((6, 4), dtype=np.float64))

        self.assertIn("context_length", str(context.exception))

    def test_rejects_invalid_projection_shape(self) -> None:
        with self.assertRaises(MultiHeadSelfAttentionError):
            MultiHeadSelfAttention(_config(), query_projection=np.zeros((3, 4)))

    def test_invalid_head_count_is_rejected_by_transformer_config(self) -> None:
        with self.assertRaises(ValueError):
            TransformerConfig(
                vocabulary_size=16,
                context_length=5,
                embedding_dim=5,
                feed_forward_dim=20,
                num_layers=1,
                num_attention_heads=2,
                dropout_probability=0.0,
                weight_init_std=0.02,
            )

    def test_training_attention_dropout_interface_is_explicitly_unimplemented(self) -> None:
        attention = MultiHeadSelfAttention(_config(dropout_probability=0.1), random_seed=1)
        inputs = np.zeros((2, 4), dtype=np.float64)

        with self.assertRaises(MultiHeadSelfAttentionError) as context:
            attention.forward(inputs, training=True)

        self.assertIn("dropout", str(context.exception))

    def test_json_state_round_trips(self) -> None:
        config = _config(dropout_probability=0.0)
        attention = MultiHeadSelfAttention(config, random_seed=123)

        loaded = MultiHeadSelfAttention.from_json(config, attention.to_json())

        np.testing.assert_array_equal(loaded.query_projection, attention.query_projection)
        np.testing.assert_array_equal(loaded.key_projection, attention.key_projection)
        np.testing.assert_array_equal(loaded.value_projection, attention.value_projection)
        np.testing.assert_array_equal(loaded.output_projection, attention.output_projection)


def _config(
    *,
    weight_init_std: float = 0.02,
    dropout_probability: float = 0.0,
) -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=16,
        context_length=5,
        embedding_dim=4,
        feed_forward_dim=16,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=dropout_probability,
        weight_init_std=weight_init_std,
    )


def _identity_attention() -> MultiHeadSelfAttention:
    identity = np.eye(4, dtype=np.float64)
    return MultiHeadSelfAttention(
        _config(),
        query_projection=identity,
        key_projection=identity,
        value_projection=identity,
        output_projection=identity,
    )


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
