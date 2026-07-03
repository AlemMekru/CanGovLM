from __future__ import annotations

from unittest import TestCase

import numpy as np

from cangovlm.model import PositionalEmbedding, PositionalEmbeddingError, TransformerConfig


class PositionalEmbeddingTests(TestCase):
    def test_initializes_embedding_matrix_from_transformer_config(self) -> None:
        config = _config()
        embedding = PositionalEmbedding(config, random_seed=42)

        self.assertEqual(embedding.embedding_matrix.shape, (config.context_length, 6))
        self.assertEqual(embedding.context_length, config.context_length)
        self.assertEqual(embedding.embedding_dim, 6)

    def test_forward_accepts_sequence_length(self) -> None:
        embedding = _embedding_with_known_weights()

        output = embedding.forward(3)

        self.assertEqual(output.shape, (3, 4))
        np.testing.assert_array_equal(output, embedding.embedding_matrix[:3])

    def test_forward_accepts_numpy_integer_sequence_length(self) -> None:
        embedding = _embedding_with_known_weights()

        output = embedding.forward(np.int64(3))

        self.assertEqual(output.shape, (3, 4))
        np.testing.assert_array_equal(output, embedding.embedding_matrix[:3])

    def test_forward_accepts_single_sequence_token_embeddings(self) -> None:
        embedding = _embedding_with_known_weights()
        token_embeddings = np.zeros((3, 4), dtype=np.float64)

        output = embedding.forward(token_embeddings)

        self.assertEqual(output.shape, (3, 4))
        np.testing.assert_array_equal(output, embedding.embedding_matrix[:3])

    def test_forward_accepts_batched_token_embeddings(self) -> None:
        embedding = _embedding_with_known_weights()
        token_embeddings = np.zeros((2, 3, 4), dtype=np.float64)

        output = embedding.forward(token_embeddings)

        self.assertEqual(output.shape, (2, 3, 4))
        np.testing.assert_array_equal(output[0], embedding.embedding_matrix[:3])
        np.testing.assert_array_equal(output[1], embedding.embedding_matrix[:3])

    def test_forward_rejects_sequence_length_above_context_length(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(PositionalEmbeddingError) as context:
            embedding.forward(6)

        self.assertIn("cannot exceed context_length", str(context.exception))

    def test_forward_rejects_negative_sequence_length(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(PositionalEmbeddingError):
            embedding.forward(-1)

    def test_forward_rejects_unsupported_input_dimensions(self) -> None:
        embedding = _embedding_with_known_weights()

        for value in (
            np.array(1.0, dtype=np.float64),
            np.zeros((1, 2, 3, 4), dtype=np.float64),
        ):
            with self.subTest(shape=value.shape):
                with self.assertRaises(PositionalEmbeddingError):
                    embedding.forward(value)

    def test_forward_rejects_wrong_embedding_dimension(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(PositionalEmbeddingError) as context:
            embedding.forward(np.zeros((2, 3), dtype=np.float64))

        self.assertIn("embedding dimension", str(context.exception))

    def test_random_initialization_is_deterministic_with_seed(self) -> None:
        config = _config(weight_init_std=0.05)

        first = PositionalEmbedding(config, random_seed=123)
        second = PositionalEmbedding(config, random_seed=123)
        different = PositionalEmbedding(config, random_seed=456)

        np.testing.assert_array_equal(first.embedding_matrix, second.embedding_matrix)
        self.assertFalse(np.array_equal(first.embedding_matrix, different.embedding_matrix))

    def test_random_initialization_uses_configured_standard_deviation(self) -> None:
        config = TransformerConfig(
            vocabulary_size=16,
            context_length=2048,
            embedding_dim=16,
            feed_forward_dim=64,
            num_layers=1,
            num_attention_heads=4,
            dropout_probability=0.0,
            weight_init_std=0.05,
        )

        embedding = PositionalEmbedding(config, random_seed=1)

        self.assertAlmostEqual(float(np.std(embedding.embedding_matrix)), 0.05, places=2)

    def test_position_lookup_returns_exact_rows_from_matrix(self) -> None:
        embedding = _embedding_with_known_weights()

        output = embedding.forward(4)

        np.testing.assert_array_equal(output[0], embedding.embedding_matrix[0])
        np.testing.assert_array_equal(output[1], embedding.embedding_matrix[1])
        np.testing.assert_array_equal(output[3], embedding.embedding_matrix[3])

    def test_combine_with_token_embeddings_adds_positions_to_single_sequence(self) -> None:
        embedding = _embedding_with_known_weights()
        token_embeddings = np.ones((3, 4), dtype=np.float64)

        output = embedding.combine_with_token_embeddings(token_embeddings)

        np.testing.assert_array_equal(output, token_embeddings + embedding.embedding_matrix[:3])

    def test_combine_with_token_embeddings_adds_positions_to_batch(self) -> None:
        embedding = _embedding_with_known_weights()
        token_embeddings = np.ones((2, 3, 4), dtype=np.float64)

        output = embedding.combine_with_token_embeddings(token_embeddings)

        expected = token_embeddings + np.array(
            [embedding.embedding_matrix[:3], embedding.embedding_matrix[:3]]
        )
        np.testing.assert_array_equal(output, expected)

    def test_combine_with_token_embeddings_rejects_non_float_embeddings(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(PositionalEmbeddingError):
            embedding.combine_with_token_embeddings(np.ones((3, 4), dtype=np.int64))

    def test_embedding_matrix_is_copied_when_supplied(self) -> None:
        weights = np.arange(20, dtype=np.float64).reshape(5, 4)
        embedding = PositionalEmbedding(_known_weights_config(), embedding_matrix=weights)

        weights[0, 0] = -99.0

        self.assertEqual(embedding.embedding_matrix[0, 0], 0.0)

    def test_rejects_embedding_matrix_with_wrong_shape(self) -> None:
        with self.assertRaises(PositionalEmbeddingError):
            PositionalEmbedding(
                _known_weights_config(),
                embedding_matrix=np.zeros((4, 4), dtype=np.float64),
            )

    def test_rejects_embedding_matrix_with_non_finite_values(self) -> None:
        weights = np.arange(20, dtype=np.float64).reshape(5, 4)
        weights[0, 0] = np.inf

        with self.assertRaises(PositionalEmbeddingError):
            PositionalEmbedding(_known_weights_config(), embedding_matrix=weights)


def _config(weight_init_std: float = 0.02) -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=16,
        context_length=8,
        embedding_dim=6,
        feed_forward_dim=24,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=0.0,
        weight_init_std=weight_init_std,
    )


def _embedding_with_known_weights() -> PositionalEmbedding:
    weights = np.arange(20, dtype=np.float64).reshape(5, 4)
    return PositionalEmbedding(_known_weights_config(), embedding_matrix=weights)


def _known_weights_config() -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=16,
        context_length=5,
        embedding_dim=4,
        feed_forward_dim=16,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=0.0,
        weight_init_std=0.02,
    )
