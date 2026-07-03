from __future__ import annotations

from unittest import TestCase

import numpy as np

from cangovlm.model import TokenEmbedding, TokenEmbeddingError, TransformerConfig


class TokenEmbeddingTests(TestCase):
    def test_initializes_embedding_matrix_from_transformer_config(self) -> None:
        config = _config()
        embedding = TokenEmbedding(config, random_seed=42)

        self.assertEqual(embedding.embedding_matrix.shape, (config.vocabulary_size, 8))
        self.assertEqual(embedding.vocabulary_size, config.vocabulary_size)
        self.assertEqual(embedding.embedding_dim, 8)

    def test_forward_accepts_1d_token_ids(self) -> None:
        embedding = _embedding_with_known_weights()
        token_ids = np.array([0, 2, 4], dtype=np.int64)

        output = embedding.forward(token_ids)

        self.assertEqual(output.shape, (3, 4))
        np.testing.assert_array_equal(output[0], embedding.embedding_matrix[0])
        np.testing.assert_array_equal(output[1], embedding.embedding_matrix[2])
        np.testing.assert_array_equal(output[2], embedding.embedding_matrix[4])

    def test_forward_accepts_2d_token_ids(self) -> None:
        embedding = _embedding_with_known_weights()
        token_ids = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

        output = embedding.forward(token_ids)

        self.assertEqual(output.shape, (2, 3, 4))
        np.testing.assert_array_equal(output[0, 0], embedding.embedding_matrix[0])
        np.testing.assert_array_equal(output[0, 2], embedding.embedding_matrix[2])
        np.testing.assert_array_equal(output[1, 2], embedding.embedding_matrix[5])

    def test_forward_rejects_non_integer_token_ids(self) -> None:
        embedding = _embedding_with_known_weights()

        for token_ids in (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([True, False], dtype=np.bool_),
        ):
            with self.subTest(dtype=token_ids.dtype):
                with self.assertRaises(TokenEmbeddingError):
                    embedding.forward(token_ids)

    def test_forward_rejects_token_ids_below_vocabulary_range(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(TokenEmbeddingError) as context:
            embedding.forward(np.array([0, -1], dtype=np.int64))

        self.assertIn("non-negative", str(context.exception))

    def test_forward_rejects_token_ids_above_vocabulary_range(self) -> None:
        embedding = _embedding_with_known_weights()

        with self.assertRaises(TokenEmbeddingError) as context:
            embedding.forward(np.array([0, 6], dtype=np.int64))

        self.assertIn("within vocabulary range", str(context.exception))

    def test_forward_rejects_unsupported_input_dimensions(self) -> None:
        embedding = _embedding_with_known_weights()

        for token_ids in (
            np.array(1, dtype=np.int64),
            np.zeros((1, 2, 3), dtype=np.int64),
        ):
            with self.subTest(shape=token_ids.shape):
                with self.assertRaises(TokenEmbeddingError):
                    embedding.forward(token_ids)

    def test_random_initialization_is_deterministic_with_seed(self) -> None:
        config = _config(weight_init_std=0.05)

        first = TokenEmbedding(config, random_seed=123)
        second = TokenEmbedding(config, random_seed=123)
        different = TokenEmbedding(config, random_seed=456)

        np.testing.assert_array_equal(first.embedding_matrix, second.embedding_matrix)
        self.assertFalse(np.array_equal(first.embedding_matrix, different.embedding_matrix))

    def test_random_initialization_uses_configured_standard_deviation(self) -> None:
        config = TransformerConfig(
            vocabulary_size=2048,
            context_length=8,
            embedding_dim=16,
            feed_forward_dim=64,
            num_layers=1,
            num_attention_heads=4,
            dropout_probability=0.0,
            weight_init_std=0.05,
        )

        embedding = TokenEmbedding(config, random_seed=1)

        self.assertAlmostEqual(float(np.std(embedding.embedding_matrix)), 0.05, places=2)

    def test_embedding_lookup_returns_exact_rows_from_matrix(self) -> None:
        embedding = _embedding_with_known_weights()

        output = embedding.forward(np.array([[5, 0], [2, 2]], dtype=np.int64))

        expected = np.array(
            [
                [embedding.embedding_matrix[5], embedding.embedding_matrix[0]],
                [embedding.embedding_matrix[2], embedding.embedding_matrix[2]],
            ]
        )
        np.testing.assert_array_equal(output, expected)

    def test_embedding_matrix_is_copied_when_supplied(self) -> None:
        weights = np.arange(24, dtype=np.float64).reshape(6, 4)
        embedding = TokenEmbedding(_known_weights_config(), embedding_matrix=weights)

        weights[0, 0] = -99.0

        self.assertEqual(embedding.embedding_matrix[0, 0], 0.0)

    def test_rejects_embedding_matrix_with_wrong_shape(self) -> None:
        with self.assertRaises(TokenEmbeddingError):
            TokenEmbedding(_config(), embedding_matrix=np.zeros((5, 4), dtype=np.float64))

    def test_rejects_embedding_matrix_with_non_finite_values(self) -> None:
        weights = np.arange(24, dtype=np.float64).reshape(6, 4)
        weights[0, 0] = np.nan

        with self.assertRaises(TokenEmbeddingError):
            TokenEmbedding(_known_weights_config(), embedding_matrix=weights)


def _config(weight_init_std: float = 0.02) -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=6,
        context_length=8,
        embedding_dim=8,
        feed_forward_dim=32,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=0.0,
        weight_init_std=weight_init_std,
    )


def _embedding_with_known_weights() -> TokenEmbedding:
    weights = np.arange(24, dtype=np.float64).reshape(6, 4)
    return TokenEmbedding(_known_weights_config(), embedding_matrix=weights)


def _known_weights_config() -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=6,
        context_length=8,
        embedding_dim=4,
        feed_forward_dim=16,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=0.0,
        weight_init_std=0.02,
    )
