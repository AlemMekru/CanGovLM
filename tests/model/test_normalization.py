from __future__ import annotations

from unittest import TestCase

import numpy as np

from cangovlm.model import LayerNorm, LayerNormError, TransformerConfig


class LayerNormTests(TestCase):
    def test_initializes_gamma_beta_and_epsilon_from_config(self) -> None:
        config = _config(layer_norm_epsilon=1e-6)
        layer_norm = LayerNorm(config)

        np.testing.assert_array_equal(layer_norm.gamma, np.ones(4, dtype=np.float64))
        np.testing.assert_array_equal(layer_norm.beta, np.zeros(4, dtype=np.float64))
        self.assertEqual(layer_norm.epsilon, 1e-6)

    def test_forward_preserves_2d_shape(self) -> None:
        layer_norm = LayerNorm(_config())
        inputs = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])

        output = layer_norm.forward(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_forward_preserves_3d_shape(self) -> None:
        layer_norm = LayerNorm(_config())
        inputs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

        output = layer_norm.forward(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_forward_outputs_zero_mean_before_gamma_beta_shift(self) -> None:
        layer_norm = LayerNorm(_config(layer_norm_epsilon=1e-12))
        inputs = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]])

        output = layer_norm.forward(inputs)

        np.testing.assert_allclose(np.mean(output, axis=-1), np.zeros(2), atol=1e-10)

    def test_forward_outputs_unit_variance_before_gamma_beta_scale(self) -> None:
        layer_norm = LayerNorm(_config(layer_norm_epsilon=1e-12))
        inputs = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]])

        output = layer_norm.forward(inputs)

        np.testing.assert_allclose(np.var(output, axis=-1), np.ones(2), atol=1e-10)

    def test_forward_applies_gamma_and_beta(self) -> None:
        config = _config(layer_norm_epsilon=1e-12)
        gamma = np.array([1.0, 2.0, 3.0, 4.0])
        beta = np.array([0.5, -0.5, 1.0, -1.0])
        layer_norm = LayerNorm(config, gamma=gamma, beta=beta)
        inputs = np.array([[1.0, 2.0, 3.0, 4.0]])

        output = layer_norm.forward(inputs)

        mean = np.mean(inputs, axis=-1, keepdims=True)
        variance = np.var(inputs, axis=-1, keepdims=True)
        expected = ((inputs - mean) / np.sqrt(variance + config.layer_norm_epsilon)) * gamma + beta
        np.testing.assert_allclose(output, expected, atol=1e-12)

    def test_epsilon_prevents_division_by_zero_for_constant_inputs(self) -> None:
        layer_norm = LayerNorm(_config(layer_norm_epsilon=1e-3))
        inputs = np.full((2, 4), 7.0)

        output = layer_norm.forward(inputs)

        np.testing.assert_array_equal(output, np.zeros_like(inputs))

    def test_epsilon_changes_output_for_low_variance_inputs(self) -> None:
        inputs = np.array([[1.0, 1.0, 1.0, 1.001]])
        small_epsilon = LayerNorm(_config(layer_norm_epsilon=1e-12))
        large_epsilon = LayerNorm(_config(layer_norm_epsilon=1e-2))

        small_output = small_epsilon.forward(inputs)
        large_output = large_epsilon.forward(inputs)

        self.assertGreater(np.linalg.norm(small_output), np.linalg.norm(large_output))

    def test_rejects_invalid_input_dimensions(self) -> None:
        layer_norm = LayerNorm(_config())

        for inputs in (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.zeros((1, 2, 3, 4), dtype=np.float64),
        ):
            with self.subTest(shape=inputs.shape):
                with self.assertRaises(LayerNormError):
                    layer_norm.forward(inputs)

    def test_rejects_wrong_embedding_dimension(self) -> None:
        layer_norm = LayerNorm(_config())

        with self.assertRaises(LayerNormError) as context:
            layer_norm.forward(np.zeros((2, 3), dtype=np.float64))

        self.assertIn("last input dimension", str(context.exception))

    def test_rejects_non_float_inputs(self) -> None:
        layer_norm = LayerNorm(_config())

        with self.assertRaises(LayerNormError):
            layer_norm.forward(np.zeros((2, 4), dtype=np.int64))

    def test_initialization_is_deterministic(self) -> None:
        first = LayerNorm(_config())
        second = LayerNorm(_config())

        np.testing.assert_array_equal(first.gamma, second.gamma)
        np.testing.assert_array_equal(first.beta, second.beta)

    def test_gamma_and_beta_are_copied_when_supplied(self) -> None:
        gamma = np.array([1.0, 2.0, 3.0, 4.0])
        beta = np.array([4.0, 3.0, 2.0, 1.0])
        layer_norm = LayerNorm(_config(), gamma=gamma, beta=beta)

        gamma[0] = -99.0
        beta[0] = -99.0

        self.assertEqual(layer_norm.gamma[0], 1.0)
        self.assertEqual(layer_norm.beta[0], 4.0)

    def test_rejects_invalid_gamma_or_beta_shape(self) -> None:
        with self.assertRaises(LayerNormError):
            LayerNorm(_config(), gamma=np.ones(3), beta=np.zeros(4))

        with self.assertRaises(LayerNormError):
            LayerNorm(_config(), gamma=np.ones(4), beta=np.zeros(3))

    def test_json_state_round_trips(self) -> None:
        config = _config(layer_norm_epsilon=1e-6)
        layer_norm = LayerNorm(
            config,
            gamma=np.array([1.0, 2.0, 3.0, 4.0]),
            beta=np.array([0.5, 0.25, -0.25, -0.5]),
        )

        loaded = LayerNorm.from_json(config, layer_norm.to_json())

        self.assertEqual(loaded.epsilon, layer_norm.epsilon)
        np.testing.assert_array_equal(loaded.gamma, layer_norm.gamma)
        np.testing.assert_array_equal(loaded.beta, layer_norm.beta)


def _config(layer_norm_epsilon: float = 1e-5) -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=16,
        context_length=8,
        embedding_dim=4,
        feed_forward_dim=16,
        num_layers=1,
        num_attention_heads=2,
        dropout_probability=0.0,
        weight_init_std=0.02,
        layer_norm_epsilon=layer_norm_epsilon,
    )
