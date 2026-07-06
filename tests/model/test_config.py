from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.model import (
    DEFAULT_TRANSFORMER_CONFIG,
    TransformerConfig,
    TransformerConfigError,
    load_transformer_config,
    write_transformer_config,
)


class TransformerConfigTests(TestCase):
    def test_default_config_is_suitable_for_initial_cangovlm_model(self) -> None:
        config = DEFAULT_TRANSFORMER_CONFIG

        self.assertEqual(config.vocabulary_size, 259)
        self.assertEqual(config.context_length, 256)
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.feed_forward_dim, 512)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.dropout_probability, 0.1)
        self.assertEqual(config.weight_init_std, 0.02)
        self.assertEqual(config.layer_norm_epsilon, 1e-5)
        self.assertEqual(config.head_dim, 32)
        self.assertTrue(config.has_valid_attention_dimensions)
        self.assertEqual(config.parameter_sanity_checks, ())
        self.assertGreater(config.estimated_parameter_count, 0)

    def test_config_is_immutable_after_creation(self) -> None:
        config = TransformerConfig()

        with self.assertRaises(FrozenInstanceError):
            config.embedding_dim = 256

    def test_valid_config_computes_attention_dimensions(self) -> None:
        config = TransformerConfig(
            vocabulary_size=1024,
            context_length=128,
            embedding_dim=96,
            feed_forward_dim=384,
            num_layers=2,
            num_attention_heads=6,
            dropout_probability=0.0,
            weight_init_std=0.01,
            layer_norm_epsilon=1e-6,
        )

        self.assertEqual(config.head_dim, 16)
        self.assertEqual(config.attention_dim, 96)
        self.assertTrue(config.has_valid_attention_dimensions)

    def test_to_dict_and_from_dict_round_trip(self) -> None:
        config = TransformerConfig(
            vocabulary_size=2048,
            context_length=512,
            embedding_dim=256,
            feed_forward_dim=1024,
            num_layers=6,
            num_attention_heads=8,
            dropout_probability=0.2,
            weight_init_std=0.015,
            layer_norm_epsilon=1e-6,
        )

        loaded = TransformerConfig.from_dict(config.to_dict())

        self.assertEqual(loaded, config)

    def test_from_dict_uses_defaults_for_missing_fields(self) -> None:
        config = TransformerConfig.from_dict({"embedding_dim": 64, "num_attention_heads": 4})

        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.vocabulary_size, DEFAULT_TRANSFORMER_CONFIG.vocabulary_size)

    def test_json_serialization_round_trips(self) -> None:
        config = TransformerConfig(embedding_dim=64, feed_forward_dim=256, num_attention_heads=4)

        loaded = TransformerConfig.from_json(config.to_json())

        self.assertEqual(loaded, config)

    def test_write_and_load_transformer_config_round_trips_json_file(self) -> None:
        config = TransformerConfig(embedding_dim=64, feed_forward_dim=256, num_attention_heads=4)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transformer_config.json"
            write_transformer_config(path, config)
            loaded = load_transformer_config(path)

        self.assertEqual(loaded, config)

    def test_rejects_invalid_positive_integer_fields(self) -> None:
        invalid_values = {
            "vocabulary_size": 0,
            "context_length": -1,
            "embedding_dim": 0,
            "feed_forward_dim": 0,
            "num_layers": 0,
            "num_attention_heads": False,
        }

        for field_name, value in invalid_values.items():
            with self.subTest(field_name=field_name):
                with self.assertRaises(TransformerConfigError):
                    TransformerConfig(**{field_name: value})

    def test_rejects_invalid_dropout_probability(self) -> None:
        for value in (-0.1, 1.0, True):
            with self.subTest(value=value):
                with self.assertRaises(TransformerConfigError):
                    TransformerConfig(dropout_probability=value)

    def test_rejects_invalid_weight_initialization_std(self) -> None:
        for value in (0.0, -0.1, "0.02"):
            with self.subTest(value=value):
                with self.assertRaises(TransformerConfigError):
                    TransformerConfig(weight_init_std=value)

    def test_rejects_invalid_layer_norm_epsilon(self) -> None:
        for value in (0.0, -0.1, "1e-5"):
            with self.subTest(value=value):
                with self.assertRaises(TransformerConfigError):
                    TransformerConfig(layer_norm_epsilon=value)

    def test_rejects_attention_dimensions_that_do_not_divide_evenly(self) -> None:
        with self.assertRaises(TransformerConfigError) as context:
            TransformerConfig(embedding_dim=130, num_attention_heads=8)

        self.assertIn("embedding_dim must be divisible", str(context.exception))

    def test_rejects_attention_head_count_larger_than_embedding_dimension(self) -> None:
        with self.assertRaises(TransformerConfigError) as context:
            TransformerConfig(embedding_dim=4, num_attention_heads=8)

        self.assertIn("num_attention_heads cannot exceed embedding_dim", str(context.exception))

    def test_rejects_feed_forward_dimension_smaller_than_embedding_dimension(self) -> None:
        with self.assertRaises(TransformerConfigError) as context:
            TransformerConfig(embedding_dim=128, feed_forward_dim=64)

        self.assertIn("feed_forward_dim must be at least embedding_dim", str(context.exception))

    def test_parameter_sanity_checks_warn_for_unusual_valid_config(self) -> None:
        config = TransformerConfig(embedding_dim=16, feed_forward_dim=16, num_attention_heads=4)

        self.assertIn(
            "feed_forward_dim is less than 2x embedding_dim",
            config.parameter_sanity_checks,
        )
        self.assertIn("head_dim is very small", config.parameter_sanity_checks)

    def test_from_dict_rejects_unknown_fields(self) -> None:
        with self.assertRaises(TransformerConfigError) as context:
            TransformerConfig.from_dict({"embedding_dim": 64, "unknown": 1})

        self.assertIn("Unknown transformer config fields", str(context.exception))

    def test_from_json_rejects_invalid_json(self) -> None:
        with self.assertRaises(TransformerConfigError):
            TransformerConfig.from_json("{not json}")
