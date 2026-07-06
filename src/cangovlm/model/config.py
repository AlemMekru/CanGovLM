"""Transformer configuration primitives for CanGovLM.

This module defines model configuration only. It intentionally contains no
neural-network layers, tensor code, pretrained weights, or framework bindings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


DEFAULT_VOCABULARY_SIZE = 259
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_FEED_FORWARD_DIM = 512
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_ATTENTION_HEADS = 4
DEFAULT_DROPOUT_PROBABILITY = 0.1
DEFAULT_WEIGHT_INIT_STD = 0.02
DEFAULT_LAYER_NORM_EPSILON = 1e-5


class TransformerConfigError(ValueError):
    """Raised when transformer configuration data is invalid."""


@dataclass(frozen=True)
class TransformerConfig:
    """Immutable decoder-only Transformer configuration.

    The configuration is deliberately independent from any implementation of
    embeddings, attention, feed-forward layers, blocks, or forward passes.
    """

    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE
    context_length: int = DEFAULT_CONTEXT_LENGTH
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    feed_forward_dim: int = DEFAULT_FEED_FORWARD_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS
    dropout_probability: float = DEFAULT_DROPOUT_PROBABILITY
    weight_init_std: float = DEFAULT_WEIGHT_INIT_STD
    layer_norm_epsilon: float = DEFAULT_LAYER_NORM_EPSILON

    def __post_init__(self) -> None:
        validate_transformer_config(self)

    @property
    def head_dim(self) -> int:
        """Per-head attention dimension."""

        return self.embedding_dim // self.num_attention_heads

    @property
    def attention_dim(self) -> int:
        """Total attention dimension implied by all attention heads."""

        return self.head_dim * self.num_attention_heads

    @property
    def has_valid_attention_dimensions(self) -> bool:
        """Whether attention heads exactly cover the embedding dimension."""

        return self.attention_dim == self.embedding_dim

    @property
    def estimated_parameter_count(self) -> int:
        """Approximate parameter count for sanity checks and size comparison.

        The estimate covers a conventional decoder-only Transformer with tied
        output embeddings. It is informational only and does not bind the later
        model implementation to a specific parameter layout.
        """

        token_embeddings = self.vocabulary_size * self.embedding_dim
        positional_embeddings = self.context_length * self.embedding_dim
        attention_per_layer = 4 * self.embedding_dim * self.embedding_dim
        feed_forward_per_layer = 2 * self.embedding_dim * self.feed_forward_dim
        norm_and_bias_per_layer = 8 * self.embedding_dim + self.feed_forward_dim
        transformer_layers = self.num_layers * (
            attention_per_layer + feed_forward_per_layer + norm_and_bias_per_layer
        )
        final_norm = 2 * self.embedding_dim
        return token_embeddings + positional_embeddings + transformer_layers + final_norm

    @property
    def parameter_sanity_checks(self) -> tuple[str, ...]:
        """Human-readable warnings for unusual but valid model dimensions."""

        warnings: list[str] = []
        if self.feed_forward_dim < 2 * self.embedding_dim:
            warnings.append("feed_forward_dim is less than 2x embedding_dim")
        if self.head_dim < 8:
            warnings.append("head_dim is very small")
        if self.estimated_parameter_count > 1_000_000_000:
            warnings.append("estimated_parameter_count exceeds one billion")
        return tuple(warnings)

    def to_dict(self) -> dict[str, int | float]:
        """Return a deterministic JSON-compatible dictionary."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TransformerConfig":
        """Create a configuration from a JSON-compatible dictionary."""

        if not isinstance(data, dict):
            raise TransformerConfigError("TransformerConfig data must be a JSON object")

        known_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(data) - known_fields)
        if unknown_fields:
            raise TransformerConfigError(f"Unknown transformer config fields: {unknown_fields}")

        return cls(**{field.name: data.get(field.name, field.default) for field in fields(cls)})

    def to_json(self) -> str:
        """Serialize the configuration as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TransformerConfig":
        """Load a configuration from a JSON string."""

        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise TransformerConfigError("Invalid transformer config JSON") from error
        return cls.from_dict(data)


def validate_transformer_config(config: TransformerConfig) -> None:
    """Validate every TransformerConfig field."""

    _validate_positive_int("vocabulary_size", config.vocabulary_size)
    _validate_positive_int("context_length", config.context_length)
    _validate_positive_int("embedding_dim", config.embedding_dim)
    _validate_positive_int("feed_forward_dim", config.feed_forward_dim)
    _validate_positive_int("num_layers", config.num_layers)
    _validate_positive_int("num_attention_heads", config.num_attention_heads)
    _validate_probability("dropout_probability", config.dropout_probability)
    _validate_positive_float("weight_init_std", config.weight_init_std)
    _validate_positive_float("layer_norm_epsilon", config.layer_norm_epsilon)

    if config.num_attention_heads > config.embedding_dim:
        raise TransformerConfigError("num_attention_heads cannot exceed embedding_dim")
    if config.embedding_dim % config.num_attention_heads != 0:
        raise TransformerConfigError("embedding_dim must be divisible by num_attention_heads")
    if config.feed_forward_dim < config.embedding_dim:
        raise TransformerConfigError("feed_forward_dim must be at least embedding_dim")


def write_transformer_config(path: str | Path, config: TransformerConfig) -> None:
    """Write a TransformerConfig to a JSON file."""

    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.to_json() + "\n", encoding="utf-8")


def load_transformer_config(path: str | Path) -> TransformerConfig:
    """Load a TransformerConfig from a JSON file."""

    config_path = Path(path)
    return TransformerConfig.from_json(config_path.read_text(encoding="utf-8"))


def _validate_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TransformerConfigError(f"{name} must be an integer")
    if value <= 0:
        raise TransformerConfigError(f"{name} must be positive")


def _validate_probability(name: str, value: object) -> None:
    if not _is_real_number(value):
        raise TransformerConfigError(f"{name} must be a real number")
    if not 0.0 <= float(value) < 1.0:
        raise TransformerConfigError(f"{name} must be in the range [0.0, 1.0)")


def _validate_positive_float(name: str, value: object) -> None:
    if not _is_real_number(value):
        raise TransformerConfigError(f"{name} must be a real number")
    if float(value) <= 0.0:
        raise TransformerConfigError(f"{name} must be positive")


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


DEFAULT_TRANSFORMER_CONFIG = TransformerConfig()
