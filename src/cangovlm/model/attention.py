"""Self-attention layers for CanGovLM."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from cangovlm.model.config import TransformerConfig


class MultiHeadSelfAttentionError(ValueError):
    """Raised when attention configuration, parameters, or inputs are invalid."""


class MultiHeadSelfAttention:
    """NumPy decoder-only multi-head self-attention."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        random_seed: int | None = None,
        query_projection: np.ndarray | None = None,
        key_projection: np.ndarray | None = None,
        value_projection: np.ndarray | None = None,
        output_projection: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.dropout_probability = config.dropout_probability
        self._validate_config()

        rng = np.random.default_rng(random_seed)
        self.query_projection = self._projection_or_random(query_projection, rng)
        self.key_projection = self._projection_or_random(key_projection, rng)
        self.value_projection = self._projection_or_random(value_projection, rng)
        self.output_projection = self._projection_or_random(output_projection, rng)

    def forward(
        self,
        inputs: np.ndarray,
        *,
        return_attention_weights: bool = False,
        training: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply causal multi-head self-attention."""

        values, squeeze_batch = self._validate_inputs(inputs)
        query = values @ self.query_projection
        key = values @ self.key_projection
        value = values @ self.value_projection

        query_heads = self.split_heads(query)
        key_heads = self.split_heads(key)
        value_heads = self.split_heads(value)

        attention_weights = self._scaled_dot_product_attention(query_heads, key_heads)
        attention_weights = self._apply_attention_dropout(attention_weights, training=training)
        context_heads = attention_weights @ value_heads
        context = self.merge_heads(context_heads)
        output = context @ self.output_projection

        if squeeze_batch:
            output = output[0]
            attention_weights = attention_weights[0]

        if return_attention_weights:
            return output, attention_weights
        return output

    def split_heads(self, inputs: np.ndarray) -> np.ndarray:
        """Split a batched embedding tensor into attention heads."""

        values, _ = self._validate_inputs(inputs)
        batch_size, sequence_length, _ = values.shape
        reshaped = values.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        return np.transpose(reshaped, (0, 2, 1, 3))

    def merge_heads(self, inputs: np.ndarray) -> np.ndarray:
        """Merge a batched head tensor back to embedding dimensions."""

        values = np.asarray(inputs)
        if values.ndim != 4:
            raise MultiHeadSelfAttentionError("head tensor must be 4D")
        if values.shape[1] != self.num_heads or values.shape[3] != self.head_dim:
            raise MultiHeadSelfAttentionError(
                "head tensor must have shape "
                f"(batch, {self.num_heads}, sequence, {self.head_dim})"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise MultiHeadSelfAttentionError("head tensor must contain floating point values")
        if not np.all(np.isfinite(values)):
            raise MultiHeadSelfAttentionError("head tensor must contain only finite values")

        batch_size, _, sequence_length, _ = values.shape
        transposed = np.transpose(values, (0, 2, 1, 3))
        return transposed.reshape(batch_size, sequence_length, self.embedding_dim)

    def causal_mask(self, sequence_length: int) -> np.ndarray:
        """Return a lower-triangular decoder attention mask."""

        if not isinstance(sequence_length, (int, np.integer)) or isinstance(sequence_length, bool):
            raise MultiHeadSelfAttentionError("sequence_length must be an integer")
        sequence_length = int(sequence_length)
        if sequence_length < 0:
            raise MultiHeadSelfAttentionError("sequence_length must be non-negative")
        if sequence_length > self.config.context_length:
            raise MultiHeadSelfAttentionError("sequence_length cannot exceed context_length")
        return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible attention state."""

        return {
            "dropout_probability": self.dropout_probability,
            "query_projection": self.query_projection.tolist(),
            "key_projection": self.key_projection.tolist(),
            "value_projection": self.value_projection.tolist(),
            "output_projection": self.output_projection.tolist(),
        }

    @classmethod
    def from_dict(
        cls,
        config: TransformerConfig,
        data: dict[str, object],
    ) -> "MultiHeadSelfAttention":
        """Create attention from JSON-compatible state."""

        if not isinstance(data, dict):
            raise MultiHeadSelfAttentionError("attention data must be a JSON object")

        dropout_probability = _float_state_value(
            data.get("dropout_probability", config.dropout_probability),
            "dropout_probability",
        )
        if dropout_probability != config.dropout_probability:
            raise MultiHeadSelfAttentionError(
                "attention dropout_probability must match TransformerConfig.dropout_probability"
            )

        return cls(
            config,
            query_projection=_array_state_value(data, "query_projection"),
            key_projection=_array_state_value(data, "key_projection"),
            value_projection=_array_state_value(data, "value_projection"),
            output_projection=_array_state_value(data, "output_projection"),
        )

    def to_json(self) -> str:
        """Serialize attention state as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, config: TransformerConfig, text: str) -> "MultiHeadSelfAttention":
        """Load attention state from a JSON string."""

        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise MultiHeadSelfAttentionError("Invalid attention JSON") from error
        return cls.from_dict(config, data)

    def _scaled_dot_product_attention(
        self,
        query_heads: np.ndarray,
        key_heads: np.ndarray,
    ) -> np.ndarray:
        scores = query_heads @ np.swapaxes(key_heads, -1, -2)
        scores = scores / math.sqrt(self.head_dim)
        sequence_length = scores.shape[-1]
        mask = self.causal_mask(sequence_length)
        masked_scores = np.where(mask, scores, -np.inf)
        return _softmax(masked_scores, axis=-1)

    def _apply_attention_dropout(
        self,
        attention_weights: np.ndarray,
        *,
        training: bool,
    ) -> np.ndarray:
        if training and self.dropout_probability > 0.0:
            raise MultiHeadSelfAttentionError(
                "attention dropout training behavior is not implemented yet"
            )
        return attention_weights

    def _projection_or_random(
        self,
        value: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if value is None:
            return rng.normal(
                loc=0.0,
                scale=self.config.weight_init_std,
                size=(self.embedding_dim, self.embedding_dim),
            )

        projection = np.array(value, dtype=np.float64, copy=True)
        expected_shape = (self.embedding_dim, self.embedding_dim)
        if projection.shape != expected_shape:
            raise MultiHeadSelfAttentionError(
                f"projection matrices must have shape {expected_shape}, got {projection.shape}"
            )
        if not np.all(np.isfinite(projection)):
            raise MultiHeadSelfAttentionError("projection matrices must contain only finite values")
        return projection

    def _validate_config(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise MultiHeadSelfAttentionError(
                "embedding_dim must be divisible by num_attention_heads"
            )
        if self.config.attention_dim != self.embedding_dim:
            raise MultiHeadSelfAttentionError("attention dimensions must match embedding_dim")

    def _validate_inputs(self, inputs: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(inputs)
        squeeze_batch = False

        if values.ndim == 2:
            values = values[np.newaxis, :, :]
            squeeze_batch = True
        elif values.ndim != 3:
            raise MultiHeadSelfAttentionError("inputs must be a 2D or 3D tensor")

        if values.shape[-1] != self.embedding_dim:
            raise MultiHeadSelfAttentionError(
                "last input dimension must match TransformerConfig.embedding_dim"
            )
        if values.shape[1] > self.config.context_length:
            raise MultiHeadSelfAttentionError("sequence length cannot exceed context_length")
        if not np.issubdtype(values.dtype, np.floating):
            raise MultiHeadSelfAttentionError("inputs must contain floating point values")
        if not np.all(np.isfinite(values)):
            raise MultiHeadSelfAttentionError("inputs must contain only finite values")
        return values, squeeze_batch


def _softmax(values: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def _float_state_value(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise MultiHeadSelfAttentionError(f"{name} must be a real number") from error


def _array_state_value(data: dict[str, object], name: str) -> np.ndarray:
    if name not in data:
        raise MultiHeadSelfAttentionError(f"{name} is required")
    try:
        return np.asarray(data[name], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MultiHeadSelfAttentionError(f"{name} must be numeric array data") from error
