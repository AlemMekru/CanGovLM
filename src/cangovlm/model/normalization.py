"""Normalization layers for CanGovLM."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from cangovlm.model.config import TransformerConfig


class LayerNormError(ValueError):
    """Raised when LayerNorm parameters or inputs are invalid."""


class LayerNorm:
    """NumPy layer normalization over the final tensor dimension."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.epsilon = config.layer_norm_epsilon
        self.gamma = self._parameter_or_default(gamma, default=1.0, name="gamma")
        self.beta = self._parameter_or_default(beta, default=0.0, name="beta")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize inputs over the final dimension and apply gamma/beta."""

        values = self._validate_inputs(inputs)
        mean = np.mean(values, axis=-1, keepdims=True)
        variance = np.var(values, axis=-1, keepdims=True)
        normalized = (values - mean) / np.sqrt(variance + self.epsilon)
        return normalized * self.gamma + self.beta

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible LayerNorm state."""

        return {
            "epsilon": self.epsilon,
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
        }

    @classmethod
    def from_dict(cls, config: TransformerConfig, data: dict[str, object]) -> "LayerNorm":
        """Create LayerNorm from JSON-compatible state."""

        if not isinstance(data, dict):
            raise LayerNormError("LayerNorm data must be a JSON object")

        epsilon = _float_state_value(data.get("epsilon", config.layer_norm_epsilon), "epsilon")
        if epsilon != config.layer_norm_epsilon:
            raise LayerNormError(
                "LayerNorm epsilon must match TransformerConfig.layer_norm_epsilon"
            )

        return cls(
            config,
            gamma=np.asarray(data.get("gamma"), dtype=np.float64),
            beta=np.asarray(data.get("beta"), dtype=np.float64),
        )

    def to_json(self) -> str:
        """Serialize LayerNorm state as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, config: TransformerConfig, text: str) -> "LayerNorm":
        """Load LayerNorm state from a JSON string."""

        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise LayerNormError("Invalid LayerNorm JSON") from error
        return cls.from_dict(config, data)

    def _parameter_or_default(
        self,
        value: np.ndarray | None,
        *,
        default: float,
        name: str,
    ) -> np.ndarray:
        if value is None:
            parameter = np.full((self.config.embedding_dim,), default, dtype=np.float64)
        else:
            parameter = np.array(value, dtype=np.float64, copy=True)

        if parameter.shape != (self.config.embedding_dim,):
            raise LayerNormError(
                f"{name} must have shape ({self.config.embedding_dim},), got {parameter.shape}"
            )
        if not np.all(np.isfinite(parameter)):
            raise LayerNormError(f"{name} must contain only finite values")
        return parameter

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs)

        if values.ndim not in {2, 3}:
            raise LayerNormError("inputs must be a 2D or 3D tensor")
        if values.shape[-1] != self.config.embedding_dim:
            raise LayerNormError("last input dimension must match TransformerConfig.embedding_dim")
        if not np.issubdtype(values.dtype, np.floating):
            raise LayerNormError("inputs must contain floating point values")
        if not np.all(np.isfinite(values)):
            raise LayerNormError("inputs must contain only finite values")
        return values


def _float_state_value(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise LayerNormError(f"{name} must be a real number") from error
