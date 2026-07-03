"""Token embedding layer for CanGovLM.

This module implements token lookup only. It intentionally contains no
positional embeddings, attention, normalization, Transformer blocks, or
training logic.
"""

from __future__ import annotations

import numpy as np

from cangovlm.model.config import TransformerConfig


class TokenEmbeddingError(ValueError):
    """Raised when token embedding inputs or weights are invalid."""


class TokenEmbedding:
    """NumPy token embedding lookup table."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        random_seed: int | None = None,
        embedding_matrix: np.ndarray | None = None,
    ) -> None:
        self.config = config

        if embedding_matrix is None:
            rng = np.random.default_rng(random_seed)
            self.embedding_matrix = rng.normal(
                loc=0.0,
                scale=config.weight_init_std,
                size=(config.vocabulary_size, config.embedding_dim),
            )
        else:
            self.embedding_matrix = np.array(embedding_matrix, dtype=np.float64, copy=True)
            self._validate_embedding_matrix()

    @property
    def vocabulary_size(self) -> int:
        """Number of token rows in the embedding table."""

        return self.embedding_matrix.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Width of each token embedding vector."""

        return self.embedding_matrix.shape[1]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return embeddings for a 1D or 2D array of token IDs."""

        validated_ids = self._validate_token_ids(token_ids)
        return self.embedding_matrix[validated_ids]

    def _validate_embedding_matrix(self) -> None:
        expected_shape = (self.config.vocabulary_size, self.config.embedding_dim)
        if self.embedding_matrix.shape != expected_shape:
            raise TokenEmbeddingError(
                "embedding_matrix must have shape "
                f"{expected_shape}, got {self.embedding_matrix.shape}"
            )
        if not np.issubdtype(self.embedding_matrix.dtype, np.floating):
            raise TokenEmbeddingError("embedding_matrix must contain floating point values")
        if not np.all(np.isfinite(self.embedding_matrix)):
            raise TokenEmbeddingError("embedding_matrix must contain only finite values")

    def _validate_token_ids(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids)

        if ids.ndim not in {1, 2}:
            raise TokenEmbeddingError("token_ids must be a 1D or 2D array")
        if not np.issubdtype(ids.dtype, np.integer) or np.issubdtype(ids.dtype, np.bool_):
            raise TokenEmbeddingError("token_ids must contain integers")
        if ids.size == 0:
            return ids
        if np.any(ids < 0):
            raise TokenEmbeddingError("token_ids must be non-negative")
        if np.any(ids >= self.config.vocabulary_size):
            raise TokenEmbeddingError("token_ids must be within vocabulary range")
        return ids
