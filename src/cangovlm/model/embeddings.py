"""Embedding layers for CanGovLM.

This module implements token and positional embedding lookup only. It
intentionally contains no attention, normalization, Transformer blocks, or
training logic.
"""

from __future__ import annotations

import numpy as np

from cangovlm.model.config import TransformerConfig


class TokenEmbeddingError(ValueError):
    """Raised when token embedding inputs or weights are invalid."""


class PositionalEmbeddingError(ValueError):
    """Raised when positional embedding inputs or weights are invalid."""


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


class PositionalEmbedding:
    """NumPy learnable positional embedding lookup table."""

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
                size=(config.context_length, config.embedding_dim),
            )
        else:
            self.embedding_matrix = np.array(embedding_matrix, dtype=np.float64, copy=True)
            self._validate_embedding_matrix()

    @property
    def context_length(self) -> int:
        """Maximum supported sequence length."""

        return self.embedding_matrix.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Width of each positional embedding vector."""

        return self.embedding_matrix.shape[1]

    def forward(self, sequence_or_embeddings: int | np.ndarray) -> np.ndarray:
        """Return positional embeddings for a sequence length or embedding tensor."""

        sequence_length, batch_size = self._input_shape(sequence_or_embeddings)
        positions = self.embedding_matrix[:sequence_length]

        if batch_size is None:
            return positions
        return np.broadcast_to(positions, (batch_size, sequence_length, self.embedding_dim)).copy()

    def combine_with_token_embeddings(self, token_embeddings: np.ndarray) -> np.ndarray:
        """Add positional embeddings to token embeddings."""

        embeddings = np.asarray(token_embeddings)
        if not np.issubdtype(embeddings.dtype, np.floating):
            raise PositionalEmbeddingError("token_embeddings must contain floating point values")
        return embeddings + self.forward(embeddings)

    def _validate_embedding_matrix(self) -> None:
        expected_shape = (self.config.context_length, self.config.embedding_dim)
        if self.embedding_matrix.shape != expected_shape:
            raise PositionalEmbeddingError(
                "embedding_matrix must have shape "
                f"{expected_shape}, got {self.embedding_matrix.shape}"
            )
        if not np.issubdtype(self.embedding_matrix.dtype, np.floating):
            raise PositionalEmbeddingError("embedding_matrix must contain floating point values")
        if not np.all(np.isfinite(self.embedding_matrix)):
            raise PositionalEmbeddingError("embedding_matrix must contain only finite values")

    def _input_shape(self, sequence_or_embeddings: int | np.ndarray) -> tuple[int, int | None]:
        if isinstance(sequence_or_embeddings, (int, np.integer)) and not isinstance(
            sequence_or_embeddings, bool
        ):
            sequence_length = int(sequence_or_embeddings)
            batch_size = None
        else:
            embeddings = np.asarray(sequence_or_embeddings)
            if embeddings.ndim == 2:
                sequence_length = embeddings.shape[0]
                batch_size = None
            elif embeddings.ndim == 3:
                batch_size = embeddings.shape[0]
                sequence_length = embeddings.shape[1]
            else:
                raise PositionalEmbeddingError(
                    "input must be a sequence length, 2D embeddings, or 3D embeddings"
                )
            if embeddings.shape[-1] != self.config.embedding_dim:
                raise PositionalEmbeddingError(
                    "embedding dimension must match TransformerConfig.embedding_dim"
                )

        if sequence_length < 0:
            raise PositionalEmbeddingError("sequence length must be non-negative")
        if sequence_length > self.config.context_length:
            raise PositionalEmbeddingError("sequence length cannot exceed context_length")
        return sequence_length, batch_size
