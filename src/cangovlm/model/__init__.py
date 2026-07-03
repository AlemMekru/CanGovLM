"""Model configuration primitives for CanGovLM."""

from cangovlm.model.config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_DROPOUT_PROBABILITY,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_FEED_FORWARD_DIM,
    DEFAULT_NUM_ATTENTION_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TRANSFORMER_CONFIG,
    DEFAULT_VOCABULARY_SIZE,
    DEFAULT_WEIGHT_INIT_STD,
    TransformerConfig,
    TransformerConfigError,
    load_transformer_config,
    validate_transformer_config,
    write_transformer_config,
)

__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_DROPOUT_PROBABILITY",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_FEED_FORWARD_DIM",
    "DEFAULT_NUM_ATTENTION_HEADS",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_TRANSFORMER_CONFIG",
    "DEFAULT_VOCABULARY_SIZE",
    "DEFAULT_WEIGHT_INIT_STD",
    "TransformerConfig",
    "TransformerConfigError",
    "load_transformer_config",
    "validate_transformer_config",
    "write_transformer_config",
]
