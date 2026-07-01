"""Data pipeline primitives for CanGovLM."""

from cangovlm.data.source_registry import (
    SUPPORTED_COLLECTION_METHODS,
    SUPPORTED_DOCUMENT_FORMATS,
    SUPPORTED_LANGUAGES,
    OfficialSource,
    SourceLicense,
    SourceRegistry,
    SourceRegistryError,
    SourceValidationError,
    load_source_registry,
    validate_sources,
)

__all__ = [
    "SUPPORTED_COLLECTION_METHODS",
    "SUPPORTED_DOCUMENT_FORMATS",
    "SUPPORTED_LANGUAGES",
    "OfficialSource",
    "SourceLicense",
    "SourceRegistry",
    "SourceRegistryError",
    "SourceValidationError",
    "load_source_registry",
    "validate_sources",
]
