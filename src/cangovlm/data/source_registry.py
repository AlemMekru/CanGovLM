"""Official source registry for the CanGovLM corpus.

The registry is the first executable layer of the corpus pipeline. It records
approved or reviewed official sources, validates their metadata, and loads them
from configuration without downloading any documents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_LANGUAGES = frozenset({"en", "fr", "bilingual"})
SUPPORTED_DOCUMENT_FORMATS = frozenset({"html", "pdf", "txt", "xml", "json", "csv", "docx"})
SUPPORTED_COLLECTION_METHODS = frozenset(
    {
        "api",
        "dataset_catalog",
        "manual_snapshot",
        "rss",
        "sitemap",
        "static_url_list",
    }
)

SOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class SourceRegistryError(ValueError):
    """Base error for invalid source registry data."""


class SourceValidationError(SourceRegistryError):
    """Raised when one or more sources fail validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class SourceLicense:
    """License metadata for an official source."""

    name: str
    url: str | None = None


@dataclass(frozen=True)
class OfficialSource:
    """Metadata for one official corpus source."""

    source_id: str
    name: str
    organization: str
    base_url: str
    languages: tuple[str, ...]
    license: SourceLicense
    document_formats: tuple[str, ...]
    collection_method: str
    enabled: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "OfficialSource":
        """Create a source from configuration data."""

        license_data = data.get("license")
        if isinstance(license_data, dict):
            source_license = SourceLicense(
                name=str(license_data.get("name", "")),
                url=_optional_string(license_data.get("url")),
            )
        else:
            source_license = SourceLicense(name=str(license_data or ""))

        return cls(
            source_id=str(data.get("source_id", "")),
            name=str(data.get("name", "")),
            organization=str(data.get("organization", "")),
            base_url=str(data.get("base_url", "")),
            languages=_string_tuple(data.get("languages")),
            license=source_license,
            document_formats=_string_tuple(data.get("document_formats")),
            collection_method=str(data.get("collection_method", "")),
            enabled=bool(data.get("enabled", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class SourceRegistry:
    """Validated collection of official corpus sources."""

    sources: tuple[OfficialSource, ...]

    def __post_init__(self) -> None:
        validate_sources(self.sources)

    def get(self, source_id: str) -> OfficialSource:
        """Return one source by ID."""

        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"Unknown source_id: {source_id}")

    def enabled_sources(self) -> tuple[OfficialSource, ...]:
        """Return sources currently enabled for collection."""

        return tuple(source for source in self.sources if source.enabled)

    def disabled_sources(self) -> tuple[OfficialSource, ...]:
        """Return sources currently disabled or excluded from collection."""

        return tuple(source for source in self.sources if not source.enabled)

    def for_language(self, language: str, *, enabled_only: bool = True) -> tuple[OfficialSource, ...]:
        """Return sources that support a language."""

        if language not in {"en", "fr"}:
            raise ValueError(f"Language filter must be 'en' or 'fr', got {language!r}")

        candidates = self.enabled_sources() if enabled_only else self.sources
        return tuple(
            source
            for source in candidates
            if language in source.languages or "bilingual" in source.languages
        )


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Load and validate a source registry JSON file."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise SourceValidationError(["Source registry config must be a JSON object."])

    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list):
        raise SourceValidationError(["Source registry config must contain a 'sources' list."])

    sources = tuple(OfficialSource.from_dict(source) for source in raw_sources)
    return SourceRegistry(sources=sources)


def validate_sources(sources: Iterable[OfficialSource]) -> None:
    """Validate source metadata and raise one aggregated error if needed."""

    source_list = list(sources)
    errors: list[str] = []

    if not source_list:
        errors.append("Source registry must contain at least one source.")

    seen_source_ids: set[str] = set()
    for source in source_list:
        errors.extend(_validate_source(source))
        if source.source_id in seen_source_ids:
            errors.append(f"{source.source_id}: duplicate source_id.")
        seen_source_ids.add(source.source_id)

    if errors:
        raise SourceValidationError(errors)


def _validate_source(source: OfficialSource) -> list[str]:
    errors: list[str] = []
    prefix = source.source_id or "<missing source_id>"

    if not SOURCE_ID_PATTERN.fullmatch(source.source_id):
        errors.append(f"{prefix}: source_id must be lowercase snake_case.")
    if not source.name.strip():
        errors.append(f"{prefix}: name is required.")
    if not source.organization.strip():
        errors.append(f"{prefix}: organization is required.")
    if not _is_valid_http_url(source.base_url):
        errors.append(f"{prefix}: base_url must be an absolute http(s) URL.")
    if not source.languages:
        errors.append(f"{prefix}: at least one language is required.")
    for language in source.languages:
        if language not in SUPPORTED_LANGUAGES:
            errors.append(f"{prefix}: unsupported language {language!r}.")
    if not source.license.name.strip():
        errors.append(f"{prefix}: license.name is required.")
    if source.license.url is not None and not _is_valid_http_url(source.license.url):
        errors.append(f"{prefix}: license.url must be an absolute http(s) URL when provided.")
    if not source.document_formats:
        errors.append(f"{prefix}: at least one document format is required.")
    for document_format in source.document_formats:
        if document_format not in SUPPORTED_DOCUMENT_FORMATS:
            errors.append(f"{prefix}: unsupported document format {document_format!r}.")
    if source.collection_method not in SUPPORTED_COLLECTION_METHODS:
        errors.append(f"{prefix}: unsupported collection method {source.collection_method!r}.")

    return errors


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

