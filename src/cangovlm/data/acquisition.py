"""Document acquisition framework for CanGovLM.

This module defines acquisition interfaces and raw-document metadata. It does
not implement network requests, scraping, parsing, cleaning, or extraction.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from cangovlm.data.source_registry import OfficialSource

ACQUIRABLE_DOCUMENT_FORMATS = frozenset({"html", "pdf", "xml", "json", "txt", "csv"})
DOCUMENT_FORMAT_EXTENSIONS = {
    "html": ".html",
    "pdf": ".pdf",
    "xml": ".xml",
    "json": ".json",
    "txt": ".txt",
    "csv": ".csv",
}

DOCUMENT_ID_HASH_LENGTH = 12
DOCUMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*_(?:en|fr)_[a-f0-9]{12}$")


class AcquisitionError(ValueError):
    """Base error for acquisition framework validation failures."""


class SourceAcquirer(ABC):
    """Source-specific abstraction for future document discovery.

    Implementations may later discover URLs from sitemaps, APIs, static URL
    lists, or dataset catalogs. They must not parse or clean document content.
    """

    def __init__(self, source: OfficialSource) -> None:
        self.source = source

    @abstractmethod
    def iter_requests(self) -> Iterable["AcquisitionRequest"]:
        """Yield acquisition requests for this source."""


class AcquisitionClient(ABC):
    """Interface for future clients that turn requests into raw documents."""

    @abstractmethod
    def acquire(self, request: "AcquisitionRequest") -> "RawDocument":
        """Acquire one raw document for a request."""


@dataclass(frozen=True)
class AcquisitionRequest:
    """Request metadata for one future raw document acquisition."""

    source_id: str
    url: str
    language: str
    document_format: str
    document_id: str | None = None
    canonical_url: str | None = None
    expected_content_type: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_language(self.language)
        _validate_document_format(self.document_format)
        _validate_http_url(self.url, "url")
        if self.canonical_url is not None:
            _validate_http_url(self.canonical_url, "canonical_url")

    def resolved_document_id(self) -> str:
        """Return the configured document ID or derive one from source and URL."""

        if self.document_id:
            return self.document_id
        return stable_document_id(
            source_id=self.source_id,
            language=self.language,
            stable_identity=self.canonical_url or self.url,
        )


@dataclass(frozen=True)
class RawDocument:
    """Immutable raw document bytes plus acquisition metadata."""

    document_id: str
    source_id: str
    url: str
    language: str
    document_format: str
    content: bytes
    retrieved_at: datetime
    canonical_url: str | None = None
    content_type: str | None = None
    status_code: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_language(self.language)
        _validate_document_format(self.document_format)
        _validate_http_url(self.url, "url")
        if self.canonical_url is not None:
            _validate_http_url(self.canonical_url, "canonical_url")
        if not isinstance(self.content, bytes):
            raise AcquisitionError("RawDocument.content must be bytes.")
        if not self.content:
            raise AcquisitionError("RawDocument.content must not be empty.")

    @property
    def sha256(self) -> str:
        """SHA-256 hash of immutable raw content bytes."""

        return sha256_bytes(self.content)

    @property
    def byte_size(self) -> int:
        """Raw content size in bytes."""

        return len(self.content)


@dataclass(frozen=True)
class DownloadManifestRecord:
    """Manifest schema for one immutable raw document."""

    document_id: str
    source_id: str
    source_name: str
    organization: str
    language: str
    url: str
    canonical_url: str | None
    retrieved_at: str
    document_format: str
    content_type: str | None
    raw_path: str
    raw_sha256: str
    byte_size: int
    collection_method: str
    license_name: str
    license_url: str | None
    status_code: int | None
    enabled_source: bool
    notes: str

    @classmethod
    def from_raw_document(
        cls,
        raw_document: RawDocument,
        source: OfficialSource,
        *,
        raw_path: str | Path,
    ) -> "DownloadManifestRecord":
        """Create a manifest record for a stored raw document."""

        if raw_document.source_id != source.source_id:
            raise AcquisitionError(
                "Raw document source_id does not match source metadata: "
                f"{raw_document.source_id!r} != {source.source_id!r}"
            )

        return cls(
            document_id=raw_document.document_id,
            source_id=source.source_id,
            source_name=source.name,
            organization=source.organization,
            language=raw_document.language,
            url=raw_document.url,
            canonical_url=raw_document.canonical_url,
            retrieved_at=raw_document.retrieved_at.astimezone(timezone.utc).isoformat(),
            document_format=raw_document.document_format,
            content_type=raw_document.content_type,
            raw_path=str(raw_path),
            raw_sha256=raw_document.sha256,
            byte_size=raw_document.byte_size,
            collection_method=source.collection_method,
            license_name=source.license.name,
            license_url=source.license.url,
            status_code=raw_document.status_code,
            enabled_source=source.enabled,
            notes=source.notes,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest record."""

        return {
            "document_id": self.document_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "organization": self.organization,
            "language": self.language,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "retrieved_at": self.retrieved_at,
            "document_format": self.document_format,
            "content_type": self.content_type,
            "raw_path": self.raw_path,
            "raw_sha256": self.raw_sha256,
            "byte_size": self.byte_size,
            "collection_method": self.collection_method,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "status_code": self.status_code,
            "enabled_source": self.enabled_source,
            "notes": self.notes,
        }


def stable_document_id(*, source_id: str, language: str, stable_identity: str) -> str:
    """Generate a stable document ID from source, language, and identity."""

    _validate_language(language)
    digest = sha256_text(f"{source_id}\n{language}\n{stable_identity}")[:DOCUMENT_ID_HASH_LENGTH]
    return f"{source_id}_{language}_{digest}"


def extension_for_format(document_format: str) -> str:
    """Return the file extension for a supported raw document format."""

    _validate_document_format(document_format)
    return DOCUMENT_FORMAT_EXTENSIONS[document_format]


def raw_storage_path(corpus_root: str | Path, raw_document: RawDocument) -> Path:
    """Generate the immutable raw storage path for a document."""

    extension = extension_for_format(raw_document.document_format)
    filename = f"{raw_document.document_id}{extension}"
    return Path(corpus_root) / "raw" / raw_document.language / raw_document.source_id / filename


def request_storage_path(
    corpus_root: str | Path,
    request: AcquisitionRequest,
) -> Path:
    """Generate the future raw storage path for an acquisition request."""

    extension = extension_for_format(request.document_format)
    filename = f"{request.resolved_document_id()}{extension}"
    return Path(corpus_root) / "raw" / request.language / request.source_id / filename


def sha256_bytes(content: bytes) -> str:
    """Compute a SHA-256 digest for bytes."""

    return hashlib.sha256(content).hexdigest()


def sha256_text(text: str) -> str:
    """Compute a SHA-256 digest for UTF-8 text."""

    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest for a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_language(language: str) -> None:
    if language not in {"en", "fr"}:
        raise AcquisitionError(f"language must be 'en' or 'fr', got {language!r}.")


def _validate_document_format(document_format: str) -> None:
    if document_format not in ACQUIRABLE_DOCUMENT_FORMATS:
        raise AcquisitionError(f"unsupported document format {document_format!r}.")


def _validate_http_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AcquisitionError(f"{field_name} must be an absolute http(s) URL.")
