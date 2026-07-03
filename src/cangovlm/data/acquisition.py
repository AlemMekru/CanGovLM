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
from time import monotonic, sleep
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
CANADA_CA_SOURCE_ID = "canada_ca"
CANADA_CA_ROBOTS_URL = "https://www.canada.ca/robots.txt"
CANADA_CA_TERMS_URL = "https://www.canada.ca/en/transparency/terms.html"


class AcquisitionError(ValueError):
    """Base error for acquisition framework validation failures."""


class RobotsTxtError(AcquisitionError):
    """Raised when robots.txt does not permit acquisition."""


class TermsNotAcknowledgedError(AcquisitionError):
    """Raised when source terms have not been explicitly acknowledged."""


class TransportError(AcquisitionError):
    """Raised when an acquisition transport cannot fetch a URL."""


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


class HttpTransport(ABC):
    """Dependency-injected HTTP transport for acquisition clients."""

    @abstractmethod
    def get(self, url: str, *, headers: dict[str, str] | None = None) -> "HttpResponse":
        """Return one HTTP response."""


@dataclass(frozen=True)
class HttpResponse:
    """Raw HTTP response returned by a transport."""

    url: str
    status_code: int
    headers: dict[str, str]
    content: bytes


class UrllibHttpTransport(HttpTransport):
    """Small standard-library HTTP transport."""

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        request = Request(url, headers=headers or {})
        try:
            with urlopen(request) as response:  # noqa: S310 - explicit acquisition transport.
                response_headers = dict(response.headers.items())
                return HttpResponse(
                    url=response.geturl(),
                    status_code=response.status,
                    headers=response_headers,
                    content=response.read(),
                )
        except OSError as error:
            raise TransportError(f"Failed to fetch {url}: {error}") from error


@dataclass(frozen=True)
class RateLimitConfig:
    """Configurable rate limiting for polite acquisition."""

    min_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.min_interval_seconds < 0:
            raise AcquisitionError("min_interval_seconds must be non-negative.")


@dataclass(frozen=True)
class RetryConfig:
    """Configurable retry policy with exponential backoff."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise AcquisitionError("max_attempts must be at least 1.")
        if self.initial_backoff_seconds < 0:
            raise AcquisitionError("initial_backoff_seconds must be non-negative.")
        if self.backoff_multiplier < 1:
            raise AcquisitionError("backoff_multiplier must be at least 1.")


@dataclass(frozen=True)
class StoredRawDocument:
    """Result of acquiring and storing one immutable raw document."""

    raw_document: "RawDocument"
    raw_path: Path
    manifest_record: "DownloadManifestRecord"


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


class CanadaCaAcquisitionClient(AcquisitionClient):
    """Acquisition client for approved Canada.ca HTML pages."""

    def __init__(
        self,
        *,
        source: OfficialSource,
        corpus_root: str | Path,
        transport: HttpTransport | None = None,
        rate_limit: RateLimitConfig | None = None,
        retry: RetryConfig | None = None,
        terms_acknowledged: bool = False,
        user_agent: str = "CanGovLM/0.1 (+https://www.canada.ca/)",
        sleeper=sleep,
        clock=monotonic,
    ) -> None:
        if source.source_id != CANADA_CA_SOURCE_ID:
            raise AcquisitionError("CanadaCaAcquisitionClient requires the canada_ca source.")
        if not source.enabled:
            raise AcquisitionError("Canada.ca source must be enabled before acquisition.")
        if "html" not in source.document_formats:
            raise AcquisitionError("Canada.ca source must support html document acquisition.")

        self.source = source
        self.corpus_root = Path(corpus_root)
        self.transport = transport or UrllibHttpTransport()
        self.rate_limit = rate_limit or RateLimitConfig()
        self.retry = retry or RetryConfig()
        self.terms_acknowledged = terms_acknowledged
        self.user_agent = user_agent
        self._sleeper = sleeper
        self._clock = clock
        self._last_request_at: float | None = None
        self._robots_policy: RobotsPolicy | None = None

    def acquire(self, request: AcquisitionRequest) -> RawDocument:
        """Acquire one Canada.ca HTML page without storing it."""

        self._validate_request(request)
        self._ensure_terms_acknowledged()
        self._ensure_robots_allowed(request.url)
        response = self._get_with_retries(request.url)
        self._validate_html_response(response)

        return RawDocument(
            document_id=request.resolved_document_id(),
            source_id=request.source_id,
            url=request.url,
            canonical_url=request.canonical_url or response.url,
            language=request.language,
            document_format="html",
            content=response.content,
            retrieved_at=datetime.now(timezone.utc),
            content_type=_header_value(response.headers, "content-type"),
            status_code=response.status_code,
            metadata={
                "terms_url": self.source.license.url or CANADA_CA_TERMS_URL,
                "robots_url": CANADA_CA_ROBOTS_URL,
            },
        )

    def acquire_and_store(self, request: AcquisitionRequest) -> StoredRawDocument:
        """Acquire, immutably store, and describe one raw HTML document."""

        raw_document = self.acquire(request)
        raw_path = raw_storage_path(self.corpus_root, raw_document)
        write_raw_document(raw_path, raw_document)
        manifest_record = DownloadManifestRecord.from_raw_document(
            raw_document,
            self.source,
            raw_path=raw_path,
        )
        return StoredRawDocument(
            raw_document=raw_document,
            raw_path=raw_path,
            manifest_record=manifest_record,
        )

    def _validate_request(self, request: AcquisitionRequest) -> None:
        if request.source_id != CANADA_CA_SOURCE_ID:
            raise AcquisitionError("Canada.ca client only accepts canada_ca requests.")
        if request.document_format != "html":
            raise AcquisitionError("Canada.ca client only downloads HTML pages.")
        parsed = urlparse(request.url)
        if parsed.netloc.lower() != "www.canada.ca":
            raise AcquisitionError("Canada.ca client only accepts www.canada.ca URLs.")

    def _ensure_terms_acknowledged(self) -> None:
        if not self.terms_acknowledged:
            raise TermsNotAcknowledgedError(
                "Canada.ca terms must be acknowledged before acquisition: "
                f"{self.source.license.url or CANADA_CA_TERMS_URL}"
            )

    def _ensure_robots_allowed(self, url: str) -> None:
        policy = self._robots_policy
        if policy is None:
            response = self._get_with_retries(CANADA_CA_ROBOTS_URL)
            policy = RobotsPolicy.from_text(response.content.decode("utf-8", errors="replace"))
            self._robots_policy = policy

        if not policy.is_allowed(url):
            raise RobotsTxtError(f"robots.txt disallows acquisition of {url}")

    def _get_with_retries(self, url: str) -> HttpResponse:
        attempt = 1
        backoff = self.retry.initial_backoff_seconds
        while True:
            self._apply_rate_limit()
            try:
                response = self.transport.get(url, headers={"User-Agent": self.user_agent})
            except TransportError:
                if attempt >= self.retry.max_attempts:
                    raise
                self._sleeper(backoff)
                backoff *= self.retry.backoff_multiplier
                attempt += 1
                continue

            if response.status_code < 500 or attempt >= self.retry.max_attempts:
                return response

            self._sleeper(backoff)
            backoff *= self.retry.backoff_multiplier
            attempt += 1

    def _apply_rate_limit(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = self.rate_limit.min_interval_seconds - elapsed
            if remaining > 0:
                self._sleeper(remaining)
                now = self._clock()
        self._last_request_at = now

    def _validate_html_response(self, response: HttpResponse) -> None:
        if response.status_code != 200:
            raise TransportError(f"Canada.ca returned status {response.status_code}.")
        content_type = _header_value(response.headers, "content-type") or ""
        if "html" not in content_type.lower():
            raise TransportError(f"Canada.ca response is not HTML: {content_type!r}.")


@dataclass(frozen=True)
class RobotsPolicy:
    """Minimal robots.txt policy for User-agent: * disallow rules."""

    disallow_patterns: tuple[str, ...]

    @classmethod
    def from_text(cls, robots_text: str) -> "RobotsPolicy":
        patterns: list[str] = []
        applies_to_all = False
        tokens = _robots_tokens(robots_text)

        index = 0
        while index < len(tokens):
            token = tokens[index]
            lower = token.lower()
            if lower == "user-agent:" and index + 1 < len(tokens):
                applies_to_all = tokens[index + 1] == "*"
                index += 2
                continue
            if lower == "disallow:" and index + 1 < len(tokens):
                if applies_to_all and tokens[index + 1]:
                    patterns.append(tokens[index + 1])
                index += 2
                continue
            index += 1

        return cls(disallow_patterns=tuple(patterns))

    def is_allowed(self, url: str) -> bool:
        path = urlparse(url).path or "/"
        return not any(_robots_pattern_matches(pattern, path) for pattern in self.disallow_patterns)


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


def write_raw_document(path: str | Path, raw_document: RawDocument) -> None:
    """Write raw content immutably.

    Existing files are allowed only when their bytes already match the raw
    document. Different bytes at the same path indicate an attempted mutation.
    """

    raw_path = Path(path)
    if raw_path.exists():
        existing_sha256 = sha256_file(raw_path)
        if existing_sha256 != raw_document.sha256:
            raise AcquisitionError(f"Refusing to overwrite immutable raw file: {raw_path}")
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_document.content)


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


def _header_value(headers: dict[str, str], name: str) -> str | None:
    lower_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lower_name:
            return value
    return None


def _robots_tokens(robots_text: str) -> list[str]:
    text_without_comments = []
    for line in robots_text.splitlines():
        text_without_comments.append(line.split("#", maxsplit=1)[0])
    return " ".join(text_without_comments).split()


def _robots_pattern_matches(pattern: str, path: str) -> bool:
    escaped_parts = [re.escape(part) for part in pattern.split("*")]
    regex = "^" + ".*".join(escaped_parts)
    return re.match(regex, path) is not None
