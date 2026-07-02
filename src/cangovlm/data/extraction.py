"""Text extraction framework for CanGovLM.

This module defines extraction interfaces, result models, and validation. It
does not implement format parsing, PDF extraction, cleaning, scraping, or
network access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from cangovlm.data.acquisition import ACQUIRABLE_DOCUMENT_FORMATS, RawDocument, sha256_text

EXTRACTABLE_DOCUMENT_FORMATS = ACQUIRABLE_DOCUMENT_FORMATS


class ExtractionError(ValueError):
    """Base error for extraction framework failures."""


class ExtractionValidationError(ExtractionError):
    """Raised when one or more extraction validation checks fail."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class ExtractionMetadata:
    """Metadata produced by an extractor for one raw document."""

    extractor_name: str
    extractor_version: str
    source_format: str
    language: str
    title: str | None = None
    warnings: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionResult:
    """Plain-text extraction result for one raw document."""

    document_id: str
    source_id: str
    language: str
    text: str
    metadata: ExtractionMetadata

    @property
    def text_sha256(self) -> str:
        """SHA-256 hash of extracted UTF-8 text."""

        return sha256_text(self.text)

    @property
    def character_count(self) -> int:
        """Number of Unicode code points in extracted text."""

        return len(self.text)

    @property
    def line_count(self) -> int:
        """Number of logical lines in extracted text."""

        if not self.text:
            return 0
        return self.text.count("\n") + 1


class DocumentExtractor(ABC):
    """Abstract extractor interface for one document format."""

    document_format: str
    extractor_name: str
    extractor_version: str

    def __init__(self, *, extractor_name: str, extractor_version: str) -> None:
        self.extractor_name = extractor_name
        self.extractor_version = extractor_version

    def can_extract(self, raw_document: RawDocument) -> bool:
        """Return whether this extractor supports a raw document."""

        return raw_document.document_format == self.document_format

    def extract(self, raw_document: RawDocument) -> ExtractionResult:
        """Extract text from a raw document and validate the result."""

        if not self.can_extract(raw_document):
            raise ExtractionError(
                f"{self.extractor_name} cannot extract format "
                f"{raw_document.document_format!r}."
            )

        result = self._extract(raw_document)
        validate_extraction_result(result, raw_document=raw_document)
        return result

    @abstractmethod
    def _extract(self, raw_document: RawDocument) -> ExtractionResult:
        """Format-specific extraction implementation."""


class HtmlExtractor(DocumentExtractor):
    """Interface for future HTML text extractors."""

    document_format = "html"


class TxtExtractor(DocumentExtractor):
    """Interface for future TXT text extractors."""

    document_format = "txt"


class XmlExtractor(DocumentExtractor):
    """Interface for future XML text extractors."""

    document_format = "xml"


class JsonExtractor(DocumentExtractor):
    """Interface for future JSON text extractors."""

    document_format = "json"


class CsvExtractor(DocumentExtractor):
    """Interface for future CSV text extractors."""

    document_format = "csv"


class PdfExtractor(DocumentExtractor):
    """Placeholder interface for future PDF text extractors."""

    document_format = "pdf"


def validate_extraction_result(
    result: ExtractionResult,
    *,
    raw_document: RawDocument | None = None,
) -> None:
    """Validate an extraction result against framework requirements."""

    errors: list[str] = []
    prefix = result.document_id or "<missing document_id>"

    _require_text(errors, prefix, "document_id", result.document_id)
    _require_text(errors, prefix, "source_id", result.source_id)
    if result.language not in {"en", "fr"}:
        errors.append(f"{prefix}: language must be 'en' or 'fr'.")
    if not result.text:
        errors.append(f"{prefix}: extracted text must not be empty.")
    if result.metadata.source_format not in EXTRACTABLE_DOCUMENT_FORMATS:
        errors.append(
            f"{prefix}: unsupported source_format {result.metadata.source_format!r}."
        )
    _require_text(errors, prefix, "metadata.extractor_name", result.metadata.extractor_name)
    _require_text(errors, prefix, "metadata.extractor_version", result.metadata.extractor_version)
    if result.metadata.language != result.language:
        errors.append(f"{prefix}: metadata language must match result language.")

    if raw_document is not None:
        if result.document_id != raw_document.document_id:
            errors.append(f"{prefix}: result document_id must match raw document.")
        if result.source_id != raw_document.source_id:
            errors.append(f"{prefix}: result source_id must match raw document.")
        if result.language != raw_document.language:
            errors.append(f"{prefix}: result language must match raw document.")
        if result.metadata.source_format != raw_document.document_format:
            errors.append(f"{prefix}: metadata source_format must match raw document format.")

    if errors:
        raise ExtractionValidationError(errors)


def extracted_storage_path(raw_path: str, *, extracted_root: str = "corpus/extracted") -> str:
    """Generate the future extracted text path corresponding to a raw path.

    This is path mapping only. It does not read, parse, or write document text.
    """

    parts = raw_path.split("/")
    if len(parts) < 5 or "raw" not in parts:
        raise ExtractionError(f"raw_path does not match expected corpus raw layout: {raw_path}")

    raw_index = parts.index("raw")
    language = parts[raw_index + 1]
    source_id = parts[raw_index + 2]
    filename = parts[-1].rsplit(".", maxsplit=1)[0] + ".txt"
    return "/".join([extracted_root, language, source_id, filename])


def _require_text(errors: list[str], prefix: str, field_name: str, value: str) -> None:
    if not str(value).strip():
        errors.append(f"{prefix}: {field_name} is required.")

