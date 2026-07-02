"""Text extraction framework for CanGovLM.

This module defines extraction interfaces, result models, and validation. It
does not implement format parsing, PDF extraction, cleaning, scraping, or
network access.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser

from cangovlm.data.acquisition import ACQUIRABLE_DOCUMENT_FORMATS, RawDocument, sha256_text

EXTRACTABLE_DOCUMENT_FORMATS = ACQUIRABLE_DOCUMENT_FORMATS
HTML_EXTRACTOR_VERSION = "0.1.0"
HTML_CONTENT_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"})
HTML_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "form", "nav", "header", "footer"})
HTML_CHROME_HINTS = frozenset(
    {
        "banner",
        "breadcrumb",
        "cookie",
        "footer",
        "header",
        "menu",
        "navigation",
        "nav",
        "search",
        "skip",
        "toolbar",
    }
)


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
    """HTML extractor for title, headings, paragraphs, and lists."""

    document_format = "html"

    def __init__(
        self,
        *,
        extractor_name: str = "html_structural_extractor",
        extractor_version: str = HTML_EXTRACTOR_VERSION,
    ) -> None:
        super().__init__(extractor_name=extractor_name, extractor_version=extractor_version)

    def _extract(self, raw_document: RawDocument) -> ExtractionResult:
        html = raw_document.content.decode("utf-8")
        parser = _StructuredHtmlTextParser()
        parser.feed(html)
        parser.close()

        text_blocks = parser.text_blocks()
        text = "\n\n".join(text_blocks)

        return ExtractionResult(
            document_id=raw_document.document_id,
            source_id=raw_document.source_id,
            language=raw_document.language,
            text=text,
            metadata=ExtractionMetadata(
                extractor_name=self.extractor_name,
                extractor_version=self.extractor_version,
                source_format=raw_document.document_format,
                language=raw_document.language,
                title=parser.title,
                warnings=parser.warnings(),
                extra={
                    "text_blocks": str(len(text_blocks)),
                    "skipped_chrome_blocks": str(parser.skipped_chrome_blocks),
                },
            ),
        )


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


class _StructuredHtmlTextParser(HTMLParser):
    """Small structural HTML-to-text parser for official document pages."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.skipped_chrome_blocks = 0
        self._blocks: list[str] = []
        self._current_tag: str | None = None
        self._current_parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._list_stack: list[dict[str, int | str]] = []
        self._current_list_prefix = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth += 1
            return

        if self._should_skip(normalized_tag, attrs):
            self._skip_depth = 1
            self.skipped_chrome_blocks += 1
            return

        if normalized_tag == "title":
            self._in_title = True
            self._title_parts = []
            return

        if normalized_tag in {"ol", "ul"}:
            self._list_stack.append({"tag": normalized_tag, "counter": 1})
            return

        if normalized_tag in HTML_CONTENT_TAGS:
            self._start_block(normalized_tag)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if normalized_tag == "title":
            self._in_title = False
            title = _normalize_whitespace(" ".join(self._title_parts))
            self.title = title or None
            return

        if normalized_tag in {"ol", "ul"}:
            if self._list_stack:
                self._list_stack.pop()
            return

        if normalized_tag == self._current_tag:
            self._finish_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._current_tag is not None:
            self._current_parts.append(data)

    def text_blocks(self) -> tuple[str, ...]:
        if not self._blocks:
            return ()

        blocks: list[str] = []
        if self.title:
            blocks.append(self.title)
        blocks.extend(self._blocks)
        return tuple(blocks)

    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.title is None:
            warnings.append("missing_title")
        if not self._blocks:
            warnings.append("no_structural_text_blocks")
        return tuple(warnings)

    def _start_block(self, tag: str) -> None:
        if self._current_tag is not None:
            self._finish_block()
        self._current_tag = tag
        self._current_parts = []
        self._current_list_prefix = self._list_prefix() if tag == "li" else ""

    def _finish_block(self) -> None:
        text = _normalize_whitespace(" ".join(self._current_parts))
        if text:
            self._blocks.append(f"{self._current_list_prefix}{text}")
        if self._current_tag == "li":
            self._increment_ordered_list_counter()
        self._current_tag = None
        self._current_parts = []
        self._current_list_prefix = ""

    def _list_prefix(self) -> str:
        if not self._list_stack:
            return "- "
        current_list = self._list_stack[-1]
        if current_list["tag"] == "ol":
            return f"{current_list['counter']}. "
        return "- "

    def _increment_ordered_list_counter(self) -> None:
        if self._list_stack and self._list_stack[-1]["tag"] == "ol":
            counter = int(self._list_stack[-1]["counter"])
            self._list_stack[-1]["counter"] = counter + 1

    def _should_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in HTML_SKIP_TAGS:
            return True

        attr_text = " ".join(
            value.lower()
            for name, value in attrs
            if value is not None and name.lower() in {"class", "id", "role", "aria-label"}
        )
        if not attr_text:
            return False

        attr_tokens = set(re.split(r"[^a-z0-9]+", attr_text))
        return bool(attr_tokens & HTML_CHROME_HINTS)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
