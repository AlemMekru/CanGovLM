from __future__ import annotations

from datetime import datetime, timezone
from unittest import TestCase

from cangovlm.data import (
    CsvExtractor,
    ExtractionError,
    ExtractionMetadata,
    ExtractionResult,
    ExtractionValidationError,
    HtmlExtractor,
    JsonExtractor,
    PdfExtractor,
    RawDocument,
    TxtExtractor,
    XmlExtractor,
    extracted_storage_path,
    sha256_text,
    stable_document_id,
    validate_extraction_result,
)


class ExtractionFrameworkTests(TestCase):
    def test_format_specific_extractors_expose_expected_formats(self) -> None:
        self.assertEqual(HtmlExtractor.document_format, "html")
        self.assertEqual(TxtExtractor.document_format, "txt")
        self.assertEqual(XmlExtractor.document_format, "xml")
        self.assertEqual(JsonExtractor.document_format, "json")
        self.assertEqual(CsvExtractor.document_format, "csv")
        self.assertEqual(PdfExtractor.document_format, "pdf")

    def test_txt_extractor_subclass_can_return_valid_extraction_result(self) -> None:
        raw_document = _raw_document(document_format="txt", content=b"Hello Canada\n")
        extractor = _SyntheticTxtExtractor()

        result = extractor.extract(raw_document)

        self.assertEqual(result.text, "Hello Canada\n")
        self.assertEqual(result.text_sha256, sha256_text("Hello Canada\n"))
        self.assertEqual(result.character_count, len("Hello Canada\n"))
        self.assertEqual(result.line_count, 2)
        self.assertEqual(result.metadata.source_format, "txt")

    def test_extractor_rejects_unsupported_raw_format(self) -> None:
        extractor = _SyntheticTxtExtractor()

        with self.assertRaises(ExtractionError):
            extractor.extract(_raw_document(document_format="html", content=b"<p>Hello</p>"))

    def test_extraction_validation_rejects_empty_text(self) -> None:
        result = _result(text="")

        with self.assertRaises(ExtractionValidationError) as context:
            validate_extraction_result(result)

        self.assertIn("extracted text must not be empty", str(context.exception))

    def test_extraction_validation_rejects_mismatched_raw_document_metadata(self) -> None:
        raw_document = _raw_document(document_format="txt")
        result = _result(document_id="canada_ca_en_aaaaaaaaaaaa")

        with self.assertRaises(ExtractionValidationError) as context:
            validate_extraction_result(result, raw_document=raw_document)

        self.assertIn("result document_id must match raw document", str(context.exception))

    def test_extraction_validation_requires_metadata_language_to_match(self) -> None:
        result = _result(metadata=_metadata(language="fr"))

        with self.assertRaises(ExtractionValidationError) as context:
            validate_extraction_result(result)

        self.assertIn("metadata language must match result language", str(context.exception))

    def test_extracted_storage_path_maps_raw_layout_to_extracted_txt_layout(self) -> None:
        raw_path = "corpus/raw/en/canada_ca/canada_ca_en_2f4c8a9b0000.html"

        self.assertEqual(
            extracted_storage_path(raw_path),
            "corpus/extracted/en/canada_ca/canada_ca_en_2f4c8a9b0000.txt",
        )

    def test_extracted_storage_path_rejects_non_raw_layout(self) -> None:
        with self.assertRaises(ExtractionError):
            extracted_storage_path("corpus/other/file.html")

    def test_pdf_extractor_is_placeholder_interface_only(self) -> None:
        class SyntheticPdfExtractor(PdfExtractor):
            def _extract(self, raw_document: RawDocument) -> ExtractionResult:
                return _result(
                    document_id=raw_document.document_id,
                    source_id=raw_document.source_id,
                    language=raw_document.language,
                    metadata=_metadata(source_format="pdf"),
                )

        extractor = SyntheticPdfExtractor(
            extractor_name="synthetic_pdf_placeholder",
            extractor_version="0.1",
        )
        raw_document = _raw_document(document_format="pdf", content=b"%PDF synthetic")

        self.assertTrue(extractor.can_extract(raw_document))


class _SyntheticTxtExtractor(TxtExtractor):
    def __init__(self) -> None:
        super().__init__(extractor_name="synthetic_txt", extractor_version="0.1")

    def _extract(self, raw_document: RawDocument) -> ExtractionResult:
        return ExtractionResult(
            document_id=raw_document.document_id,
            source_id=raw_document.source_id,
            language=raw_document.language,
            text=raw_document.content.decode("utf-8"),
            metadata=ExtractionMetadata(
                extractor_name=self.extractor_name,
                extractor_version=self.extractor_version,
                source_format=raw_document.document_format,
                language=raw_document.language,
            ),
        )


def _raw_document(
    *,
    document_id: str | None = None,
    source_id: str = "canada_ca",
    language: str = "en",
    document_format: str = "txt",
    content: bytes = b"Hello Canada",
) -> RawDocument:
    resolved_document_id = document_id or stable_document_id(
        source_id=source_id,
        language=language,
        stable_identity="https://www.canada.ca/example",
    )
    return RawDocument(
        document_id=resolved_document_id,
        source_id=source_id,
        url="https://www.canada.ca/example",
        canonical_url="https://www.canada.ca/example",
        language=language,
        document_format=document_format,
        content=content,
        retrieved_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content_type=f"text/{document_format}",
        status_code=200,
    )


def _metadata(
    *,
    source_format: str = "txt",
    language: str = "en",
) -> ExtractionMetadata:
    return ExtractionMetadata(
        extractor_name="synthetic_txt",
        extractor_version="0.1",
        source_format=source_format,
        language=language,
    )


def _result(
    *,
    document_id: str = "canada_ca_en_2f4c8a9b0000",
    source_id: str = "canada_ca",
    language: str = "en",
    text: str = "Hello Canada",
    metadata: ExtractionMetadata | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id,
        source_id=source_id,
        language=language,
        text=text,
        metadata=metadata or _metadata(language=language),
    )
