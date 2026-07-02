from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase

from cangovlm.data import (
    ExtractionError,
    ExtractionValidationError,
    HtmlExtractor,
    RawDocument,
    stable_document_id,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "html"


class HtmlExtractorTests(TestCase):
    def test_html_extractor_preserves_title_headings_paragraphs_and_lists(self) -> None:
        raw_document = _html_raw_document("official_page.html")

        result = HtmlExtractor().extract(raw_document)

        self.assertEqual(result.metadata.title, "Official Program Update")
        self.assertEqual(
            result.text,
            "\n\n".join(
                [
                    "Official Program Update",
                    "Official Program Update",
                    "The Government of Canada announced an update.",
                    "Eligibility",
                    "- Applicants must live in Canada.",
                    "- Applicants must provide supporting documents.",
                    "Steps",
                    "1. Read the official guidance.",
                    "2. Submit the application.",
                    "Contact the department for more information.",
                ]
            ),
        )

    def test_html_extractor_removes_obvious_page_chrome(self) -> None:
        result = HtmlExtractor().extract(_html_raw_document("official_page.html"))

        self.assertNotIn("Government navigation header", result.text)
        self.assertNotIn("Site menu", result.text)
        self.assertNotIn("We use cookies", result.text)
        self.assertNotIn("Footer links", result.text)
        self.assertGreaterEqual(int(result.metadata.extra["skipped_chrome_blocks"]), 4)

    def test_html_extractor_preserves_utf8_french_text(self) -> None:
        raw_document = _html_raw_document("french_page.html", language="fr")

        result = HtmlExtractor().extract(raw_document)

        self.assertIn("Santé Canada annonce une mise à jour", result.text)
        self.assertIn("Québec", result.text)
        self.assertEqual(result.metadata.language, "fr")
        self.assertEqual(result.metadata.source_format, "html")

    def test_html_extractor_populates_metadata(self) -> None:
        result = HtmlExtractor().extract(_html_raw_document("official_page.html"))

        self.assertEqual(result.metadata.extractor_name, "html_structural_extractor")
        self.assertEqual(result.metadata.extractor_version, "0.1.0")
        self.assertEqual(result.metadata.source_format, "html")
        self.assertEqual(result.metadata.warnings, ())
        self.assertEqual(result.metadata.extra["text_blocks"], "10")

    def test_html_extractor_rejects_pages_without_structural_text_blocks(self) -> None:
        with self.assertRaises(ExtractionValidationError):
            HtmlExtractor().extract(_html_raw_document("chrome_only.html"))

    def test_html_extractor_rejects_non_html_raw_documents(self) -> None:
        raw_document = _html_raw_document("official_page.html", document_format="txt")

        with self.assertRaises(ExtractionError):
            HtmlExtractor().extract(raw_document)


def _html_raw_document(
    fixture_name: str,
    *,
    language: str = "en",
    document_format: str = "html",
) -> RawDocument:
    content = (FIXTURE_DIR / fixture_name).read_bytes()
    document_id = stable_document_id(
        source_id="canada_ca",
        language=language,
        stable_identity=f"https://www.canada.ca/{fixture_name}",
    )
    return RawDocument(
        document_id=document_id,
        source_id="canada_ca",
        url=f"https://www.canada.ca/{fixture_name}",
        canonical_url=f"https://www.canada.ca/{fixture_name}",
        language=language,
        document_format=document_format,
        content=content,
        retrieved_at=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content_type="text/html",
        status_code=200,
    )
