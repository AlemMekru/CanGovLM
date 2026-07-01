import hashlib
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.data import (
    ACQUIRABLE_DOCUMENT_FORMATS,
    AcquisitionError,
    AcquisitionRequest,
    DownloadManifestRecord,
    OfficialSource,
    RawDocument,
    SourceLicense,
    extension_for_format,
    raw_storage_path,
    request_storage_path,
    sha256_bytes,
    sha256_file,
    sha256_text,
    stable_document_id,
)


class AcquisitionFrameworkTests(TestCase):
    def test_acquirable_formats_cover_required_future_formats(self) -> None:
        self.assertEqual(ACQUIRABLE_DOCUMENT_FORMATS, {"html", "pdf", "xml", "json", "txt", "csv"})

    def test_stable_document_id_is_deterministic_and_named_by_source_and_language(self) -> None:
        first = stable_document_id(
            source_id="canada_ca",
            language="en",
            stable_identity="https://www.canada.ca/example",
        )
        second = stable_document_id(
            source_id="canada_ca",
            language="en",
            stable_identity="https://www.canada.ca/example",
        )

        self.assertEqual(first, second)
        self.assertRegex(first, r"^canada_ca_en_[a-f0-9]{12}$")

    def test_hashing_utilities_hash_bytes_text_and_files(self) -> None:
        content = b"official raw bytes"
        expected = hashlib.sha256(content).hexdigest()

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.bin"
            path.write_bytes(content)

            self.assertEqual(sha256_bytes(content), expected)
            self.assertEqual(sha256_text("official raw bytes"), expected)
            self.assertEqual(sha256_file(path), expected)

    def test_extension_for_format_supports_required_formats(self) -> None:
        self.assertEqual(extension_for_format("html"), ".html")
        self.assertEqual(extension_for_format("pdf"), ".pdf")
        self.assertEqual(extension_for_format("xml"), ".xml")
        self.assertEqual(extension_for_format("json"), ".json")
        self.assertEqual(extension_for_format("txt"), ".txt")
        self.assertEqual(extension_for_format("csv"), ".csv")

    def test_acquisition_request_derives_document_id_and_storage_path(self) -> None:
        request = AcquisitionRequest(
            source_id="canada_ca",
            url="https://www.canada.ca/example",
            canonical_url="https://www.canada.ca/example",
            language="en",
            document_format="html",
        )

        path = request_storage_path("corpus", request)

        self.assertRegex(request.resolved_document_id(), r"^canada_ca_en_[a-f0-9]{12}$")
        self.assertEqual(path.parent, Path("corpus/raw/en/canada_ca"))
        self.assertEqual(path.suffix, ".html")

    def test_raw_document_is_immutable_bytes_plus_metadata(self) -> None:
        raw_document = _raw_document()

        self.assertEqual(raw_document.sha256, sha256_bytes(b"<html>hello</html>"))
        self.assertEqual(raw_document.byte_size, len(b"<html>hello</html>"))

        with self.assertRaises(Exception):
            raw_document.document_id = "changed"  # type: ignore[misc]

    def test_raw_storage_path_uses_language_source_id_document_id_and_extension(self) -> None:
        raw_document = _raw_document()

        self.assertEqual(
            raw_storage_path("corpus", raw_document),
            Path(f"corpus/raw/en/canada_ca/{raw_document.document_id}.html"),
        )

    def test_manifest_record_preserves_source_and_raw_document_metadata(self) -> None:
        source = _source()
        raw_document = _raw_document()
        raw_path = raw_storage_path("corpus", raw_document)

        record = DownloadManifestRecord.from_raw_document(raw_document, source, raw_path=raw_path)
        record_dict = record.to_dict()

        self.assertEqual(record.document_id, raw_document.document_id)
        self.assertEqual(record.source_name, "Canada.ca")
        self.assertEqual(record.organization, "Government of Canada")
        self.assertEqual(record.raw_sha256, raw_document.sha256)
        self.assertEqual(record.byte_size, raw_document.byte_size)
        self.assertEqual(record.license_name, source.license.name)
        self.assertEqual(record_dict["raw_path"], str(raw_path))

    def test_manifest_record_rejects_mismatched_source(self) -> None:
        source = _source(source_id="justice_laws")

        with self.assertRaises(AcquisitionError):
            DownloadManifestRecord.from_raw_document(
                _raw_document(source_id="canada_ca"),
                source,
                raw_path="corpus/raw/en/canada_ca/doc.html",
            )

    def test_invalid_request_metadata_is_rejected(self) -> None:
        with self.assertRaises(AcquisitionError):
            AcquisitionRequest(
                source_id="canada_ca",
                url="not-a-url",
                language="en",
                document_format="html",
            )

        with self.assertRaises(AcquisitionError):
            AcquisitionRequest(
                source_id="canada_ca",
                url="https://www.canada.ca/example",
                language="de",
                document_format="html",
            )

        with self.assertRaises(AcquisitionError):
            AcquisitionRequest(
                source_id="canada_ca",
                url="https://www.canada.ca/example",
                language="en",
                document_format="docx",
            )

    def test_empty_raw_document_content_is_rejected(self) -> None:
        with self.assertRaises(AcquisitionError):
            _raw_document(content=b"")


def _source(*, source_id: str = "canada_ca") -> OfficialSource:
    return OfficialSource(
        source_id=source_id,
        name="Canada.ca",
        organization="Government of Canada",
        base_url="https://www.canada.ca/",
        languages=("en", "fr"),
        license=SourceLicense(
            name="Government of Canada site terms",
            url="https://www.canada.ca/en/transparency/terms.html",
        ),
        document_formats=("html",),
        collection_method="sitemap",
        enabled=True,
        notes="Official federal web content.",
    )


def _raw_document(
    *,
    source_id: str = "canada_ca",
    content: bytes = b"<html>hello</html>",
) -> RawDocument:
    document_id = stable_document_id(
        source_id=source_id,
        language="en",
        stable_identity="https://www.canada.ca/example",
    )
    return RawDocument(
        document_id=document_id,
        source_id=source_id,
        url="https://www.canada.ca/example",
        canonical_url="https://www.canada.ca/example",
        language="en",
        document_format="html",
        content=content,
        retrieved_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        content_type="text/html",
        status_code=200,
    )
