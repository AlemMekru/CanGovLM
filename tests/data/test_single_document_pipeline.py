from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.data import (
    AcquisitionRequest,
    CanadaCaAcquisitionClient,
    CanadaCaSingleDocumentPipeline,
    HttpResponse,
    HttpTransport,
    OfficialSource,
    RateLimitConfig,
    RetryConfig,
    SourceLicense,
    TransportError,
    load_manifest,
    sha256_bytes,
    write_manifest,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "html"


class SingleDocumentPipelineTests(TestCase):
    def test_pipeline_acquires_stores_extracts_and_validates_one_canada_ca_document(self) -> None:
        html = (FIXTURE_DIR / "official_page.html").read_bytes()
        url = "https://www.canada.ca/en/example.html"
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                url: [_response(url, html)],
            }
        )

        with TemporaryDirectory() as temp_dir:
            corpus_root = Path(temp_dir) / "corpus"
            pipeline = CanadaCaSingleDocumentPipeline(
                acquisition_client=_client(corpus_root, transport=transport)
            )

            result = pipeline.run(_request(url))

            self.assertTrue(result.stored_raw_document.raw_path.exists())
            self.assertTrue(result.extracted_path.exists())
            self.assertEqual(result.stored_raw_document.raw_path.read_bytes(), html)
            self.assertIn("Official Program Update", result.extracted_path.read_text(encoding="utf-8"))
            self.assertEqual(result.raw_manifest_record.raw_sha256, sha256_bytes(html))
            self.assertEqual(result.extracted_manifest_record.manifest_stage, "extracted")
            self.assertEqual(
                result.extracted_manifest_record.extracted_sha256,
                result.extraction_result.text_sha256,
            )
            self.assertEqual(
                result.extracted_manifest_record.raw_path,
                str(result.stored_raw_document.raw_path),
            )
            self.assertEqual(
                result.extracted_manifest_record.extracted_path,
                str(result.extracted_path),
            )
            self.assertEqual(
                result.extracted_path.parent,
                corpus_root / "extracted" / "en" / "canada_ca",
            )

    def test_pipeline_extracted_manifest_can_be_written_and_loaded_as_jsonl(self) -> None:
        html = (FIXTURE_DIR / "official_page.html").read_bytes()
        url = "https://www.canada.ca/en/manifest.html"
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                url: [_response(url, html)],
            }
        )

        with TemporaryDirectory() as temp_dir:
            pipeline = CanadaCaSingleDocumentPipeline(
                acquisition_client=_client(Path(temp_dir) / "corpus", transport=transport)
            )
            result = pipeline.run(_request(url))
            manifest_path = Path(temp_dir) / "corpus" / "manifests" / "extracted_manifest.jsonl"

            write_manifest(manifest_path, [result.extracted_manifest_record])
            loaded = load_manifest(manifest_path)

            self.assertEqual(loaded.records, (result.extracted_manifest_record,))

    def test_pipeline_refuses_to_mutate_existing_outputs(self) -> None:
        html = (FIXTURE_DIR / "official_page.html").read_bytes()
        url = "https://www.canada.ca/en/immutable.html"
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                url: [
                    _response(url, html),
                    _response(url, html.replace(b"Official Program Update", b"Changed Title")),
                ],
            }
        )

        with TemporaryDirectory() as temp_dir:
            pipeline = CanadaCaSingleDocumentPipeline(
                acquisition_client=_client(Path(temp_dir) / "corpus", transport=transport)
            )
            pipeline.run(_request(url))

            with self.assertRaises(Exception):
                pipeline.run(_request(url))


class _FakeTransport(HttpTransport):
    def __init__(self, responses: dict[str, list[HttpResponse]]) -> None:
        self.responses = {url: list(values) for url, values in responses.items()}

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        if url not in self.responses or not self.responses[url]:
            raise TransportError(f"No fake response configured for {url}")
        return self.responses[url].pop(0)


def _client(corpus_root: str | Path, *, transport: HttpTransport) -> CanadaCaAcquisitionClient:
    return CanadaCaAcquisitionClient(
        source=_source(),
        corpus_root=corpus_root,
        transport=transport,
        rate_limit=RateLimitConfig(min_interval_seconds=0),
        retry=RetryConfig(max_attempts=1),
        terms_acknowledged=True,
        sleeper=lambda seconds: None,
        clock=lambda: 0.0,
    )


def _request(url: str) -> AcquisitionRequest:
    return AcquisitionRequest(
        source_id="canada_ca",
        url=url,
        canonical_url=url,
        language="en",
        document_format="html",
    )


def _response(url: str, content: bytes) -> HttpResponse:
    return HttpResponse(
        url=url,
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=content,
    )


def _source() -> OfficialSource:
    return OfficialSource(
        source_id="canada_ca",
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
        collection_date="2026-07-02",
        version_or_snapshot="initial_source_registry_v20260702",
        notes="Official federal web content.",
    )
