from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.data import (
    AcquisitionRequest,
    CanadaCaAcquisitionClient,
    HttpResponse,
    HttpTransport,
    OfficialSource,
    RateLimitConfig,
    RawDocument,
    RobotsPolicy,
    RobotsTxtError,
    SourceLicense,
    TermsNotAcknowledgedError,
    TransportError,
    RetryConfig,
    load_source_registry,
    raw_storage_path,
    sha256_bytes,
    stable_document_id,
    write_raw_document,
)


class CanadaCaAcquisitionClientTests(TestCase):
    def test_acquire_and_store_downloads_html_and_creates_manifest(self) -> None:
        html = b"<!doctype html><html><body><p>Hello Canada</p></body></html>"
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                "https://www.canada.ca/en/example.html": [
                    _response("https://www.canada.ca/en/example.html", html)
                ],
            }
        )

        with TemporaryDirectory() as temp_dir:
            client = _client(temp_dir, transport=transport)
            request = _request("https://www.canada.ca/en/example.html")

            stored = client.acquire_and_store(request)

            self.assertEqual(stored.raw_document.content, html)
            self.assertEqual(stored.raw_document.sha256, sha256_bytes(html))
            self.assertEqual(stored.raw_path, raw_storage_path(temp_dir, stored.raw_document))
            self.assertEqual(stored.raw_path.read_bytes(), html)
            self.assertEqual(stored.manifest_record.raw_sha256, sha256_bytes(html))
            self.assertEqual(stored.manifest_record.raw_path, str(stored.raw_path))
            self.assertEqual(stored.manifest_record.source_id, "canada_ca")
            self.assertEqual(stored.manifest_record.document_format, "html")

    def test_client_requires_terms_acknowledgement(self) -> None:
        with self.assertRaises(TermsNotAcknowledgedError):
            _client("/tmp", terms_acknowledged=False).acquire(
                _request("https://www.canada.ca/en/example.html")
            )

    def test_client_rejects_non_canada_ca_urls(self) -> None:
        outside_request = AcquisitionRequest(
            source_id="canada_ca",
            url="https://example.com/page.html",
            language="en",
            document_format="html",
        )
        with self.assertRaises(Exception):
            _client("/tmp").acquire(outside_request)

        pdf_request = AcquisitionRequest(
            source_id="canada_ca",
            url="https://www.canada.ca/en/example.html",
            language="en",
            document_format="pdf",
        )

        with self.assertRaises(Exception):
            _client("/tmp").acquire(pdf_request)

    def test_robots_policy_disallows_blocked_paths(self) -> None:
        policy = RobotsPolicy.from_text(
            "User-agent: *\n"
            "Disallow: /en/sr/srb.html\n"
            "Disallow: /en/*/menu.html\n"
        )

        self.assertFalse(policy.is_allowed("https://www.canada.ca/en/sr/srb.html"))
        self.assertFalse(policy.is_allowed("https://www.canada.ca/en/topic/menu.html"))
        self.assertTrue(policy.is_allowed("https://www.canada.ca/en/example.html"))

    def test_client_raises_when_robots_disallows_url(self) -> None:
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response(
                        "https://www.canada.ca/robots.txt",
                        b"User-agent: *\nDisallow: /en/blocked.html\n",
                    )
                ],
            }
        )

        with self.assertRaises(RobotsTxtError):
            _client("/tmp", transport=transport).acquire(
                _request("https://www.canada.ca/en/blocked.html")
            )

    def test_client_retries_5xx_with_exponential_backoff(self) -> None:
        sleeper = _FakeSleeper()
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                "https://www.canada.ca/en/retry.html": [
                    _response("https://www.canada.ca/en/retry.html", b"", status_code=500),
                    _response("https://www.canada.ca/en/retry.html", b"<html>ok</html>"),
                ],
            }
        )

        client = _client(
            "/tmp",
            transport=transport,
            retry=RetryConfig(max_attempts=2, initial_backoff_seconds=0.5),
            rate_limit=RateLimitConfig(min_interval_seconds=0),
            sleeper=sleeper,
        )

        raw_document = client.acquire(_request("https://www.canada.ca/en/retry.html"))

        self.assertEqual(raw_document.content, b"<html>ok</html>")
        self.assertEqual(sleeper.calls, [0.5])

    def test_client_applies_configurable_rate_limit(self) -> None:
        sleeper = _FakeSleeper()
        clock = _FakeClock()
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                "https://www.canada.ca/en/rate.html": [
                    _response("https://www.canada.ca/en/rate.html", b"<html>ok</html>")
                ],
            }
        )

        client = _client(
            "/tmp",
            transport=transport,
            rate_limit=RateLimitConfig(min_interval_seconds=2.0),
            sleeper=sleeper.sleep,
            clock=clock.now,
        )

        client.acquire(_request("https://www.canada.ca/en/rate.html"))

        self.assertEqual(sleeper.calls, [2.0])

    def test_client_rejects_non_html_response(self) -> None:
        transport = _FakeTransport(
            {
                "https://www.canada.ca/robots.txt": [
                    _response("https://www.canada.ca/robots.txt", b"User-agent: *\n")
                ],
                "https://www.canada.ca/en/data.json": [
                    _response(
                        "https://www.canada.ca/en/data.json",
                        b"{}",
                        headers={"Content-Type": "application/json"},
                    )
                ],
            }
        )

        with self.assertRaises(TransportError):
            _client("/tmp", transport=transport).acquire(
                _request("https://www.canada.ca/en/data.json")
            )

    def test_write_raw_document_refuses_to_mutate_existing_file(self) -> None:
        raw_document = _raw_document(b"<html>first</html>")
        changed_document = _raw_document(b"<html>second</html>")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.html"
            write_raw_document(path, raw_document)
            write_raw_document(path, raw_document)

            with self.assertRaises(Exception):
                write_raw_document(path, changed_document)

    def test_repository_canada_ca_source_is_valid_for_client(self) -> None:
        registry = load_source_registry("configs/data/source_registry.json")
        source = registry.get("canada_ca")

        client = CanadaCaAcquisitionClient(
            source=source,
            corpus_root="/tmp",
            transport=_FakeTransport({}),
            terms_acknowledged=True,
        )

        self.assertEqual(client.source.source_id, "canada_ca")


class _FakeTransport(HttpTransport):
    def __init__(self, responses: dict[str, list[HttpResponse]]) -> None:
        self.responses = {url: list(values) for url, values in responses.items()}
        self.requested_urls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        self.requested_urls.append(url)
        if url not in self.responses or not self.responses[url]:
            raise TransportError(f"No fake response configured for {url}")
        return self.responses[url].pop(0)


class _FakeSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)


class _FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def now(self) -> float:
        return self.current


def _client(
    corpus_root: str,
    *,
    transport: HttpTransport | None = None,
    rate_limit: RateLimitConfig | None = None,
    retry: RetryConfig | None = None,
    terms_acknowledged: bool = True,
    sleeper=None,
    clock=None,
) -> CanadaCaAcquisitionClient:
    return CanadaCaAcquisitionClient(
        source=_source(),
        corpus_root=corpus_root,
        transport=transport or _FakeTransport({}),
        rate_limit=rate_limit or RateLimitConfig(min_interval_seconds=0),
        retry=retry or RetryConfig(max_attempts=1),
        terms_acknowledged=terms_acknowledged,
        sleeper=sleeper or (lambda seconds: None),
        clock=clock or (lambda: 0.0),
    )


def _request(url: str) -> AcquisitionRequest:
    return AcquisitionRequest(
        source_id="canada_ca",
        url=url,
        canonical_url=url,
        language="en",
        document_format="html",
    )


def _response(
    url: str,
    content: bytes,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        url=url,
        status_code=status_code,
        headers=headers or {"Content-Type": "text/html; charset=utf-8"},
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


def _raw_document(content: bytes) -> RawDocument:
    url = "https://www.canada.ca/en/example.html"
    return RawDocument(
        document_id=stable_document_id(
            source_id="canada_ca",
            language="en",
            stable_identity=url,
        ),
        source_id="canada_ca",
        url=url,
        canonical_url=url,
        language="en",
        document_format="html",
        content=content,
        retrieved_at=datetime(2026, 7, 2, tzinfo=timezone.utc),
        content_type="text/html",
        status_code=200,
    )
