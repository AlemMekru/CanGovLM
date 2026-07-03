import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.data import (
    OfficialSource,
    SourceLicense,
    SourceRegistry,
    SourceValidationError,
    load_source_registry,
    validate_sources,
)


class SourceRegistryTests(TestCase):
    def test_source_registry_loads_from_json_configuration(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sources.json"
            path.write_text(
                json.dumps({"sources": [_source_dict(source_id="canada_ca")]}),
                encoding="utf-8",
            )

            registry = load_source_registry(path)

            self.assertEqual(registry.get("canada_ca").name, "Canada.ca")
            self.assertEqual(registry.get("canada_ca").collection_date, "2026-07-02")
            self.assertEqual(len(registry.enabled_sources()), 1)

    def test_registry_filters_enabled_sources_by_language(self) -> None:
        registry = SourceRegistry(
            sources=(
                _source(source_id="english_source", languages=("en",), enabled=True),
                _source(source_id="french_source", languages=("fr",), enabled=True),
                _source(source_id="disabled_source", languages=("en",), enabled=False),
                _source(source_id="bilingual_source", languages=("bilingual",), enabled=True),
            )
        )

        self.assertEqual(
            [source.source_id for source in registry.for_language("en")],
            ["english_source", "bilingual_source"],
        )
        self.assertEqual(
            [source.source_id for source in registry.for_language("fr")],
            ["french_source", "bilingual_source"],
        )

    def test_registry_can_include_disabled_sources_for_language_review(self) -> None:
        registry = SourceRegistry(
            sources=(
                _source(source_id="enabled_source", languages=("en",), enabled=True),
                _source(source_id="disabled_source", languages=("en",), enabled=False),
            )
        )

        self.assertEqual(
            [source.source_id for source in registry.for_language("en", enabled_only=False)],
            ["enabled_source", "disabled_source"],
        )

    def test_registry_rejects_duplicate_source_ids(self) -> None:
        with self.assertRaises(SourceValidationError) as context:
            SourceRegistry(
                sources=(
                    _source(source_id="canada_ca"),
                    _source(source_id="canada_ca"),
                )
            )

        self.assertIn("duplicate source_id", str(context.exception))

    def test_source_validation_rejects_invalid_required_metadata(self) -> None:
        source = _source(source_id="Bad-ID", base_url="not-a-url", languages=("de",))

        with self.assertRaises(SourceValidationError) as context:
            validate_sources([source])

        message = str(context.exception)
        self.assertIn("source_id must be lowercase snake_case", message)
        self.assertIn("base_url must be an absolute http(s) URL", message)
        self.assertIn("unsupported language 'de'", message)

    def test_source_validation_requires_source_catalog_version_metadata(self) -> None:
        source = _source(collection_date="", version_or_snapshot="")

        with self.assertRaises(SourceValidationError) as context:
            validate_sources([source])

        message = str(context.exception)
        self.assertIn("collection_date is required", message)
        self.assertIn("version_or_snapshot is required", message)

    def test_source_validation_rejects_unsupported_document_format(self) -> None:
        source = _source(document_formats=("html", "epub"))

        with self.assertRaises(SourceValidationError) as context:
            validate_sources([source])

        self.assertIn("unsupported document format 'epub'", str(context.exception))

    def test_source_validation_rejects_unsupported_collection_method(self) -> None:
        source = _source(collection_method="web_crawl")

        with self.assertRaises(SourceValidationError) as context:
            validate_sources([source])

        self.assertIn("unsupported collection method 'web_crawl'", str(context.exception))

    def test_loader_rejects_missing_sources_list(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sources.json"
            path.write_text(json.dumps({"not_sources": []}), encoding="utf-8")

            with self.assertRaises(SourceValidationError):
                load_source_registry(path)

    def test_repository_registry_configuration_is_valid(self) -> None:
        registry = load_source_registry("configs/data/source_registry.json")

        self.assertEqual(
            [source.source_id for source in registry.sources],
            [
                "canada_ca",
                "justice_laws",
                "statistics_canada",
                "open_government_portal",
                "parliament_of_canada",
            ],
        )
        self.assertEqual(len(registry.enabled_sources()), 5)
        for source in registry.sources:
            self.assertEqual(source.collection_date, "2026-07-02")
            self.assertEqual(source.version_or_snapshot, "initial_source_registry_v20260702")


def _source(
    *,
    source_id: str = "canada_ca",
    base_url: str = "https://www.canada.ca/",
    languages: tuple[str, ...] = ("en", "fr"),
    document_formats: tuple[str, ...] = ("html",),
    collection_method: str = "sitemap",
    enabled: bool = True,
    collection_date: str = "2026-07-02",
    version_or_snapshot: str = "initial_source_registry_v20260702",
) -> OfficialSource:
    return OfficialSource(
        source_id=source_id,
        name="Canada.ca",
        organization="Government of Canada",
        base_url=base_url,
        languages=languages,
        license=SourceLicense(
            name="Government of Canada site terms",
            url="https://www.canada.ca/en/transparency/terms.html",
        ),
        document_formats=document_formats,
        collection_method=collection_method,
        enabled=enabled,
        collection_date=collection_date,
        version_or_snapshot=version_or_snapshot,
        notes="Official source registry test fixture.",
    )


def _source_dict(*, source_id: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "name": "Canada.ca",
        "organization": "Government of Canada",
        "base_url": "https://www.canada.ca/",
        "languages": ["en", "fr"],
        "license": {
            "name": "Government of Canada site terms",
            "url": "https://www.canada.ca/en/transparency/terms.html",
        },
        "document_formats": ["html"],
        "collection_method": "sitemap",
        "enabled": True,
        "collection_date": "2026-07-02",
        "version_or_snapshot": "initial_source_registry_v20260702",
        "notes": "Official source registry test fixture.",
    }
