from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from cangovlm.data import (
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_STAGES,
    Manifest,
    ManifestError,
    ManifestRecord,
    ManifestValidationError,
    load_manifest,
    validate_manifest_records,
    write_manifest,
)


class ManifestFrameworkTests(TestCase):
    def test_manifest_stages_cover_corpus_specification(self) -> None:
        self.assertEqual(
            MANIFEST_STAGES,
            {"raw", "extracted", "cleaned", "deduplicated", "snapshot"},
        )

    def test_manifest_record_serializes_quality_warnings_as_json_list(self) -> None:
        record = _record(manifest_stage="raw", quality_warnings=("warning",))

        serialized = record.to_dict()

        self.assertEqual(serialized["manifest_schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(serialized["quality_warnings"], ["warning"])

    def test_manifest_record_round_trips_through_dict(self) -> None:
        record = _record(manifest_stage="cleaned")

        loaded = ManifestRecord.from_dict(record.to_dict())

        self.assertEqual(loaded, record)

    def test_write_manifest_is_deterministic_and_sorted_by_document_id(self) -> None:
        first = _record(document_id="canada_ca_en_aaaaaaaaaaaa", manifest_stage="raw")
        second = _record(document_id="canada_ca_en_bbbbbbbbbbbb", manifest_stage="raw")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw_manifest.jsonl"
            write_manifest(path, [second, first])
            first_write = path.read_text(encoding="utf-8")
            write_manifest(path, [second, first])
            second_write = path.read_text(encoding="utf-8")

        lines = first_write.splitlines()
        self.assertEqual(first_write, second_write)
        self.assertIn('"document_id":"canada_ca_en_aaaaaaaaaaaa"', lines[0])
        self.assertIn('"document_id":"canada_ca_en_bbbbbbbbbbbb"', lines[1])

    def test_load_manifest_loads_jsonl_records(self) -> None:
        record = _record(manifest_stage="raw")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw_manifest.jsonl"
            write_manifest(path, [record])

            manifest = load_manifest(path)

        self.assertEqual(manifest.records, (record,))

    def test_load_manifest_rejects_invalid_jsonl(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")

            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_manifest_validation_rejects_duplicate_document_ids(self) -> None:
        record = _record(manifest_stage="raw")

        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest_records([record, record])

        self.assertIn("duplicate document_id", str(context.exception))

    def test_manifest_validation_enforces_stage_specific_fields(self) -> None:
        record = _record(manifest_stage="extracted", extracted_path=None)

        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest_records([record])

        self.assertIn("extracted_path is required", str(context.exception))

    def test_manifest_validation_rejects_unsupported_schema_version(self) -> None:
        record = _record(manifest_stage="raw", manifest_schema_version="9.9")

        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest_records([record])

        self.assertIn("unsupported manifest_schema_version", str(context.exception))

    def test_manifest_validation_rejects_duplicate_without_canonical_document(self) -> None:
        record = _record(manifest_stage="deduplicated", is_duplicate=True, duplicate_of=None)

        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest_records([record])

        self.assertIn("duplicate records must include duplicate_of", str(context.exception))

    def test_snapshot_records_must_be_accepted_non_duplicates_with_split(self) -> None:
        record = _record(manifest_stage="snapshot", split="training")

        with self.assertRaises(ManifestValidationError) as context:
            validate_manifest_records([record])

        self.assertIn("unsupported snapshot split", str(context.exception))

    def test_manifest_querying_filters_records(self) -> None:
        raw = _record(document_id="canada_ca_en_aaaaaaaaaaaa", manifest_stage="raw")
        snapshot = _record(
            document_id="justice_laws_fr_bbbbbbbbbbbb",
            source_id="justice_laws",
            language="fr",
            manifest_stage="snapshot",
            split="validation",
        )
        duplicate = _record(
            document_id="canada_ca_en_cccccccccccc",
            manifest_stage="deduplicated",
            is_duplicate=True,
            duplicate_of="canada_ca_en_aaaaaaaaaaaa",
            deduplicated_path=None,
        )
        manifest = Manifest((raw, snapshot, duplicate))

        self.assertEqual(manifest.get(raw.document_id), raw)
        self.assertEqual(manifest.stages, ("deduplicated", "raw", "snapshot"))
        self.assertEqual(manifest.schema_versions, (MANIFEST_SCHEMA_VERSION,))
        self.assertEqual(manifest.for_stage("snapshot"), (snapshot,))
        self.assertEqual(manifest.for_language("fr"), (snapshot,))
        self.assertEqual(manifest.for_source("canada_ca"), (raw, duplicate))
        self.assertEqual(manifest.accepted(), (snapshot,))
        self.assertEqual(manifest.duplicates(), (duplicate,))
        self.assertEqual(manifest.for_snapshot_split("validation"), (snapshot,))

    def test_manifest_loaded_json_is_plain_jsonl_objects(self) -> None:
        record = _record(manifest_stage="snapshot")

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "snapshot_manifest.jsonl"
            write_manifest(path, [record])
            loaded_json = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(loaded_json["manifest_stage"], "snapshot")
        self.assertEqual(loaded_json["split"], "train")


def _record(
    *,
    document_id: str = "canada_ca_en_2f4c8a9b0000",
    source_id: str = "canada_ca",
    language: str = "en",
    manifest_stage: str,
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION,
    raw_path: str | None = "corpus/raw/en/canada_ca/canada_ca_en_2f4c8a9b0000.html",
    raw_sha256: str | None = "a" * 64,
    extracted_path: str | None = "corpus/extracted/en/canada_ca/canada_ca_en_2f4c8a9b0000.txt",
    extracted_sha256: str | None = "b" * 64,
    cleaned_path: str | None = "corpus/cleaned/en/canada_ca/canada_ca_en_2f4c8a9b0000.txt",
    cleaned_sha256: str | None = "c" * 64,
    deduplicated_path: str | None = (
        "corpus/deduplicated/en/canada_ca/canada_ca_en_2f4c8a9b0000.txt"
    ),
    quality_status: str | None = "accepted",
    is_duplicate: bool = False,
    duplicate_of: str | None = None,
    split: str | None = None,
    quality_warnings: tuple[str, ...] = (),
) -> ManifestRecord:
    if manifest_stage == "raw":
        extracted_path = None
        extracted_sha256 = None
        cleaned_path = None
        cleaned_sha256 = None
        deduplicated_path = None
        quality_status = None
    elif manifest_stage == "extracted":
        cleaned_path = None
        cleaned_sha256 = None
        deduplicated_path = None
        quality_status = None
    elif manifest_stage == "cleaned":
        deduplicated_path = None
    elif manifest_stage == "snapshot" and split is None:
        split = "train"

    return ManifestRecord(
        document_id=document_id,
        source_id=source_id,
        source_name="Canada.ca",
        organization="Government of Canada",
        language=language,
        url="https://www.canada.ca/example",
        canonical_url="https://www.canada.ca/example",
        title="Example Official Page",
        retrieved_at="2026-07-01T00:00:00Z",
        published_at=None,
        modified_at=None,
        license_name="Government of Canada site terms",
        license_url="https://www.canada.ca/en/transparency/terms.html",
        document_format="html",
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        extracted_path=extracted_path,
        extracted_sha256=extracted_sha256,
        cleaned_path=cleaned_path,
        cleaned_sha256=cleaned_sha256,
        deduplicated_path=deduplicated_path,
        quality_status=quality_status,
        quality_warnings=quality_warnings,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        pipeline_version="0.1.0",
        manifest_stage=manifest_stage,
        manifest_schema_version=manifest_schema_version,
        snapshot_id="v20260701" if manifest_stage == "snapshot" else None,
        split=split,
    )
