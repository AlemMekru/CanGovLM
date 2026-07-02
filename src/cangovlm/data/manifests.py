"""JSONL manifest framework for the CanGovLM corpus pipeline.

Manifests are the audit trail between corpus stages. This module serializes,
loads, validates, versions, and queries manifest records without acquiring,
extracting, cleaning, or parsing documents.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from cangovlm.data.acquisition import DOCUMENT_ID_PATTERN

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_STAGES = frozenset({"raw", "extracted", "cleaned", "deduplicated", "snapshot"})
QUALITY_STATUSES = frozenset({"accepted", "rejected", "needs_review"})
SNAPSHOT_SPLITS = frozenset({"train", "validation", "test"})


class ManifestError(ValueError):
    """Base error for invalid manifest data."""


class ManifestValidationError(ManifestError):
    """Raised when one or more manifest records fail validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class ManifestRecord:
    """One JSONL manifest record for a corpus document."""

    document_id: str
    source_id: str
    source_name: str
    organization: str
    language: str
    url: str
    canonical_url: str | None
    title: str | None
    retrieved_at: str | None
    published_at: str | None
    modified_at: str | None
    license_name: str
    license_url: str | None
    document_format: str
    raw_path: str | None
    raw_sha256: str | None
    extracted_path: str | None
    extracted_sha256: str | None
    cleaned_path: str | None
    cleaned_sha256: str | None
    deduplicated_path: str | None
    quality_status: str | None
    quality_warnings: tuple[str, ...]
    is_duplicate: bool
    duplicate_of: str | None
    pipeline_version: str
    manifest_stage: str
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    snapshot_id: str | None = None
    split: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ManifestRecord":
        """Create a manifest record from JSON-compatible data."""

        return cls(
            document_id=str(data.get("document_id", "")),
            source_id=str(data.get("source_id", "")),
            source_name=str(data.get("source_name", "")),
            organization=str(data.get("organization", "")),
            language=str(data.get("language", "")),
            url=str(data.get("url", "")),
            canonical_url=_optional_string(data.get("canonical_url")),
            title=_optional_string(data.get("title")),
            retrieved_at=_optional_string(data.get("retrieved_at")),
            published_at=_optional_string(data.get("published_at")),
            modified_at=_optional_string(data.get("modified_at")),
            license_name=str(data.get("license_name", "")),
            license_url=_optional_string(data.get("license_url")),
            document_format=str(data.get("document_format", "")),
            raw_path=_optional_string(data.get("raw_path")),
            raw_sha256=_optional_string(data.get("raw_sha256")),
            extracted_path=_optional_string(data.get("extracted_path")),
            extracted_sha256=_optional_string(data.get("extracted_sha256")),
            cleaned_path=_optional_string(data.get("cleaned_path")),
            cleaned_sha256=_optional_string(data.get("cleaned_sha256")),
            deduplicated_path=_optional_string(data.get("deduplicated_path")),
            quality_status=_optional_string(data.get("quality_status")),
            quality_warnings=_string_tuple(data.get("quality_warnings")),
            is_duplicate=bool(data.get("is_duplicate", False)),
            duplicate_of=_optional_string(data.get("duplicate_of")),
            pipeline_version=str(data.get("pipeline_version", "")),
            manifest_stage=str(data.get("manifest_stage", "")),
            manifest_schema_version=str(
                data.get("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
            ),
            snapshot_id=_optional_string(data.get("snapshot_id")),
            split=_optional_string(data.get("split")),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible dictionary."""

        return {
            field.name: _json_value(getattr(self, field.name))
            for field in fields(self)
        }


@dataclass(frozen=True)
class Manifest:
    """A validated collection of manifest records."""

    records: tuple[ManifestRecord, ...]

    def __post_init__(self) -> None:
        validate_manifest_records(self.records)

    @property
    def stages(self) -> tuple[str, ...]:
        """Manifest stages present in deterministic order."""

        return tuple(sorted({record.manifest_stage for record in self.records}))

    @property
    def schema_versions(self) -> tuple[str, ...]:
        """Schema versions present in deterministic order."""

        return tuple(sorted({record.manifest_schema_version for record in self.records}))

    def get(self, document_id: str) -> ManifestRecord:
        """Return one record by document ID."""

        for record in self.records:
            if record.document_id == document_id:
                return record
        raise KeyError(f"Unknown document_id: {document_id}")

    def for_stage(self, stage: str) -> tuple[ManifestRecord, ...]:
        """Return records for one corpus stage."""

        _validate_stage_name(stage)
        return tuple(record for record in self.records if record.manifest_stage == stage)

    def for_language(self, language: str) -> tuple[ManifestRecord, ...]:
        """Return records for one language."""

        if language not in {"en", "fr", "bilingual"}:
            raise ManifestError(f"Unsupported language query: {language!r}")
        return tuple(record for record in self.records if record.language == language)

    def for_source(self, source_id: str) -> tuple[ManifestRecord, ...]:
        """Return records for one source ID."""

        return tuple(record for record in self.records if record.source_id == source_id)

    def accepted(self) -> tuple[ManifestRecord, ...]:
        """Return accepted records that are not marked as duplicates."""

        return tuple(
            record
            for record in self.records
            if record.quality_status == "accepted" and not record.is_duplicate
        )

    def duplicates(self) -> tuple[ManifestRecord, ...]:
        """Return records marked as duplicates."""

        return tuple(record for record in self.records if record.is_duplicate)

    def for_snapshot_split(self, split: str) -> tuple[ManifestRecord, ...]:
        """Return snapshot records for one train/validation/test split."""

        if split not in SNAPSHOT_SPLITS:
            raise ManifestError(f"Unsupported snapshot split query: {split!r}")
        return tuple(
            record
            for record in self.records
            if record.manifest_stage == "snapshot" and record.split == split
        )


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    """Write records to JSONL in deterministic order."""

    manifest = Manifest(tuple(records))
    sorted_records = sorted(manifest.records, key=lambda record: record.document_id)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in sorted_records:
            file.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
            file.write("\n")


def load_manifest(path: str | Path) -> Manifest:
    """Load a JSONL manifest file."""

    records: list[ManifestRecord] = []
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{manifest_path}:{line_number}: invalid JSONL") from error
            if not isinstance(data, dict):
                raise ManifestError(f"{manifest_path}:{line_number}: record must be a JSON object")
            records.append(ManifestRecord.from_dict(data))

    return Manifest(tuple(records))


def validate_manifest_records(records: Iterable[ManifestRecord]) -> None:
    """Validate manifest records and raise one aggregated error if needed."""

    record_list = list(records)
    errors: list[str] = []
    seen_document_ids: set[str] = set()

    for record in record_list:
        errors.extend(_validate_record(record))
        if record.document_id in seen_document_ids:
            errors.append(f"{record.document_id}: duplicate document_id.")
        seen_document_ids.add(record.document_id)

    if errors:
        raise ManifestValidationError(errors)


def _validate_record(record: ManifestRecord) -> list[str]:
    errors: list[str] = []
    prefix = record.document_id or "<missing document_id>"

    _require_text(errors, prefix, "document_id", record.document_id)
    if record.document_id and not DOCUMENT_ID_PATTERN.fullmatch(record.document_id):
        errors.append(f"{prefix}: document_id does not match the corpus naming convention.")
    _require_text(errors, prefix, "source_id", record.source_id)
    _require_text(errors, prefix, "source_name", record.source_name)
    _require_text(errors, prefix, "organization", record.organization)
    if record.language not in {"en", "fr", "bilingual"}:
        errors.append(f"{prefix}: unsupported language {record.language!r}.")
    _require_text(errors, prefix, "url", record.url)
    _require_text(errors, prefix, "license_name", record.license_name)
    _require_text(errors, prefix, "document_format", record.document_format)
    _require_text(errors, prefix, "pipeline_version", record.pipeline_version)
    if record.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"{prefix}: unsupported manifest_schema_version "
            f"{record.manifest_schema_version!r}."
        )
    if record.manifest_stage not in MANIFEST_STAGES:
        errors.append(f"{prefix}: unsupported manifest_stage {record.manifest_stage!r}.")
        return errors

    if record.quality_status is not None and record.quality_status not in QUALITY_STATUSES:
        errors.append(f"{prefix}: unsupported quality_status {record.quality_status!r}.")
    if record.is_duplicate and not record.duplicate_of:
        errors.append(f"{prefix}: duplicate records must include duplicate_of.")

    _validate_stage_requirements(record, errors, prefix)
    return errors


def _validate_stage_requirements(
    record: ManifestRecord,
    errors: list[str],
    prefix: str,
) -> None:
    if record.manifest_stage in {"raw", "extracted", "cleaned", "deduplicated", "snapshot"}:
        _require_text(errors, prefix, "retrieved_at", record.retrieved_at)
        _require_text(errors, prefix, "raw_path", record.raw_path)
        _require_text(errors, prefix, "raw_sha256", record.raw_sha256)

    if record.manifest_stage in {"extracted", "cleaned", "deduplicated", "snapshot"}:
        _require_text(errors, prefix, "extracted_path", record.extracted_path)
        _require_text(errors, prefix, "extracted_sha256", record.extracted_sha256)

    if record.manifest_stage in {"cleaned", "deduplicated", "snapshot"}:
        _require_text(errors, prefix, "cleaned_path", record.cleaned_path)
        _require_text(errors, prefix, "cleaned_sha256", record.cleaned_sha256)
        _require_text(errors, prefix, "quality_status", record.quality_status)

    if record.manifest_stage in {"deduplicated", "snapshot"}:
        if record.quality_status == "accepted" and not record.is_duplicate:
            _require_text(errors, prefix, "deduplicated_path", record.deduplicated_path)

    if record.manifest_stage == "snapshot":
        _require_text(errors, prefix, "snapshot_id", record.snapshot_id)
        _require_text(errors, prefix, "split", record.split)
        if record.split is not None and record.split not in SNAPSHOT_SPLITS:
            errors.append(f"{prefix}: unsupported snapshot split {record.split!r}.")
        if record.quality_status != "accepted":
            errors.append(f"{prefix}: snapshot records must have quality_status 'accepted'.")
        if record.is_duplicate:
            errors.append(f"{prefix}: duplicate records cannot be included in snapshots.")


def _require_text(errors: list[str], prefix: str, field_name: str, value: str | None) -> None:
    if value is None or not str(value).strip():
        errors.append(f"{prefix}: {field_name} is required.")


def _validate_stage_name(stage: str) -> None:
    if stage not in MANIFEST_STAGES:
        raise ManifestError(f"Unsupported manifest stage query: {stage!r}")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value
