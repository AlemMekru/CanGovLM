"""Single-document acquisition-to-extraction demonstration pipeline.

This module demonstrates the first end-to-end corpus workflow for one approved
Canada.ca HTML document. It intentionally stops after extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cangovlm.data.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    CanadaCaAcquisitionClient,
    DownloadManifestRecord,
    StoredRawDocument,
    sha256_text,
)
from cangovlm.data.extraction import ExtractionResult, HtmlExtractor, extracted_storage_path
from cangovlm.data.manifests import ManifestRecord, validate_manifest_records


@dataclass(frozen=True)
class SingleDocumentPipelineResult:
    """Artifacts produced by the single-document demonstration pipeline."""

    stored_raw_document: StoredRawDocument
    extraction_result: ExtractionResult
    extracted_path: Path
    raw_manifest_record: DownloadManifestRecord
    extracted_manifest_record: ManifestRecord


class CanadaCaSingleDocumentPipeline:
    """Run acquisition through extraction for one Canada.ca HTML document."""

    def __init__(
        self,
        *,
        acquisition_client: CanadaCaAcquisitionClient,
        html_extractor: HtmlExtractor | None = None,
        pipeline_version: str = "0.1.0",
    ) -> None:
        self.acquisition_client = acquisition_client
        self.html_extractor = html_extractor or HtmlExtractor()
        self.pipeline_version = pipeline_version

    def run(self, request: AcquisitionRequest) -> SingleDocumentPipelineResult:
        """Acquire, store, extract, write text, and validate one document."""

        stored = self.acquisition_client.acquire_and_store(request)
        extraction_result = self.html_extractor.extract(stored.raw_document)
        extracted_path = Path(
            extracted_storage_path(
                str(stored.raw_path),
                extracted_root=str(_extracted_root_for_raw_path(stored.raw_path)),
            )
        )
        write_extracted_text(extracted_path, extraction_result)
        extracted_manifest_record = build_extracted_manifest_record(
            raw_manifest_record=stored.manifest_record,
            extraction_result=extraction_result,
            extracted_path=extracted_path,
            pipeline_version=self.pipeline_version,
        )
        validate_single_document_pipeline_output(
            stored_raw_document=stored,
            extraction_result=extraction_result,
            extracted_path=extracted_path,
            extracted_manifest_record=extracted_manifest_record,
        )

        return SingleDocumentPipelineResult(
            stored_raw_document=stored,
            extraction_result=extraction_result,
            extracted_path=extracted_path,
            raw_manifest_record=stored.manifest_record,
            extracted_manifest_record=extracted_manifest_record,
        )


def build_extracted_manifest_record(
    *,
    raw_manifest_record: DownloadManifestRecord,
    extraction_result: ExtractionResult,
    extracted_path: str | Path,
    pipeline_version: str,
) -> ManifestRecord:
    """Create the extracted-stage manifest record for one document."""

    return ManifestRecord(
        document_id=raw_manifest_record.document_id,
        source_id=raw_manifest_record.source_id,
        source_name=raw_manifest_record.source_name,
        organization=raw_manifest_record.organization,
        language=raw_manifest_record.language,
        url=raw_manifest_record.url,
        canonical_url=raw_manifest_record.canonical_url,
        title=extraction_result.metadata.title,
        retrieved_at=raw_manifest_record.retrieved_at,
        published_at=None,
        modified_at=None,
        license_name=raw_manifest_record.license_name,
        license_url=raw_manifest_record.license_url,
        document_format=raw_manifest_record.document_format,
        raw_path=raw_manifest_record.raw_path,
        raw_sha256=raw_manifest_record.raw_sha256,
        extracted_path=str(extracted_path),
        extracted_sha256=extraction_result.text_sha256,
        cleaned_path=None,
        cleaned_sha256=None,
        deduplicated_path=None,
        quality_status=None,
        quality_warnings=extraction_result.metadata.warnings,
        is_duplicate=False,
        duplicate_of=None,
        pipeline_version=pipeline_version,
        manifest_stage="extracted",
    )


def write_extracted_text(path: str | Path, extraction_result: ExtractionResult) -> None:
    """Write extracted UTF-8 text immutably."""

    extracted_path = Path(path)
    content = extraction_result.text.encode("utf-8")
    expected_sha256 = extraction_result.text_sha256

    if extracted_path.exists():
        existing_sha256 = sha256_text(extracted_path.read_text(encoding="utf-8"))
        if existing_sha256 != expected_sha256:
            raise AcquisitionError(f"Refusing to overwrite extracted text: {extracted_path}")
        return

    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.write_bytes(content)


def validate_single_document_pipeline_output(
    *,
    stored_raw_document: StoredRawDocument,
    extraction_result: ExtractionResult,
    extracted_path: str | Path,
    extracted_manifest_record: ManifestRecord,
) -> None:
    """Validate the acquisition-to-extraction handoff for one document."""

    errors: list[str] = []
    extracted_path = Path(extracted_path)

    if not stored_raw_document.raw_path.exists():
        errors.append("raw file does not exist")
    elif stored_raw_document.raw_path.read_bytes() != stored_raw_document.raw_document.content:
        errors.append("raw file bytes do not match raw document content")

    if not extracted_path.exists():
        errors.append("extracted text file does not exist")
    elif extracted_path.read_text(encoding="utf-8") != extraction_result.text:
        errors.append("extracted file text does not match extraction result")

    if extracted_manifest_record.raw_sha256 != stored_raw_document.raw_document.sha256:
        errors.append("extracted manifest raw_sha256 does not match raw document")
    if extracted_manifest_record.extracted_sha256 != extraction_result.text_sha256:
        errors.append("extracted manifest hash does not match extraction result")
    if extracted_manifest_record.document_id != stored_raw_document.raw_document.document_id:
        errors.append("extracted manifest document_id does not match raw document")

    if errors:
        raise AcquisitionError("\n".join(errors))

    validate_manifest_records([extracted_manifest_record])


def _extracted_root_for_raw_path(raw_path: Path) -> Path:
    return raw_path.parents[3] / "extracted"
