# CanGovLM Corpus Specification

This document defines the technical specification for the CanGovLM training corpus. Every future corpus processing component should follow this specification unless a documented design decision supersedes it.

CanGovLM is trained only on official Canadian government text. The corpus must therefore be reproducible, auditable, language-aware, and traceable from final training examples back to source documents.

## 1. Corpus Objectives

The CanGovLM corpus exists to support training a small decoder-only Transformer from scratch.

Primary objectives:

- Collect authoritative official Canadian government text.
- Preserve provenance for every document.
- Support English first and French later.
- Produce reproducible train, validation, and test snapshots.
- Keep raw, extracted, cleaned, and deduplicated stages separate.
- Document licensing and source-selection decisions.
- Avoid unofficial, third-party, or AI-generated text.

Non-objectives:

- Broad web crawling.
- Training on unofficial summaries.
- Translating documents.
- Rewriting source text for style.
- Mixing undocumented local files into training snapshots.

## 2. Supported Languages

### English

English is the first supported language for corpus creation and tokenizer/model development.

Language code:

```text
en
```

### French

French support will be added after the English pipeline is stable.

Language code:

```text
fr
```

### Bilingual Sources

Some official sources publish paired English and French documents. These should be represented as separate documents with explicit cross-language metadata when available.

Example:

```json
{
  "document_id": "justice_laws_en_abc123",
  "language": "en",
  "translated_document_id": "justice_laws_fr_def456"
}
```

## 3. Supported Document Formats

The pipeline should support these formats over time:

| Format | Phase | Notes |
| --- | --- | --- |
| HTML | Initial | Preferred for web-native official pages. |
| TXT | Initial | Preferred when clean official plain text exists. |
| XML | Early | Useful for structured legal and parliamentary text. |
| JSON | Early | Useful for API and dataset metadata. |
| CSV | Early | Useful for dataset records with text fields. |
| PDF | Later | Common for reports, but extraction quality must be validated. |
| DOCX | Later | Only if official and necessary. |

Raw files must retain their original format. Extracted files should use UTF-8 plain text.

## 4. Required Document Metadata

Every document must have metadata linking it to an approved source.

Required fields:

| Field | Description |
| --- | --- |
| `document_id` | Stable unique identifier for the document. |
| `source_id` | Identifier of the approved source catalog entry. |
| `source_name` | Human-readable source name. |
| `organization` | Responsible official organization. |
| `language` | `en`, `fr`, or `bilingual` before splitting. |
| `url` | Original URL or dataset record URL. |
| `canonical_url` | Canonical URL when known. |
| `title` | Document title when available. |
| `retrieved_at` | Collection timestamp. |
| `published_at` | Publication date when available. |
| `modified_at` | Last modified date when available. |
| `license_name` | License or terms name. |
| `license_url` | License or terms URL. |
| `document_format` | Original format, such as HTML or PDF. |
| `raw_path` | Path to immutable raw file. |
| `raw_sha256` | SHA-256 hash of raw file bytes. |
| `extracted_path` | Path to extracted text file. |
| `extracted_sha256` | SHA-256 hash of extracted text. |
| `cleaned_path` | Path to cleaned text file. |
| `cleaned_sha256` | SHA-256 hash of cleaned text. |
| `deduplicated_path` | Path to included deduplicated text, if included. |
| `quality_status` | `accepted`, `rejected`, or `needs_review`. |
| `quality_warnings` | List of quality warnings. |
| `is_duplicate` | Whether the document is duplicate or near-duplicate. |
| `duplicate_of` | Canonical document ID if duplicate. |
| `pipeline_version` | Corpus pipeline version. |

Example manifest record:

```json
{
  "document_id": "canada_ca_en_2f4c8a9b",
  "source_id": "canada_ca",
  "source_name": "Canada.ca",
  "organization": "Government of Canada",
  "language": "en",
  "url": "https://www.canada.ca/example",
  "canonical_url": "https://www.canada.ca/example",
  "title": "Example Official Page",
  "retrieved_at": "2026-07-01T00:00:00Z",
  "published_at": null,
  "modified_at": null,
  "license_name": "Government of Canada site terms",
  "license_url": "https://www.canada.ca/en/transparency/terms.html",
  "document_format": "html",
  "raw_path": "corpus/raw/en/canada_ca/canada_ca_en_2f4c8a9b.html",
  "raw_sha256": "example",
  "extracted_path": "corpus/extracted/en/canada_ca/canada_ca_en_2f4c8a9b.txt",
  "extracted_sha256": "example",
  "cleaned_path": "corpus/cleaned/en/canada_ca/canada_ca_en_2f4c8a9b.txt",
  "cleaned_sha256": "example",
  "deduplicated_path": "corpus/deduplicated/en/canada_ca/canada_ca_en_2f4c8a9b.txt",
  "quality_status": "accepted",
  "quality_warnings": [],
  "is_duplicate": false,
  "duplicate_of": null,
  "pipeline_version": "0.1.0"
}
```

## 5. Standard Directory Structure

The corpus directory should be stage-oriented and language-aware.

```text
corpus/
├── sources/
│   ├── en_sources.yaml
│   └── fr_sources.yaml
├── manifests/
│   ├── raw_manifest.jsonl
│   ├── extracted_manifest.jsonl
│   ├── cleaned_manifest.jsonl
│   ├── deduplicated_manifest.jsonl
│   └── snapshot_manifest.jsonl
├── raw/
│   ├── en/
│   └── fr/
├── extracted/
│   ├── en/
│   └── fr/
├── cleaned/
│   ├── en/
│   └── fr/
├── deduplicated/
│   ├── en/
│   └── fr/
└── snapshots/
    └── vYYYYMMDD/
        ├── train.txt
        ├── validation.txt
        ├── test.txt
        ├── manifest.jsonl
        ├── stats.json
        └── README.md
```

Pipeline flow:

```text
Source Selection Policy
        |
        v
Approved Source Catalog
        |
        v
Raw Archive
        |
        v
Extracted Text
        |
        v
Validated And Cleaned Text
        |
        v
Deduplicated Corpus
        |
        v
Immutable Dataset Snapshot
```

## 6. Naming Conventions

### Source IDs

Source IDs should be lowercase snake case:

```text
canada_ca
justice_laws
statistics_canada
open_government_portal
parliamentary_publications
```

### Document IDs

Document IDs should be stable and collision-resistant:

```text
{source_id}_{language}_{short_hash}
```

Example:

```text
canada_ca_en_2f4c8a9b
```

The hash should be derived from stable source identity, such as canonical URL plus source ID. If a source has no stable URL, use source-specific stable metadata and document content hash.

### File Paths

Raw files:

```text
corpus/raw/{language}/{source_id}/{document_id}.{ext}
```

Extracted text:

```text
corpus/extracted/{language}/{source_id}/{document_id}.txt
```

Cleaned text:

```text
corpus/cleaned/{language}/{source_id}/{document_id}.txt
```

Deduplicated text:

```text
corpus/deduplicated/{language}/{source_id}/{document_id}.txt
```

## 7. Language Identification Policy

Language identification should be conservative and auditable.

Policy:

- Prefer source-declared language when trustworthy.
- Validate declared language with lightweight text heuristics.
- Store both `language_hint` and final `language`.
- Reject or mark as `needs_review` if language is ambiguous.
- Do not mix English and French documents in one training file unless explicitly creating a bilingual snapshot.

Examples:

```json
{
  "language_hint": "en",
  "language": "en",
  "language_confidence": 0.98
}
```

For bilingual pages:

- Split English and French text into separate documents when possible.
- Preserve cross-language links in metadata.
- Mark unsplittable mixed-language documents as `needs_review` unless explicitly approved.

## 8. Text Extraction Policy

Extraction converts raw files into UTF-8 plain text.

General requirements:

- Preserve document order.
- Preserve headings where possible.
- Preserve legal numbering, section numbers, table labels, and citations.
- Remove obvious non-content page chrome only when extractor confidence is high.
- Record extractor name and version in metadata.
- Record extraction warnings.

Format-specific guidance:

| Format | Extraction Policy |
| --- | --- |
| HTML | Extract main content, title, headings, paragraphs, lists, and meaningful tables. Avoid navigation, footer, cookie, and search boilerplate. |
| TXT | Validate UTF-8 and pass through with minimal changes. |
| XML | Extract configured fields according to source-specific rules. Preserve structural order. |
| JSON | Extract configured text fields. Preserve record-level metadata. |
| CSV | Extract approved text columns only. Preserve row-level provenance where needed. |
| PDF | Extract page text and page count. Flag low-quality extraction, repeated headers, footers, and broken reading order. |

Extraction output must not be treated as final training text.

## 9. Cleaning And Normalization Rules

Cleaning should be conservative. It prepares extracted text for model training without rewriting the source.

Required normalization:

- Normalize Unicode to NFC.
- Normalize line endings to `\n`.
- Strip trailing whitespace.
- Collapse excessive blank lines.
- Remove known repeated boilerplate introduced by extraction.
- Remove empty documents.
- Preserve case.
- Preserve accents.
- Preserve punctuation.
- Preserve legal and policy numbering.

Allowed cleaning:

- Remove duplicated headers and footers.
- Remove navigation labels.
- Remove cookie banners.
- Normalize repeated whitespace outside preformatted sections.

Disallowed cleaning:

- Lowercasing all text.
- Removing French accents.
- Translating.
- Summarizing.
- Paraphrasing.
- Removing legal references.
- Removing all numbers.
- Silently deleting large sections without metadata warnings.

Example cleaning record:

```json
{
  "cleaner": "conservative_text_cleaner_v1",
  "normalization": ["unicode_nfc", "line_endings_lf", "trim_trailing_whitespace"],
  "removed_boilerplate": true,
  "warnings": ["repeated_footer_removed"]
}
```

## 10. Duplicate And Near-Duplicate Policy

Duplicate detection protects evaluation quality and prevents over-representing repeated official text.

### Exact Duplicates

Exact duplicates are detected using SHA-256 hashes of normalized cleaned text.

Policy:

- Keep one canonical document.
- Mark all exact duplicates with `is_duplicate: true`.
- Preserve duplicate records in manifests.
- Exclude duplicate copies from train/validation/test text files.

### Near Duplicates

Near duplicates should be detected using text similarity.

Initial policy:

- Normalize cleaned text lightly.
- Split into word shingles.
- Compute similarity between shingle sets.
- Mark documents above the configured threshold as near duplicates.

Recommended initial threshold:

```text
0.90 Jaccard similarity
```

Canonical selection should prefer:

- Official canonical URL.
- Newer version when clearly superseding older text.
- Higher extraction quality.
- HTML over PDF when text is equivalent and cleaner.

## 11. Train/Validation/Test Snapshot Strategy

Training must use immutable versioned snapshots, not moving directories.

A snapshot is a frozen dataset release. Once created, a snapshot must never be edited in place. New government documents, source corrections, extraction improvements, cleaning changes, deduplication changes, or split changes must create a new snapshot version.

Snapshot directory:

```text
corpus/snapshots/vYYYYMMDD/
```

Files:

```text
train.txt
validation.txt
test.txt
manifest.jsonl
stats.json
README.md
```

Snapshot immutability rules:

- Do not modify files inside an existing snapshot directory after release.
- Do not append new documents to an existing snapshot.
- Do not remove documents from an existing snapshot.
- Do not regenerate `train.txt`, `validation.txt`, `test.txt`, `manifest.jsonl`, `stats.json`, or `README.md` in place.
- Do not reuse a snapshot ID for different contents.
- If a mistake is found, mark the snapshot as superseded in documentation and create a new snapshot.
- Training runs must reference an explicit snapshot ID, never an implicit "latest" directory.

Split policy:

- Split at document level, not line level.
- Keep all text from a document in one split.
- Use deterministic hash-based assignment.
- Exclude rejected and duplicate documents.
- Preserve snapshot membership in metadata.

Recommended initial split:

```text
train: 98%
validation: 1%
test: 1%
```

Snapshot flow:

```text
deduplicated accepted documents
        |
        v
deterministic document-level split
        |
        v
train.txt / validation.txt / test.txt
        |
        v
snapshot manifest + stats + README
```

Snapshot lifecycle:

```text
candidate inputs
        |
        v
build candidate snapshot in temporary workspace
        |
        v
validate manifests, hashes, metadata, splits, and stats
        |
        v
publish immutable snapshot directory
        |
        v
use snapshot by explicit version ID
        |
        v
supersede with a new snapshot when data or logic changes
```

Lifecycle states:

| State | Meaning | Mutation Policy |
| --- | --- | --- |
| `candidate` | Snapshot is being assembled in a temporary workspace. | May be regenerated before release. |
| `published` | Snapshot passed QA and is available for training. | Must not be modified. |
| `superseded` | A newer snapshot should be used for future training. | Must not be modified. |
| `rejected` | Candidate failed QA and was not released. | May be deleted from temporary workspace. |

Published snapshot README files must identify the lifecycle state and, if superseded, the replacement snapshot ID.

## 12. Corpus Versioning Strategy

Corpus versions should identify immutable dataset snapshots.

Recommended version format:

```text
vYYYYMMDD
```

Examples:

```text
v20260701
v20260815
```

If more than one snapshot is created on the same date, append a deterministic sequence suffix:

```text
v20260701.1
v20260701.2
```

Each version must include:

- Snapshot ID.
- Snapshot lifecycle state.
- Source catalog version or hash.
- Pipeline version.
- Config hashes.
- Raw manifest hash.
- Extracted manifest hash.
- Cleaned manifest hash.
- Deduplicated manifest hash.
- Snapshot manifest hash.
- Split file hashes for `train.txt`, `validation.txt`, and `test.txt`.
- Creation timestamp.
- Parent or previous snapshot ID when applicable.
- Supersedes snapshot ID when applicable.
- Superseded by snapshot ID when applicable.
- Human-readable changelog.

Version changes should be created when:

- New government documents are added.
- Existing source documents are updated upstream and recollected.
- Sources are added or removed.
- Collection dates change.
- Extraction logic changes.
- Cleaning logic changes.
- Deduplication logic changes.
- Train/validation/test split strategy changes.
- Manifest schema changes.
- Source licensing or provenance metadata changes.

Snapshot version metadata example:

```json
{
  "snapshot_id": "v20260701",
  "state": "published",
  "created_at": "2026-07-01T00:00:00Z",
  "pipeline_version": "0.1.0",
  "source_catalog_sha256": "example",
  "raw_manifest_sha256": "example",
  "extracted_manifest_sha256": "example",
  "cleaned_manifest_sha256": "example",
  "deduplicated_manifest_sha256": "example",
  "snapshot_manifest_sha256": "example",
  "train_sha256": "example",
  "validation_sha256": "example",
  "test_sha256": "example",
  "previous_snapshot_id": null,
  "supersedes": null,
  "superseded_by": null,
  "changelog": "Initial English corpus snapshot."
}
```

Update policy:

```text
existing snapshot + new documents or changed pipeline logic
        |
        v
new candidate snapshot
        |
        v
QA validation
        |
        v
new published snapshot ID
```

Existing published snapshots are historical artifacts. They may be referenced, audited, or superseded, but never changed.

## 13. Quality Assurance Checklist

Before a corpus snapshot can be used for training, verify:

- All included sources passed the Source Selection Policy.
- Every document has required metadata.
- Every document has license information.
- Every document links to an approved source.
- Raw file hashes exist.
- Extracted text hashes exist.
- Cleaned text hashes exist.
- Rejected documents are excluded.
- Duplicates are excluded from final split files.
- Split assignment is deterministic.
- English snapshot contains English documents only.
- French snapshot contains French documents only.
- Empty and tiny documents are removed or reviewed.
- Boilerplate rates are acceptable.
- Encoding is valid UTF-8.
- Snapshot README is complete.
- Snapshot stats are generated.
- Snapshot ID is unique.
- Snapshot lifecycle state is documented.
- Snapshot files have SHA-256 hashes.
- Snapshot was built in a temporary workspace before publication.
- No existing published snapshot was modified.

## 14. Reproducibility Requirements

Corpus creation must be reproducible from documented inputs.

Required reproducibility artifacts:

- Source catalog.
- Pipeline configuration files.
- Manifests for every stage.
- Hashes for raw, extracted, cleaned, and snapshot files.
- Pipeline version.
- Processing timestamp.
- Snapshot README.
- Snapshot statistics.
- Snapshot lifecycle metadata.
- Changelog describing why the snapshot exists.

Every stage should be deterministic for the same inputs and configuration.

Reproducibility rule:

```text
same source catalog + same raw inputs + same configs + same pipeline version
        |
        v
same manifests + same split files + same snapshot hashes
```

Any future update command must create a new snapshot rather than modifying an existing one.

Rebuild record example:

```json
{
  "snapshot_id": "v20260701",
  "state": "published",
  "pipeline_version": "0.1.0",
  "source_catalog_sha256": "example",
  "raw_manifest_sha256": "example",
  "extracted_manifest_sha256": "example",
  "cleaning_config_sha256": "example",
  "deduplication_config_sha256": "example",
  "snapshot_manifest_sha256": "example",
  "created_at": "2026-07-01T00:00:00Z"
}
```

## 15. Future Extensibility Guidelines

The corpus pipeline should remain modular as CanGovLM grows.

Guidelines:

- Add French support through language-specific configs, not a separate pipeline.
- Add new sources through the source catalog, not one-off scripts.
- Add new file formats through extractor modules with tests.
- Keep raw files immutable.
- Keep published snapshots immutable.
- Preserve backwards compatibility for manifest fields where possible.
- Version breaking metadata changes.
- Keep document-level provenance even when chunking text later.
- Keep tokenizer and model training independent of collection code.
- Use synthetic fixtures for tests.
- Do not require internet access for unit tests.

Future extensions may include:

- French corpus snapshots.
- Bilingual snapshot variants.
- Parallel English/French document linking.
- PDF extraction quality scoring.
- MinHash-based large-scale near-duplicate detection.
- Dataset cards for public release.
- Token-level corpus statistics after tokenizer finalization.

## Summary

The CanGovLM corpus specification defines a traceable path from approved official sources to immutable training snapshots:

```text
policy -> source catalog -> raw archive -> extracted text -> cleaned text -> deduplicated corpus -> versioned snapshot
```

Every future corpus component should preserve this chain. If the chain breaks, the document should not enter a training snapshot.
