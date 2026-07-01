# CanGovLM Data Sources

This document records the source selection policy, approved source catalog, licensing notes, retrieval metadata, and provenance requirements for the CanGovLM corpus.

CanGovLM must be trained only on traceable official Canadian government text. Source quality is therefore decided before collection begins, not after data has already entered the corpus.

## Why Source Selection Comes First

Language models inherit the quality and trust boundaries of their training data. For CanGovLM, the goal is not broad web coverage; it is a small, auditable model trained from official Canadian government sources.

Adding a Source Selection Policy before collection gives the project clear guardrails:

- It prevents unofficial summaries from entering the corpus.
- It separates authoritative public records from commentary about those records.
- It makes licensing review mandatory before ingestion.
- It supports reproducible dataset creation because every source is approved before use.
- It keeps future English and French corpus expansion consistent.

## Architecture

```text
Source Selection Policy
        |
        v
Registered Official Sources
        |
        v
Acquire Immutable Raw Files
        |
        v
Extract Text And Metadata
        |
        v
Validate Quality, Language, License, And Provenance
        |
        v
Clean Text Conservatively
        |
        v
Detect Exact And Near Duplicates
        |
        v
Write Language-Specific Deduplicated Corpus
        |
        v
Create Immutable Train/Validation/Test Snapshot
        |
        v
Document Provenance, Licensing, Statistics, And Reproducibility
```

## Source Selection Policy

The Source Selection Policy defines strict inclusion and exclusion criteria before any data is collected.

### Inclusion Criteria

A source may be included only if it satisfies all required criteria:

- The source is operated by the Government of Canada, Parliament of Canada, a federal department, a federal agency, or another official federal institution.
- The text is primary-source material, not a third-party explanation of government material.
- The source URL, organization, license or terms, language, and collection method can be documented.
- The source can be collected reproducibly through stable URLs, datasets, APIs, feeds, or documented snapshots.
- The source is suitable for language modeling after conservative cleaning.
- The source does not knowingly contain AI-generated bulk text unless explicitly approved and documented as such.

### Approved Source Examples

The following source families are eligible for review and cataloging:

- Canada.ca
- Justice Laws Website
- Statistics Canada
- Open Government Portal
- Parliamentary publications
- Official federal departments and agencies

Examples of official federal departments and agencies include Health Canada, Environment and Climate Change Canada, Immigration, Refugees and Citizenship Canada, Employment and Social Development Canada, the Canada Revenue Agency, and Global Affairs Canada.

### Exclusion Criteria

The following sources must not be included:

- Wikipedia
- Blogs
- Reddit
- News websites
- AI-generated content
- Third-party summaries
- Search result snippets
- Unofficial mirrors of government documents
- Commercial legal summaries
- Social media posts unless explicitly approved as official federal records

### Review Status

Every candidate source must have one of these statuses:

- `included`: Approved for collection.
- `excluded`: Rejected and not collected.
- `needs_review`: Requires licensing, provenance, quality, or scope review before collection.

No `needs_review` source should be included in a dataset snapshot.

## Source Metadata Catalog

Every approved or reviewed source must be recorded in the metadata catalog before collection.

| Field | Required | Description |
| --- | --- | --- |
| Source name | Yes | Human-readable source name. |
| URL | Yes | Canonical source URL or dataset landing page. |
| Organization | Yes | Responsible federal organization, department, agency, or parliamentary body. |
| Language | Yes | `en`, `fr`, or `bilingual`. |
| License | Yes | License name, terms name, or governing usage policy. |
| Collection date | Yes | Date the source was collected or reviewed. |
| Version or snapshot | Yes | Dataset version, publication version, retrieval snapshot, or timestamp. |
| Document format | Yes | Expected formats such as HTML, TXT, PDF, CSV, XML, JSON, or DOCX. |
| Included | Yes | `Yes` if approved for ingestion, `No` if excluded. |
| Notes | No | Scope limits, known issues, exclusions, or collection details. |

### Catalog Template

| Source name | URL | Organization | Language | License | Collection date | Version or snapshot | Document format | Included | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Canada.ca | https://www.canada.ca/ | Government of Canada | en/fr | Site terms / applicable Government of Canada usage terms | TBD | TBD | HTML | Yes | Official federal web content. Collect only approved pages or documented collections. |
| Justice Laws Website | https://laws-lois.justice.gc.ca/ | Department of Justice Canada | en/fr | Site terms / official legal publication terms | TBD | TBD | HTML, XML, PDF | Yes | Acts and regulations. Preserve legal numbering and bilingual provenance. |
| Statistics Canada | https://www.statcan.gc.ca/ | Statistics Canada | en/fr | Statistics Canada terms / applicable open data license | TBD | TBD | HTML, CSV, JSON, PDF | Yes | Prefer text publications and metadata-rich datasets. |
| Open Government Portal | https://open.canada.ca/ | Treasury Board of Canada Secretariat | en/fr | Open Government Licence - Canada where applicable | TBD | TBD | HTML, CSV, JSON, XML, PDF | Yes | Use dataset-level metadata and license fields. |
| Parliamentary publications | https://www.parl.ca/ | Parliament of Canada | en/fr | Parliamentary publication terms | TBD | TBD | HTML, PDF, XML | Yes | Include only official publications with documented provenance. |
| Wikipedia | https://www.wikipedia.org/ | Wikimedia Foundation | multilingual | Third-party content license | N/A | N/A | HTML | No | Excluded because it is not an official Canadian government source. |

## Per-Source Documentation Template

Each source should have a dedicated entry using this template.

### Source Name

**URL:**  
TBD

**Organization:**  
TBD

**Language:**  
TBD

**License:**  
TBD

**License URL or Terms URL:**  
TBD

**Collection Date:**  
TBD

**Version or Snapshot:**  
TBD

**Document Format:**  
TBD

**Included:**  
TBD

**Collection Method:**  
TBD

**Scope Included:**  
TBD

**Scope Excluded:**  
TBD

**Provenance Notes:**  
TBD

**Quality Notes:**  
TBD

**Duplicate Handling Notes:**  
TBD

**Reviewer:**  
TBD

**Review Status:**  
TBD

## Provenance Requirements

For every collected document, downstream manifests must link back to an approved source catalog entry. At minimum, each document should preserve:

- Source name
- Source URL
- Organization
- Language
- License
- Collection date
- Version or snapshot
- Document format
- Raw file path
- Raw SHA-256 hash
- Extracted text path
- Cleaned text path
- Deduplication status
- Dataset snapshot membership

Documents that cannot be traced to an approved source entry must be excluded from training snapshots.
