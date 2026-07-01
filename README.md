# CanGovLM

CanGovLM is a tiny decoder-only Transformer language model built from scratch using official Canadian government data.

This repository is intentionally structured for incremental development. The first phase establishes the project layout only; model, tokenizer, training, and inference code will be added step by step.

## Repository Layout

- `corpus/`: Curated source text organized by language.
- `data/`: Local raw, interim, and processed datasets derived from the corpus.
- `vocabulary/`: Tokenizer vocabularies and related metadata.
- `configs/`: Reproducible configuration files for data, tokenizer, model, and training experiments.
- `src/cangovlm/`: Python package namespace for future implementation.
- `tests/`: Test suite mirroring the package layout.
- `checkpoints/`: Local model checkpoints.
- `artifacts/`: Generated outputs such as logs, samples, and reports.
- `inference/`: Future inference entry points and serving notes.
- `benchmarks/`: Future benchmark definitions and results.
- `demo/`: Future local demos or lightweight interfaces.
- `docs/`: Design notes, roadmap, data provenance, and experiment records.

