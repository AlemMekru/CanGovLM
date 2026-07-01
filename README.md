# CanGovLM

CanGovLM is a tiny decoder-only Transformer language model built entirely from scratch using official Canadian government data.

The objective of this project is to understand and implement every core component of a modern language model, including tokenization, data processing, model architecture, training, evaluation, and inference—without using pretrained weights or existing large language models.

The project is developed incrementally as an educational, research, and portfolio project. Every major component is designed, implemented, documented, and evaluated step by step.

## Project Goals

- Build a language model completely from scratch.
- Train exclusively on official Canadian government data.
- Support English first, followed by bilingual English/French.
- Maintain a transparent and reproducible training pipeline.
- Demonstrate end-to-end language model engineering.

## Roadmap

- ✅ Project setup

- ⏳ Tokenizer

- ⏳ Data pipeline

- ⏳ Transformer

- ⏳ Training

- ⏳ Evaluation

- ⏳ Inference

- ⏳ Demo

## Repository Layout

- `corpus/`: Curated source text organized by language.
- `data/`: Raw, interim, and processed datasets.
- `vocabulary/`: Tokenizer vocabulary and metadata.
- `configs/`: Configuration files.
- `src/cangovlm/`: Core implementation.
- `tests/`: Automated tests.
- `checkpoints/`: Saved model checkpoints.
- `artifacts/`: Logs, reports, and generated outputs.
- `inference/`: Text generation and inference.
- `benchmarks/`: Evaluation results.
- `demo/`: Demonstration applications.
- `docs/`: Design documents, roadmap, and experiment notes.

## Repository Structure

```text
.
├── .github
│   └── workflows
│       └── ci.yml
├── .gitignore
├── LICENSE
├── Makefile
├── artifacts
├── benchmarks
├── checkpoints
├── configs
│   ├── data
│   ├── model
│   ├── tokenizer
│   └── training
├── corpus
│   ├── en
│   └── fr
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── demo
├── docs
│   ├── data_sources.md
│   ├── design_notes.md
│   ├── experiments.md
│   └── roadmap.md
├── inference
├── notebooks
├── pyproject.toml
├── requirements-dev.txt
├── scripts
├── src
│   └── cangovlm
│       ├── data
│       ├── evaluation
│       ├── model
│       ├── tokenizer
│       ├── training
│       └── utils
├── tests
│   ├── data
│   ├── evaluation
│   ├── model
│   ├── tokenizer
│   └── training
└── vocabulary

```  

## Author

**Alem Mekru**

AI Engineer | MSc Artificial Intelligence

- GitHub: https://github.com/AlemMekru
- LinkedIn: https://www.linkedin.com/in/alemmekru/