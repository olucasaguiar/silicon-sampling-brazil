# Silicon Sampling Brazil

[![IEEE ACDSA 2027](https://img.shields.io/badge/IEEE-ACDSA%202027-blue)](paper/main.tex)
[![Python 3.14+](https://img.shields.io/badge/Python-3.14%2B-green.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/uv-enabled-brightgreen)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/olucasaguiar/silicon-sampling-brazil/actions/workflows/tests.yml/badge.svg)](https://github.com/olucasaguiar/silicon-sampling-brazil/actions/workflows/tests.yml)

Replication repository for:

> **Simulating Brazilian Public Opinion with Large Language Models: Distributional Evaluation and Prompt Engineering**  
> Lucas N. Aguiar, Matheus S. Moreira, Rogério de Oliveira  
> *Proc. of International Conference on Artificial Intelligence, Computer, Data Sciences and Applications (ACDSA 2027)*  
> IEEE, Rio de Janeiro, Brazil — February 2–4, 2027

## Overview

This repository contains the replication code and data for evaluating whether LLMs can accurately simulate public opinion distributions in Brazil (*silicon sampling*). While most literature in this area focuses on the US or Europe, we evaluate both Portuguese-centric models (Sabiá-4, Sabiazinho-4) and global baselines (GPT-5-mini, LLaMA 3.2 3B) against Brazilian survey data.

We generate synthetic personas from IBGE/SIDRA census microdata and compare simulated survey responses with ground-truth data from a CESOP/IPEC survey on perceptions of democracy. Distributional alignment is measured using Jensen-Shannon Divergence (JSD). By combining feature selection, Chain-of-Thought reasoning, and few-shot calibration, alignment improved significantly (JSD dropped from 0.525 to 0.121). In particular, Brazilian models better captured nuances of local civic skepticism compared to global baselines.

## Main Results

### Phase 1: JSD Evolution across Prompt Techniques (P02, Sabiazinho-4)

| Scenario | Technique | JSD | ΔJSD |
| :--- | :--- | :---: | :---: |
| Baseline | Raw prompt (13 attributes) | 0.525 | — |
| Scenario 1 | Feature selection (7 attributes) | 0.366 | −0.159 |
| Scenario 2 | Natural prose prompt | 0.366 | 0.000 |
| Scenario 3 | System/User + Chain of Thought | 0.297 | −0.069 |
| Scenario 4 | Few-shot calibration (3 personas) | 0.121 | −0.176 |

*Note: Lower JSD indicates closer alignment to the empirical CESOP distribution (target: JSD < 0.15).*

### Phase 2: Model Benchmark (Scenario 4 Prompt across P02, P03, P04)

| Question | Model | JSD |
| :--- | :--- | :---: |
| P02: Policy Priorities | Sabiá-4 | 0.7238 |
| P02: Policy Priorities | Sabiazinho-4 | 0.5664 |
| P02: Policy Priorities | GPT-5-mini | 0.6185 |
| P02: Policy Priorities | LLaMA 3.2 3B | 0.4882 |
| P03: Fake News | Sabiazinho-4 | 0.5452 |
| P04: Political Participation | Sabiazinho-4 | 0.2097 |
| P04: Political Participation | GPT-5-mini | 0.8263 |
| Mean JSD | Sabiazinho-4 | 0.4404 |

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Python package manager)
- API keys for [Maritaca AI](https://maritaca.ai) and/or [OpenAI](https://openai.com)
- Hugging Face token (optional, only needed for local LLaMA 3.2 inference)

### Setup

```bash
git clone https://github.com/olucasaguiar/silicon-sampling-brazil.git
cd silicon-sampling-brazil

# Configure environment variables
cp .env.example .env

# Install dependencies
uv sync
```

### Data

The raw CESOP survey file (`04832.SAV`) is licensed and cannot be redistributed directly. However, the preprocessed ground-truth file (`data/cesop/gabarito_cesop.jsonl`) is included. For instructions on obtaining the original data, see [`data/README.md`](data/README.md).

### Running Experiments

To run the full simulation pipeline from scratch (requires API keys and network access for IBGE):

```bash
# 1. Generate personas from IBGE/SIDRA
uv run python scripts/01_generate_personas.py --count 2000 --output data/personas.jsonl

# 2. Run simulation
uv run python scripts/02_run_simulation.py \
    --survey surveys/survey_percepcao_democracia.yaml \
    --personas data/personas.jsonl

# 3. Analyze results
uv run python scripts/03_analyze_results.py \
    --results-dir data/results/percepcao_democracia \
    --gabarito data/cesop/gabarito_cesop.jsonl
```

To run the evaluation on precomputed results without calling LLM APIs:

```bash
uv run python scripts/03_analyze_results.py \
    --results-dir data/results/percepcao_democracia \
    --gabarito data/cesop/gabarito_cesop.jsonl \
    --gabarito-only
```

Alternatively, you can use the `Makefile` shortcuts:

```bash
make simulate   # Generate personas and run simulation
make analyze    # Compute metrics and generate tables
make paper      # Compile LaTeX paper
```

## Repository Structure

```
silicon-sampling-brazil/
├── config.yaml              # Model providers, SIDRA API, cache, and persona settings
├── surveys/                 # Survey definitions (CESOP questions P02, P03, P04)
├── prompts/                 # Prompt templates for each experimental scenario
│   ├── scenario_1_feature_selection.yaml
│   ├── scenario_2_natural_prose.yaml
│   ├── scenario_3_cot_role_separation.yaml
│   └── scenario_4_few_shot_calibration.yaml
├── data/
│   ├── cesop/               # Pre-processed CESOP ground-truth (gabarito_cesop.jsonl)
│   └── results/             # LLM simulation outputs per model (JSONL)
├── src/
│   ├── infrastructure/      # SIDRA client, LLM adapters (Maritaca, LLaMA), cache
│   ├── persona/             # Synthetic persona generation from IBGE microdata
│   ├── simulation/          # LLM sampling simulation engine
│   └── analysis/            # JSD metrics and benchmark routines
├── scripts/                 # Execution scripts (01, 02, 03)
├── paper/                   # IEEE LaTeX source (main.tex, sections/, figures/)
└── tests/                   # Unit and integration tests
```

## Models Evaluated

| Model | Provider | Type | Origin |
| :--- | :--- | :--- | :--- |
| Sabiá-4 | Maritaca AI | API | Brazilian |
| Sabiazinho-4 | Maritaca AI | API | Brazilian |
| GPT-5-mini | OpenAI | API | Global |
| LLaMA 3.2 3B | Meta | Local (HuggingFace) | Global |

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{aguiar2027simulating,
  title     = {Simulating {Brazilian} Public Opinion with Large Language Models:
               Distributional Evaluation and Prompt Engineering},
  author    = {Aguiar, Lucas Nascimento and Moreira, Matheus dos Santos and
               de Oliveira, Rog{\'e}rio},
  booktitle = {Proc. of International Conference on Artificial Intelligence,
               Computer, Data Sciences and Applications (ACDSA 2027)},
  year      = {2027},
  publisher = {IEEE},
  address   = {Rio de Janeiro, Brazil}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
The CESOP survey data is subject to its own terms (see [`data/README.md`](data/README.md)).
