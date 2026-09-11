# Data Directory

This directory contains empirical reference data and LLM simulation results used in the paper.

---

## `cesop/gabarito_cesop.jsonl`

Pre-processed ground-truth derived from the CESOP/IPEC survey
**"Percepção dos Brasileiros acerca da Democracia"** (Study 04832, September 2023).

Each line is a JSON object with the coded responses for a single respondent:

```json
{"id": 1, "P02": ["a", "i"], "P03": ["b", "c"], "P04": "b"}
```

Fields:

- `id` — respondent identifier (from the original `ID_Ipec` field)
- `P02` — list of selected policy priority alternatives (up to 3, letters a–l)
- `P03` — list of selected fake news countermeasure alternatives (up to 6, letters a–f)
- `P04` — selected political participation alternative (single choice, letters a–c)

> **Note:** This file is a derived, coded representation of the original survey
> and does not contain any personally identifiable information.

---

## Obtaining the Raw CESOP Data (`.SAV` file)

The raw SPSS file (`04832.SAV`) is **not included** in this repository due to licensing restrictions.
To obtain it:

1. Visit the CESOP data portal: [https://www.cesop.unicamp.br/v3/portal/estudos/04832](https://www.cesop.unicamp.br/v3/portal/estudos/04832)
2. Register for a free account (if required) and download **Study 04832**.
3. Place the file at: `data/raw/04832.SAV`

The analysis script (`scripts/03_analyze_results.py`) uses `--gabarito-only` mode
by default (using `data/cesop/gabarito_cesop.jsonl`), so the raw `.SAV` file
is **not required** to reproduce the paper's results.

---

## `results/percepcao_democracia/`

Final LLM simulation output files for each model evaluated in the paper.
Each JSONL file contains the batch API responses (P02, P03, P04) for 2,000 synthetic personas
× 5 repetitions, using the Scenario 4 prompt configuration.

| File | Model |
| :--- | :--- |
| `batch_result-sabia-4.jsonl` | Sabiá-4 (Maritaca AI) |
| `batch_result_sabiazinho-4.jsonl` | Sabiazinho-4 (Maritaca AI) |
| `batch_result_gpt-5-mini.jsonl` | GPT-5-mini (OpenAI) |
| `batch-result-llama.jsonl` | LLaMA 3.2 3B (Meta, local) |
| `analise_comparativa_llms.csv` | Comparative JSD analysis summary table |

These files are the direct inputs to `scripts/03_analyze_results.py`.
