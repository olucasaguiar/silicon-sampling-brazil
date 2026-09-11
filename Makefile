.PHONY: help simulate analyze paper test lint

PYTHON = uv run python
SURVEY = surveys/survey_percepcao_democracia.yaml

help:
	@echo "Available targets:"
	@echo "  make simulate   - Generate personas and run LLM simulation"
	@echo "  make analyze    - Compute JSD metrics and generate results table"
	@echo "  make paper      - Compile LaTeX paper to PDF"
	@echo "  make test       - Run all unit tests"
	@echo "  make lint       - Run ruff linter"

simulate:
	$(PYTHON) scripts/01_generate_personas.py --count 2000 --output data/personas.jsonl
	$(PYTHON) scripts/02_run_simulation.py --survey $(SURVEY) --personas data/personas.jsonl

analyze:
	$(PYTHON) scripts/03_analyze_results.py \
		--results-dir data/results/percepcao_democracia \
		--gabarito data/cesop/gabarito_cesop.jsonl

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ scripts/ tests/
