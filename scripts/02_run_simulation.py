"""
Script 02: Run LLM Silicon Sampling Simulation.

Executes the public opinion simulation pipeline: loads synthetic personas
from a JSONL file, runs each LLM model across all survey questions for
the configured number of repetitions (majority-vote evaluation), and saves
the structured results to a JSON output file.

Survey and prompt configuration are defined in the YAML file passed via --survey.
Models are configured in config.yaml (llm.active_models).

Usage:
    uv run python scripts/02_run_simulation.py \\
        --survey surveys/survey_percepcao_democracia.yaml \\
        --personas data/personas.jsonl \\
        --output data/results/percepcao_democracia/simulation_results.json
"""

import argparse
import json
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.persona.models import Persona
from src.simulation.handler import run_simulation
from src.simulation.models import SimulationConfig, Survey, SurveyQuestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_survey_yaml(filepath: str) -> SimulationConfig:
    """Load SimulationConfig from a survey YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cenario = data.get("cenario", {})
    resultados = data.get("resultados", {})
    pesquisas_data = data.get("pesquisas", [])

    surveys = []
    for p in pesquisas_data:
        questions = [
            SurveyQuestion(
                id=str(q["id"]),
                topic=q["topico"],
                text=q["texto"],
                options=q["alternativas"],
            )
            for q in p.get("perguntas", [])
        ]
        surveys.append(Survey(id=str(p["id"]), title=p["titulo"], questions=questions))

    return SimulationConfig(
        personas=cenario.get("personas", 1),
        repetitions=cenario.get("reproducoes", 1),
        models=cenario.get("modelos", []),
        results_path=resultados.get("caminho", "./results"),
        surveys=surveys,
    )


def load_personas(personas_path: Path) -> list[Persona]:
    """Load synthetic personas from a JSONL file."""
    personas = []
    with open(personas_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                personas.append(Persona.model_validate_json(line))
    return personas


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run LLM silicon sampling simulation against a survey definition."
    )
    parser.add_argument(
        "--survey",
        required=True,
        help="Path to the survey YAML configuration file.",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="data/personas.jsonl",
        help="Path to the generated personas JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. Defaults to results_path from YAML config.",
    )
    args = parser.parse_args()

    config = parse_survey_yaml(args.survey)

    personas_path = Path(args.personas)
    if not personas_path.exists():
        logger.error(f"Personas file not found: {personas_path}")
        return

    personas = load_personas(personas_path)
    logger.info(f"Loaded {len(personas)} synthetic personas from {personas_path}")

    results_dir = Path(args.output or config.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    config = config.model_copy(update={"results_path": str(results_dir)})

    logger.info("Starting LLM simulation...")
    results = run_simulation(config=config, personas=personas)

    output_file = results_dir / f"simulation_{Path(args.survey).stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump(mode="json") for r in results], f, ensure_ascii=False, indent=2)

    logger.info(f"Simulation complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
