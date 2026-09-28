"""
Script 01: Generate Synthetic Population Personas from IBGE/SIDRA.

Generates N synthetic individual profiles using stratified probabilistic
sampling grounded in official IBGE microdata (Demographic Census 2022,
PNAD Continuous, PNS, Civil Registry, IPCA). Each persona is characterized
by up to 13 demographic, economic, health, and social attributes.

Usage:
    uv run python scripts/01_generate_personas.py \\
        --count 2000 \\
        --output data/personas.jsonl
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.infrastructure.cache import DistributionCache
from src.infrastructure.sidra import SidraClient
from src.persona import generate_batch
from src.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate synthetic Brazilian population personas from IBGE/SIDRA data."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Number of synthetic personas to generate (default: 2000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/personas.jsonl",
        help="Output JSONL file path for generated personas.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / settings.paths.distributions_db
    db_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.count} synthetic personas from IBGE/SIDRA...")

    with SidraClient() as client, DistributionCache(db_path) as cache:
        personas = generate_batch(client=client, cache=cache, count=args.count)

    logger.info(f"Generated {len(personas)} personas successfully.")

    with open(output_path, "w", encoding="utf-8") as f:
        for persona in personas:
            f.write(persona.model_dump_json() + "\n")

    logger.info(f"Personas saved to: {output_path}")


if __name__ == "__main__":
    main()
