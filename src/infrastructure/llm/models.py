from typing import Literal, Tuple

from pydantic import BaseModel

# Supported model identifiers
ModelID = Literal[
    "sabia-4",
    "sabiazinho-4",
    "gpt-5-mini",
    "llama-3.2-3b",
]


class ModelAnswer(BaseModel):
    answer: Tuple[str, str]  # (key, value) — ex: ("a", "Concordo totalmente")
    explanation: str  # model-generated reasoning for the selected answer
