from typing import Dict, List
from pydantic import BaseModel
from src.persona.models import Persona
from src.infrastructure.llm.models import ModelAnswer


class SurveyQuestion(BaseModel):
    """A survey question with multiple-choice options."""

    id: str
    topic: str
    text: str
    options: Dict[str, str]  # {"a": "Concordo totalmente", "b": "Discordo", ...}


class FormResults(BaseModel):
    """Response distribution for a single question.
    Key: option id (letter), Value: proportion (0.0 to 1.0)."""

    distribution: Dict[str, float]  # {"a": 0.4, "b": 0.4, "c": 0.2}


class FormResponse(BaseModel):
    """Result of a persona answering a question N times (majority-vote repetitions)."""

    question: str
    options: Dict[str, str]
    answers: List[ModelAnswer]
    result: FormResults


class PersonaSimulationResult(BaseModel):
    """Complete simulation result for a single persona across all survey questions."""

    persona: Persona
    responses: List[FormResponse]


class Survey(BaseModel):
    """A complete survey instrument with multiple questions."""

    id: str
    title: str
    questions: List[SurveyQuestion]


class SimulationConfig(BaseModel):
    """Simulation scenario configuration, loaded from the survey YAML file."""

    personas: int
    repetitions: int  # number of repetitions per persona/question for majority vote
    models: List[str]
    results_path: str
    surveys: List[Survey]
