from src.persona.models import Persona
from .models import SurveyQuestion


def build_persona_system_prompt(persona: Persona) -> str:
    """Build a natural language system prompt from a Persona object.

    Contextualizes the LLM into the persona's socio-economic and demographic
    profile using Brazilian Portuguese, as the survey questions are in Portuguese
    and cultural authenticity requires the persona to reason in its native language.
    """
    demo = persona.demographic
    eco = persona.economic
    health = persona.health
    social = persona.social

    urban_str = "Urbana" if demo.urban else "Rural"

    prompt = (
        f"Você é um cidadão brasileiro com o seguinte perfil:\n"
        f"- Idade: {demo.age_group}\n"
        f"- Gênero: {demo.gender}\n"
        f"- Raça/Cor: {demo.race}\n"
        f"- Região do Brasil: {demo.region} (Zona {urban_str})\n"
        f"- Escolaridade: {eco.education_level}\n"
        f"- Situação de emprego: {eco.employment_status}\n"
        f"- Faixa de renda: {eco.income_per_capita}\n"
        f"- Estado civil: {social.marital_status}\n"
        f"- Religião: {social.religion}\n"
        f"- Autoavaliação de saúde: {health.health_self_assessment}\n\n"
        "Sua tarefa é responder a perguntas de opinião pública como se você fosse essa pessoa. "
        "Considere as condições socioeconômicas e demográficas descritas para formar sua opinião, refletindo "
        "o contexto real do Brasil. Não saia do personagem. Responda de forma sincera baseando-se nas experiências "
        "prováveis de alguém com este exato perfil."
    )
    # Note: The prompt body is intentionally in Brazilian Portuguese because the survey
    # questions are in Portuguese and the persona must reason within its cultural context.
    return prompt
