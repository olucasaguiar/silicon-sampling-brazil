"""
Pacote utilizado como serviço de integração com o serviço de dados agregados do IBGE.
Documentaçao: https://servicodados.ibge.gov.br/api/docs/agregados?versao=3
"""

from .client import SidraClient
from .models import (
    SidraClassificacao,
    SidraLocalidade,
    SidraMetadata,
    SidraNivel,
    SidraResultado,
    SidraSerie,
    SidraVariableResponse,
)
from .query_builder import SidraQueryBuilder

__all__ = [
    "SidraClient",
    "SidraQueryBuilder",
    "SidraMetadata",
    "SidraVariableResponse",
    "SidraResultado",
    "SidraSerie",
    "SidraClassificacao",
    "SidraLocalidade",
    "SidraNivel",
]
