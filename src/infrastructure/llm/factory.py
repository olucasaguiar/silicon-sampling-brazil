from typing import Callable, Dict, Optional

from src.settings import settings

from .client_base import BaseLLMClient
from .llama_adapter import LlamaAdapter
from .maritaca_adapter import MaritacaAdapter


class LLMFactory:
    """Factory responsible for managing LLM adapter instances.

    Instantiates adapters based on the active_models list in config.yaml.
    Supported adapters: 'maritaca' (Sabiá-4, Sabiazinho-4) and 'llama' (LLaMA 3.2 3B).
    """

    def __init__(self):
        self._registry: Dict[str, Callable[[], BaseLLMClient]] = {}
        for m in settings.llm.active_models:
            if m.adapter == "maritaca":
                self._registry[m.id] = lambda m_id=m.id: MaritacaAdapter(m_id)
            elif m.adapter == "llama":
                self._registry[m.id] = lambda m_id=m.id, hf_id=m.huggingface_id: (
                    LlamaAdapter(m_id, hf_id)
                )

    def provide(self, model: str) -> Optional[BaseLLMClient]:
        """Return a configured adapter instance for the given model ID.

        Returns None if the model is not registered in the factory.
        """
        builder = self._registry.get(model)
        if builder:
            return builder()
        return None
