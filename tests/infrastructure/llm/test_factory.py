"""Tests for LLMFactory — covers the 4 models evaluated in the paper."""

from src.infrastructure.llm import LLMFactory
from src.infrastructure.llm.maritaca_adapter import MaritacaAdapter
from src.infrastructure.llm.llama_adapter import LlamaAdapter


def test_llm_factory_provide_maritaca_sabia4(monkeypatch):
    monkeypatch.setenv("MARITACA_API_KEY", "fake-key")
    factory = LLMFactory()
    client = factory.provide("sabia-4")
    assert isinstance(client, MaritacaAdapter)
    assert client.model_id == "sabia-4"


def test_llm_factory_provide_maritaca_sabiazinho4(monkeypatch):
    monkeypatch.setenv("MARITACA_API_KEY", "fake-key")
    factory = LLMFactory()
    client = factory.provide("sabiazinho-4")
    assert isinstance(client, MaritacaAdapter)
    assert client.model_id == "sabiazinho-4"


def test_llm_factory_provide_llama():
    factory = LLMFactory()
    client = factory.provide("llama-3.2-3b")
    assert isinstance(client, LlamaAdapter)
    assert client.huggingface_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_llm_factory_provide_invalid_returns_none():
    factory = LLMFactory()
    client = factory.provide("model-not-in-paper")
    assert client is None
