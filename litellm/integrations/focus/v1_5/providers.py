"""FOCUS participating-entity names for LiteLLM provider slugs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class ProviderEntities:
    service_provider_name: str
    host_provider_name: str
    service_name: str


_KNOWN_PROVIDERS: Final = MappingProxyType(
    {
        "openai": ProviderEntities("OpenAI", "OpenAI", "OpenAI API"),
        "text-completion-openai": ProviderEntities("OpenAI", "OpenAI", "OpenAI API"),
        "anthropic": ProviderEntities("Anthropic", "Anthropic", "Claude API"),
        "azure": ProviderEntities("Microsoft", "Microsoft", "Azure OpenAI"),
        "azure_ai": ProviderEntities("Microsoft", "Microsoft", "Azure AI Foundry"),
        "bedrock": ProviderEntities("AWS", "AWS", "Amazon Bedrock"),
        "sagemaker": ProviderEntities("AWS", "AWS", "Amazon SageMaker"),
        "vertex_ai": ProviderEntities("Google", "Google", "Vertex AI"),
        "gemini": ProviderEntities("Google", "Google", "Gemini API"),
        "mistral": ProviderEntities("Mistral AI", "Mistral AI", "Mistral API"),
        "cohere": ProviderEntities("Cohere", "Cohere", "Cohere API"),
        "groq": ProviderEntities("Groq", "Groq", "GroqCloud"),
        "deepseek": ProviderEntities("DeepSeek", "DeepSeek", "DeepSeek API"),
        "xai": ProviderEntities("xAI", "xAI", "xAI API"),
        "openrouter": ProviderEntities("OpenRouter", "OpenRouter", "OpenRouter"),
        "together_ai": ProviderEntities("Together AI", "Together AI", "Together AI"),
        "fireworks_ai": ProviderEntities("Fireworks AI", "Fireworks AI", "Fireworks AI"),
    }
)

SELF_HOSTED_PROVIDERS: Final = frozenset(
    ("hosted_vllm", "vllm", "ollama", "ollama_chat", "lm_studio", "llamafile", "triton", "xinference")
)


def provider_entities(slug: str | None, *, operator_name: str) -> ProviderEntities:
    """Map a custom_llm_provider slug to FOCUS entities; self-hosted backends are provided by the operator."""
    if not slug:
        return ProviderEntities(operator_name, operator_name, "LiteLLM")
    if slug in SELF_HOSTED_PROVIDERS:
        return ProviderEntities(operator_name, operator_name, slug)
    return _KNOWN_PROVIDERS.get(slug) or ProviderEntities(slug, slug, slug)


__all__ = ("SELF_HOSTED_PROVIDERS", "ProviderEntities", "provider_entities")
