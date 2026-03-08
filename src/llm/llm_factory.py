from src.llm.base_llm import BaseLLMProvider
from .mistral_provider import MistralProvider


class LLMFactory:
    """Factory pour créer le bon provider LLM."""

    _providers: dict[str, type] = {
        "mistral": MistralProvider,
    }

    @classmethod
    def create(cls, provider: str, model_name: str, api_key: str = "", **kwargs) -> BaseLLMProvider:
        """Crée le provider approprié."""
        if provider not in cls._providers:
            raise ValueError(
                f"Provider '{provider}' non supporté. Disponibles: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider]
        return provider_class(model_name=model_name, api_key=api_key, **kwargs)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Retourne la liste des providers disponibles."""
        return list(cls._providers.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Enregistre un nouveau provider."""
        cls._providers[name] = provider_class

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Retourne le modèle par défaut."""
        defaults = {"openai": "gpt-3.5-turbo", "mistral": "mistral-small-latest"}
        return defaults.get(provider, "")
