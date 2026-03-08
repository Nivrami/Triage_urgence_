"""
Interface abstraite pour les providers LLM.

JUSTIFIER: Pourquoi une abstraction?
- Permet de changer de provider sans modifier le code métier
- Facilite les tests (mock)
- Principe SOLID: Dependency Inversion
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """Interface abstraite pour tous les providers LLM."""

    @abstractmethod
    def __init__(self, model_name: str, api_key: str, **kwargs) -> None:
        """
        Initialise le provider.

        Args:
            model_name: Nom du modèle à utiliser
            api_key: Clé API
            **kwargs: Paramètres additionnels (temperature, etc.)
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Génère une réponse à partir des messages.

        Args:
            messages: Liste de messages au format [{"role": ..., "content": ...}]
            temperature: Température de génération
            max_tokens: Nombre max de tokens en sortie

        Returns:
            La réponse générée (string)
        """
        pass

    @abstractmethod
    def generate_with_metadata(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Génère une réponse avec métadonnées (tokens, coût).

        Returns:
            {
                "response": str,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "cost": float,
                "latency_ms": float
            }
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Compte le nombre de tokens dans un texte.

        Args:
            text: Texte à analyser

        Returns:
            Nombre de tokens
        """
        pass

    @abstractmethod
    def get_cost_per_token(self) -> dict:
        """
        Retourne le coût par token pour ce modèle.

        Returns:
            {"input": float, "output": float} (coût par token)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Retourne les informations sur le modèle.

        Returns:
            {"name": str, "provider": str, "context_window": int, ...}
        """
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calcule le coût d'une requête.

        Args:
            input_tokens: Nombre de tokens en entrée
            output_tokens: Nombre de tokens en sortie

        Returns:
            Coût en dollars
        """
        costs = self.get_cost_per_token()
        total_cost = (input_tokens * costs["input"]) + (output_tokens * costs["output"])
        return round(total_cost, 6)
