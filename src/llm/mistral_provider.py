"""
Implémentation du provider Mistral AI.

JUSTIFIER: Pourquoi Mistral?
- API simple et rapide
"""

from typing import Optional
import time
import os
from mistralai import Mistral
from dotenv import load_dotenv
from .base_llm import BaseLLMProvider

# Charger variables d'environnement
load_dotenv()

# Coûts Mistral par 1M tokens (prix actuels)
MISTRAL_PRICING = {
    "open-mistral-7b": {"input": 0.25, "output": 0.25},
    "open-mixtral-8x7b": {"input": 0.7, "output": 0.7},
    "mistral-small-latest": {"input": 1.0, "output": 3.0},
    "mistral-medium-latest": {"input": 2.7, "output": 8.1},
    "mistral-large-latest": {"input": 4.0, "output": 12.0},
}


class MistralProvider(BaseLLMProvider):
    """Provider pour l'API Mistral AI."""

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        api_key: str = "",
        temperature: float = 0.7,
        **kwargs,
    ) -> None:
        """
        Initialise le client Mistral.

        Args:
            model_name: Nom du modèle Mistral
            api_key: Clé API (ou depuis .env)
            temperature: Créativité (0-1)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = kwargs.get("max_tokens", 1000)

        # Récupérer la clé API
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(" MISTRAL_API_KEY non trouvée !")

        # Initialiser le client Mistral
        self.client = Mistral(api_key=self.api_key)

    def generate(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Génère une réponse simple."""
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f" Erreur Mistral: {e}")
            raise

    def generate_with_metadata(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Génère une réponse avec métadonnées."""
        start_time = time.time()

        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extraire les tokens
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            # Calculer le coût
            cost = self.calculate_cost(input_tokens, output_tokens)

            return {
                "response": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            print(f" Erreur Mistral: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Compte approximativement les tokens.
        Mistral utilise environ 1 token = 4 caractères pour le français.
        """
        # Approximation simple
        return len(text) // 4

    def get_cost_per_token(self) -> dict:
        """Retourne le coût par token."""
        pricing = MISTRAL_PRICING.get(self.model_name, {"input": 1.0, "output": 3.0})
        # Convertir de par 1M tokens à par token
        return {"input": pricing["input"] / 1_000_000, "output": pricing["output"] / 1_000_000}

    def get_model_info(self) -> dict:
        """Informations sur le modèle."""
        return {
            "name": self.model_name,
            "provider": "mistral",
            "context_window": 32000,  # Mistral a 32k tokens de contexte
            "supports_function_calling": True,
            "supports_json_mode": True,
        }

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calcule le coût d'une requête."""
        costs = self.get_cost_per_token()
        total_cost = (input_tokens * costs["input"]) + (output_tokens * costs["output"])
        return round(total_cost, 6)
