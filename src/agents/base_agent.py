"""
Classe abstraite pour tous les agents.

"""

from abc import ABC, abstractmethod
import json
from typing import Any, Optional

from ..llm.base_llm import BaseLLMProvider
from ..models.conversation import ConversationHistory


class BaseAgent(ABC):
    """Classe abstraite pour tous les agents du système."""

    def __init__(
        self, llm_provider: BaseLLMProvider, system_prompt: str = "", name: str = "Agent"
    ) -> None:
        """
        Initialise l'agent.

        Args:
            llm_provider: Provider LLM à utiliser
            system_prompt: Prompt système définissant le comportement
            name: Nom de l'agent (pour le logging)
        """
        self.llm = llm_provider
        self.system_prompt = system_prompt
        self.name = name

    @abstractmethod
    def run(self, input_data: Any) -> dict:
        """
        Exécute l'agent avec les données d'entrée.

        Args:
            input_data: Données d'entrée

        Returns:
            Résultat de l'exécution
        """
        pass

    def _build_messages(
        self, user_input: str, history: Optional[ConversationHistory] = None
    ) -> list[dict]:
        """
        Construit la liste des messages pour le LLM.

        Args:
            user_input: Message utilisateur actuel
            history: Historique de conversation optionnel

        Returns:
            Liste de messages au format LLM
        """
        messages = []

        # Ajouter le system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Ajouter l'historique
        if history:
            for msg in history.messages:
                messages.append(msg.to_llm_format())

        # Ajouter le message actuel
        messages.append({"role": "user", "content": user_input})

        return messages

    def _parse_response(self, response: str) -> dict:
        """
        Parse la réponse du LLM.

        Par défaut, retourne simplement la réponse.
        Override cette méthode pour un parsing custom.

        Args:
            response: Réponse brute du LLM

        Returns:
            Réponse parsée
        """
        return {"response": response}

    def _extract_json_from_response(self, response: str) -> dict:
        """
        Extrait le JSON d'une réponse.
        Utile quand le LLM doit retourner du JSON.

        Args:
            response: Réponse contenant du JSON

        Returns:
            dict: JSON parsé

        Raises:
            json.JSONDecodeError: Si le JSON est invalide
        """
        # Nettoyer les markdown code blocks
        response_clean = response.strip()

        # Retirer ```json au début
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        elif response_clean.startswith("```"):
            response_clean = response_clean[3:]

        # Retirer ``` à la fin
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]

        # Parser le JSON
        try:
            return json.loads(response_clean.strip())
        except json.JSONDecodeError as e:
            print(f"[ERREUR] Parsing JSON pour {self.name}: {e}")
            print(f"Réponse brute: {response[:200]}...")
            raise

    def get_system_prompt(self) -> str:
        """Retourne le prompt système."""
        return self.system_prompt

    def update_system_prompt(self, new_prompt: str) -> None:
        """Met à jour le prompt système."""
        self.system_prompt = new_prompt
