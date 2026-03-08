"""
Validations personnalisées.
"""

from typing import Any


def validate_patient_data(data: dict) -> tuple[bool, list[str]]:
    """
    Valide les données d'un patient.

    Args:
        data: Dictionnaire de données patient

    Returns:
        (is_valid, list_of_errors)
    """
    pass


def validate_constantes(constantes: dict) -> tuple[bool, list[str]]:
    """
    Valide les constantes vitales.

    Vérifie que les valeurs sont dans des plages réalistes:
    - FC: 30-250 bpm
    - FR: 5-60 /min
    - SpO2: 50-100%
    - TA: 50-250 mmHg
    - Température: 32-43°C

    Returns:
        (is_valid, list_of_errors)
    """
    pass


def validate_gravity_level(level: str) -> bool:
    """Vérifie que le niveau de gravité est valide."""
    pass


def validate_conversation_format(messages: list[dict]) -> tuple[bool, list[str]]:
    """Valide le format des messages de conversation."""
    pass


def sanitize_user_input(text: str) -> str:
    """
    Nettoie l'input utilisateur.

    - Supprime les caractères dangereux
    - Limite la longueur
    - Normalise les espaces
    """
    pass


def validate_api_response(response: dict, expected_fields: list[str]) -> bool:
    """Valide une réponse API."""
    pass
