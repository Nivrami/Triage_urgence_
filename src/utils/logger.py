"""
Configuration du logging.

Utilise loguru pour un logging simple et efficace.
"""

from typing import Optional


def setup_logger(
    level: str = "INFO", log_file: Optional[str] = None, rotation: str = "10 MB"
) -> None:
    """
    Configure le logger global.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Fichier de log (optionnel)
        rotation: Rotation du fichier
    """
    pass


def get_logger(name: str):
    """
    Retourne un logger nommé.

    Args:
        name: Nom du module (généralement __name__)

    Returns:
        Instance de logger
    """
    pass


class LoggerContext:
    """
    Context manager pour logger les opérations avec timing.

    Usage:
        with LoggerContext("triage_session"):
            # code
    """

    def __init__(self, operation_name: str) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
