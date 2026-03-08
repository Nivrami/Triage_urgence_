"""
Configuration centralisée du projet.
Utilise pydantic-settings pour charger les variables d'environnement.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Configuration centralisée.

    JUSTIFIER: Pourquoi pydantic-settings?
    - Validation automatique des types
    - Chargement .env automatique
    - Valeurs par défaut typées
    """

    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")

    # LLM Configuration
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-3.5-turbo", alias="LLM_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")

    # Paths
    vector_store_path: str = Field(default="data/vectors", alias="VECTOR_STORE_PATH")
    data_path: str = Field(default="data", alias="DATA_PATH")

    # Conversation Settings
    max_conversation_turns: int = Field(default=10, alias="MAX_CONVERSATION_TURNS")

    # Cost tracking (prix par 1000 tokens)
    cost_per_1k_input_tokens: float = Field(default=0.0015)
    cost_per_1k_output_tokens: float = Field(default=0.002)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_llm_config(self) -> dict:
        """Retourne la config LLM sous forme de dict."""
        pass

    def get_embedding_config(self) -> dict:
        """Retourne la config embedding sous forme de dict."""
        pass

    def validate_api_keys(self) -> bool:
        """Vérifie que les API keys nécessaires sont présentes."""
        pass


# Singleton
settings = Settings()
