"""
Gestion des embeddings pour le RAG.

JUSTIFIER le choix du modèle d'embedding:
- all-MiniLM-L6-v2: Gratuit, rapide, 384 dims, bon pour le français
- text-embedding-3-small (OpenAI): Payant mais meilleur
- ClinicalBERT: Spécialisé médical (mentionné dans les ressources)
- camembert-base: Entraîné sur du français
"""

from typing import Optional


class EmbeddingProvider:
    """Gestion des embeddings textuels."""

    # Modèles supportés avec leurs dimensions
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": {"dim": 384, "type": "sentence-transformers"},
        "paraphrase-multilingual-MiniLM-L12-v2": {"dim": 384, "type": "sentence-transformers"},
        "text-embedding-3-small": {"dim": 1536, "type": "openai"},
        "text-embedding-3-large": {"dim": 3072, "type": "openai"},
    }

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", api_key: Optional[str] = None, **kwargs
    ) -> None:
        """
        Initialise le provider d'embeddings.

        Args:
            model_name: Nom du modèle à utiliser
            api_key: Clé API (pour OpenAI)

        JUSTIFIER ton choix de modèle ici.
        """
        # 1. Sauvegarder les paramètres
        self.model_name = model_name
        self.api_key = api_key

        # 2. Vérifier que le modèle est supporté
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modèle '{model_name}' non supporté. "
                f"Modèles disponibles: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # 3. Récupérer les infos du modèle
        self.model_info = self.SUPPORTED_MODELS[model_name]

        # 4. Charger le modèle selon son type
        if self.model_info["type"] == "sentence-transformers":
            # Modèle local gratuit
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.model_type = "sentence-transformers"

        elif self.model_info["type"] == "openai":
            # Modèle OpenAI (payant)
            if not api_key:
                raise ValueError("Une clé API OpenAI est requise pour ce modèle")
            import openai

            self.client = openai.OpenAI(api_key=api_key)
            self.model_type = "openai"

    def embed_text(self, text: str) -> list[float]:
        """
        Génère l'embedding d'un texte.

        Args:
            text: Texte à encoder

        Returns:
            Vecteur d'embedding
        """
        if self.model_type == "sentence-transformers":
            # Utiliser le modèle local
            import numpy as np

            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        elif self.model_type == "openai":
            # Appel API OpenAI
            response = self.client.embeddings.create(model=self.model_name, input=text)
            return response.data[0].embedding

        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Génère les embeddings pour plusieurs textes.

        Plus efficace que d'appeler embed_text en boucle.

        Args:
            texts: Liste de textes

        Returns:
            Liste de vecteurs
        """
        if self.model_type == "sentence-transformers":
            # Encoder tous les textes en une seule fois
            import numpy as np

            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]

        elif self.model_type == "openai":
            # OpenAI peut traiter plusieurs textes en un seul appel
            response = self.client.embeddings.create(model=self.model_name, input=texts)
            return [item.embedding for item in response.data]

        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")

    def get_dimension(self) -> int:
        """Retourne la dimension des vecteurs."""
        return self.model_info["dim"]

    def get_model_info(self) -> dict:
        """Informations sur le modèle."""
        return {
            "model_name": self.model_name,
            "dimension": self.model_info["dim"],
            "type": self.model_type,
        }

    def _load_sentence_transformer(self, model_name: str) -> None:
        """Charge un modèle sentence-transformers."""
        pass

    def _load_openai_embeddings(self, api_key: str) -> None:
        """Configure les embeddings OpenAI."""
        pass
