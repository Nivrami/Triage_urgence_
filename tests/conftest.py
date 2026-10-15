"""
Fixtures partagées pour les tests du module RAG.
"""

import hashlib

import pytest


class FakeEmbedder:
    """
    Faux embedder déterministe, utilisé dans les tests unitaires.

    Permet de tester VectorStore et le pipeline RAG sans dépendre
    d'un vrai modèle sentence-transformers (pas de téléchargement,
    pas de temps de chargement).
    """

    dim = 8

    def embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255 for byte in digest[: self.dim]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return self.dim

    def get_model_info(self) -> dict:
        return {"model_name": "fake-embedder", "dimension": self.dim, "type": "fake"}


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Embedder factice, rapide et sans dépendance réseau."""
    return FakeEmbedder()
