"""
Tests d'intégration pour EmbeddingProvider avec le vrai modèle sentence-transformers.

Marqués `slow` : ils téléchargent/chargent un modèle réel.
Exécution : pytest -m slow
"""

import pytest

from src.rag.embeddings import EmbeddingProvider

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder() -> EmbeddingProvider:
    return EmbeddingProvider("paraphrase-multilingual-MiniLM-L12-v2")


def test_embedder_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError):
        EmbeddingProvider("modele-qui-nexiste-pas")


def test_embed_text_matches_declared_dimension(embedder: EmbeddingProvider) -> None:
    vector = embedder.embed_text("douleur thoracique intense")

    assert len(vector) == embedder.get_dimension()
    assert all(isinstance(value, float) for value in vector)


def test_embed_batch_returns_one_vector_per_text(embedder: EmbeddingProvider) -> None:
    texts = ["douleur thoracique", "fièvre élevée", "fracture du poignet"]

    vectors = embedder.embed_batch(texts)

    assert len(vectors) == len(texts)
    assert all(len(vector) == embedder.get_dimension() for vector in vectors)


def test_similar_texts_have_closer_embeddings_than_unrelated_ones(
    embedder: EmbeddingProvider,
) -> None:
    import numpy as np

    anchor = np.array(embedder.embed_text("douleur thoracique avec essoufflement"))
    similar = np.array(embedder.embed_text("douleur dans la poitrine et difficulté à respirer"))
    unrelated = np.array(embedder.embed_text("entorse légère de la cheville"))

    distance_similar = np.linalg.norm(anchor - similar)
    distance_unrelated = np.linalg.norm(anchor - unrelated)

    assert distance_similar < distance_unrelated
