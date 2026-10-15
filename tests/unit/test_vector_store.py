"""
Tests unitaires pour VectorStore et RAGRetriever.

Utilise un faux embedder (fixture `fake_embedder`) : aucun modèle
sentence-transformers n'est chargé, ces tests sont rapides.
"""

import pytest

from src.rag.vector_store import RAGRetriever, VectorStore


@pytest.fixture
def vector_store(tmp_path, fake_embedder) -> VectorStore:
    return VectorStore(
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="test_medical",
        embedder=fake_embedder,
    )


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "content": "Douleur thoracique avec irradiation : classification ROUGE.",
            "metadata": {"source": "protocole.pdf", "title": "CIMU", "section": "cardio"},
        },
        {
            "content": "Entorse de cheville légère : classification VERT.",
            "metadata": {"source": "protocole.pdf", "title": "CIMU", "section": "trauma"},
        },
    ]


def test_new_collection_is_empty(vector_store: VectorStore) -> None:
    stats = vector_store.get_stats()

    assert stats["total_documents"] == 0
    assert stats["collection_name"] == "test_medical"


def test_add_documents_increases_count(vector_store: VectorStore, sample_chunks: list[dict]) -> None:
    vector_store.add_documents(sample_chunks)

    stats = vector_store.get_stats()
    assert stats["total_documents"] == 2


def test_add_documents_with_empty_list_does_nothing(vector_store: VectorStore) -> None:
    vector_store.add_documents([])

    stats = vector_store.get_stats()
    assert stats["total_documents"] == 0


def test_search_returns_requested_number_of_results(
    vector_store: VectorStore, sample_chunks: list[dict]
) -> None:
    vector_store.add_documents(sample_chunks)

    results = vector_store.search("douleur thoracique", n_results=1)

    assert len(results) == 1
    assert "content" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]


def test_search_with_metadata_filter(vector_store: VectorStore, sample_chunks: list[dict]) -> None:
    vector_store.add_documents(sample_chunks)

    results = vector_store.search(
        "douleur", n_results=5, filter_metadata={"section": "cardio"}
    )

    assert len(results) == 1
    assert results[0]["metadata"]["section"] == "cardio"


def test_clear_collection_resets_count(vector_store: VectorStore, sample_chunks: list[dict]) -> None:
    vector_store.add_documents(sample_chunks)
    assert vector_store.get_stats()["total_documents"] == 2

    vector_store.clear_collection()

    assert vector_store.get_stats()["total_documents"] == 0


def test_persistence_across_instances(tmp_path, fake_embedder) -> None:
    db_path = str(tmp_path / "chroma_db")

    store_1 = VectorStore(
        persist_directory=db_path, collection_name="persist_test", embedder=fake_embedder
    )
    store_1.add_documents(
        [{"content": "Texte persistant", "metadata": {"source": "a", "title": "t", "section": "s"}}]
    )

    store_2 = VectorStore(
        persist_directory=db_path, collection_name="persist_test", embedder=fake_embedder
    )

    assert store_2.get_stats()["total_documents"] == 1


class TestRAGRetriever:
    def test_retrieve_context_formats_sources(
        self, vector_store: VectorStore, sample_chunks: list[dict]
    ) -> None:
        vector_store.add_documents(sample_chunks)
        retriever = RAGRetriever(vector_store)

        context = retriever.retrieve_context("douleur thoracique", top_k=1)

        assert "CIMU" in context
        assert "Source 1" in context

    def test_retrieve_context_no_results(self, vector_store: VectorStore) -> None:
        retriever = RAGRetriever(vector_store)

        context = retriever.retrieve_context("requête sans aucun document indexé")

        assert context == "Aucun contexte trouvé."

    def test_retrieve_with_scores_adds_relevance_score(
        self, vector_store: VectorStore, sample_chunks: list[dict]
    ) -> None:
        vector_store.add_documents(sample_chunks)
        retriever = RAGRetriever(vector_store)

        results = retriever.retrieve_with_scores("douleur", top_k=2)

        assert len(results) == 2
        assert all(0 <= r["relevance_score"] <= 1 for r in results)
