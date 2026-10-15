"""
Test d'intégration bout-en-bout du pipeline RAG avec le vrai modèle d'embeddings.

Marqué `slow` : charge un modèle réel et interroge une vraie base ChromaDB.
Exécution : pytest -m slow
"""

import pytest

from src.rag.document_loader import DocumentLoader
from src.rag.vector_store import RAGRetriever, VectorStore

pytestmark = pytest.mark.slow


@pytest.fixture
def populated_vector_store(tmp_path) -> VectorStore:
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma_db"),
        collection_name="integration_test",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    )
    loader = DocumentLoader()
    # add_documents() ne conserve que content/source/title/section/chunk_id :
    # on reflète "level" dans le texte pour pouvoir vérifier la pertinence du
    # résultat sans dépendre d'une métadonnée non conservée.
    gravity_docs = [
        {"content": doc["text"], "metadata": {"source": "gravite", "title": doc["metadata"]["level"]}}
        for doc in loader.load_gravity_categories()
    ]
    store.add_documents(gravity_docs)
    return store


def test_search_finds_relevant_gravity_category(populated_vector_store: VectorStore) -> None:
    results = populated_vector_store.search("arrêt cardiaque, détresse vitale", n_results=1)

    assert len(results) == 1
    assert results[0]["metadata"]["title"] == "ROUGE"


def test_retriever_context_contains_most_relevant_document(
    populated_vector_store: VectorStore,
) -> None:
    retriever = RAGRetriever(populated_vector_store)

    context = retriever.retrieve_context("entorse légère, symptômes mineurs", top_k=1)

    assert "VERT" in context
