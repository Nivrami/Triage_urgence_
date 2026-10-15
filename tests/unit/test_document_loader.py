"""
Tests unitaires pour DocumentLoader (chunking, prétraitement, chargement de fichiers).

Aucune dépendance externe : pas de modèle d'embeddings, pas de ChromaDB.
"""

import json

import pytest

from src.rag.document_loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader(chunk_size=500, chunk_overlap=50)


def test_preprocess_text_collapses_whitespace(loader: DocumentLoader) -> None:
    raw = "Douleur   thoracique\n\n\n\nirradiant   au bras"
    cleaned = loader.preprocess_text(raw)

    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_preprocess_text_empty_string_returns_empty(loader: DocumentLoader) -> None:
    assert loader.preprocess_text("") == ""


def test_chunk_document_short_text_not_split(loader: DocumentLoader) -> None:
    document = {"text": "Texte court.", "metadata": {"source": "test"}}

    chunks = loader.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Texte court."
    assert chunks[0]["metadata"]["is_chunked"] is False


def test_chunk_document_long_text_is_split(loader: DocumentLoader) -> None:
    long_text = "Phrase médicale de test. " * 50  # bien plus que chunk_size=500
    document = {"text": long_text, "metadata": {"source": "test"}}

    chunks = loader.chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk["metadata"]["is_chunked"] for chunk in chunks)
    # La métadonnée d'origine doit être préservée dans chaque chunk
    assert all(chunk["metadata"]["source"] == "test" for chunk in chunks)


def test_chunk_document_preserves_original_metadata(loader: DocumentLoader) -> None:
    document = {
        "text": "Texte court.",
        "metadata": {"source": "protocole.pdf", "page": 3},
    }

    chunks = loader.chunk_document(document)

    assert chunks[0]["metadata"]["source"] == "protocole.pdf"
    assert chunks[0]["metadata"]["page"] == 3


def test_chunk_documents_processes_multiple_documents(loader: DocumentLoader) -> None:
    documents = [
        {"text": "Premier document.", "metadata": {"source": "a"}},
        {"text": "Deuxième document.", "metadata": {"source": "b"}},
    ]

    chunks = loader.chunk_documents(documents)

    assert len(chunks) == 2
    sources = {chunk["metadata"]["source"] for chunk in chunks}
    assert sources == {"a", "b"}


def test_load_from_file_raises_on_missing_file(loader: DocumentLoader) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_from_file("fichier_qui_nexiste_pas.txt")


def test_load_from_file_raises_on_unsupported_extension(loader: DocumentLoader, tmp_path) -> None:
    unsupported_file = tmp_path / "data.xyz"
    unsupported_file.write_text("contenu")

    with pytest.raises(ValueError):
        loader.load_from_file(str(unsupported_file))


def test_load_from_txt_file(loader: DocumentLoader, tmp_path) -> None:
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("Douleur abdominale aiguë.", encoding="utf-8")

    documents = loader.load_from_file(str(txt_file))

    assert len(documents) == 1
    assert "Douleur abdominale aiguë" in documents[0]["text"]
    assert documents[0]["metadata"]["type"] == "txt"


def test_load_from_json_file_with_list(loader: DocumentLoader, tmp_path) -> None:
    json_file = tmp_path / "cases.json"
    json_file.write_text(
        json.dumps([{"text": "Cas clinique 1"}, {"text": "Cas clinique 2"}]),
        encoding="utf-8",
    )

    documents = loader.load_from_file(str(json_file))

    assert len(documents) == 2
    assert documents[0]["text"] == "Cas clinique 1"
    assert documents[0]["metadata"]["type"] == "json"


def test_load_gravity_categories_returns_five_levels(loader: DocumentLoader) -> None:
    categories = loader.load_gravity_categories()

    levels = {doc["metadata"]["level"] for doc in categories}
    assert levels == {"ROUGE", "ORANGE", "JAUNE", "VERT", "BLEU"}
    assert len(categories) == 5
