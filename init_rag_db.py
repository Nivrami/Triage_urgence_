"""
Script pour initialiser la base RAG avec les documents médicaux.
Lance: python init_rag_db.py
"""

import sys
from pathlib import Path

# Fix encodage Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.rag.embeddings import EmbeddingProvider
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader
from src.rag.retriever import Retriever
import shutil


def main():
    print("=" * 60)
    print("   INITIALISATION DE LA BASE RAG")
    print("=" * 60)

    db_path = ROOT_DIR / "data" / "rag_db"

    # Supprimer l'ancienne base si elle existe
    if db_path.exists():
        print(f"\n🗑️  Suppression de l'ancienne base: {db_path}")
        shutil.rmtree(db_path)

    # 1. Créer les composants
    print("\n[1/4] Création des composants RAG...")

    embedding_provider = EmbeddingProvider(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    print(f"   ✅ EmbeddingProvider: {embedding_provider.model_name}")

    vector_store = VectorStore(
        persist_path=str(db_path),
        embedding_provider=embedding_provider,
        collection_name="triage_medical",
    )
    print(f"   ✅ VectorStore: {db_path}")

    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    print(f"   ✅ DocumentLoader")

    # 2. Charger les catégories de gravité
    print("\n[2/4] Chargement des catégories de gravité...")
    categories = loader.load_gravity_categories()
    vector_store.add_documents(categories)
    print(f"   ✅ {len(categories)} catégories indexées")

    # 3. Charger les documents markdown
    print("\n[3/4] Chargement des documents markdown...")
    md_dir = ROOT_DIR / "data" / "rag_document"

    if md_dir.exists():
        all_chunks = []
        for md_file in md_dir.glob("*.md"):
            print(f"   📄 {md_file.name}...")
            docs = loader.load_from_file(str(md_file))
            chunks = loader.chunk_documents(docs)
            all_chunks.extend(chunks)
            print(f"      → {len(chunks)} chunks")

        if all_chunks:
            vector_store.add_documents(all_chunks)
            print(f"   ✅ {len(all_chunks)} chunks markdown indexés")
    else:
        print(f"   ⚠️ Dossier {md_dir} non trouvé")

    # 4. Charger les PDFs médicaux
    print("\n[4/4] Chargement des PDFs médicaux...")
    pdf_dir = ROOT_DIR / "src" / "rag"

    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"   📄 {pdf_file.name}...")
        try:
            docs = loader.load_from_file(str(pdf_file))
            chunks = loader.chunk_documents(docs)
            vector_store.add_documents(chunks)
            print(f"      → {len(chunks)} chunks indexés")
        except Exception as e:
            print(f"      ❌ Erreur: {e}")

    # Stats finales
    print("\n" + "=" * 60)
    stats = vector_store.get_collection_stats()
    print(f"✅ BASE RAG INITIALISÉE")
    print(f"   Total documents: {stats.get('count', 'N/A')}")
    print(f"   Chemin: {db_path}")
    print("=" * 60)

    # Test rapide
    print("\n🔍 Test de recherche...")
    retriever = Retriever(vector_store=vector_store)
    results = retriever.retrieve("douleur thoracique", top_k=2)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r.get('score', 'N/A'):.3f}")
        print(f"    {r.get('text', '')[:100]}...")

    print("\n✅ RAG prêt à l'emploi!")
    print("\n💡 Maintenant, mets à jour chat_interactive.py pour utiliser:")
    print(f'   vector_db_path = project_root / "data" / "rag_db"')


if __name__ == "__main__":
    main()
