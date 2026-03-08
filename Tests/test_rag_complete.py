"""
Test complet du pipeline RAG.
"""

import sys
import os
from pathlib import Path
import shutil

# Fix encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    sys.stdout.reconfigure(encoding="utf-8")

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.rag.embeddings import EmbeddingProvider
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader
from src.rag.retriever import Retriever


def test_rag_complete():
    """Test complet du pipeline RAG."""

    print("=" * 60)
    print("TEST COMPLET DU PIPELINE RAG")
    print("=" * 60)

    # Chemin de test temporaire
    test_db_path = ROOT_DIR / "data" / "test_rag_db"

    # Nettoyer si existe déjà
    if test_db_path.exists():
        shutil.rmtree(test_db_path)

    # ============================================================
    # ÉTAPE 1: Initialiser les composants
    # ============================================================
    print("\n[1/6] Initialisation des composants...")

    try:
        # EmbeddingProvider
        embedder = EmbeddingProvider("paraphrase-multilingual-MiniLM-L12-v2")
        print("   ✅ EmbeddingProvider créé")

        # VectorStore
        vector_store = VectorStore(
            persist_path=str(test_db_path), embedding_provider=embedder, collection_name="test_rag"
        )
        print("   ✅ VectorStore créé")

        # DocumentLoader
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        print("   ✅ DocumentLoader créé")

        # Retriever
        retriever = Retriever(vector_store, default_top_k=3)
        print("   ✅ Retriever créé")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    # ============================================================
    # ÉTAPE 2: Charger les catégories de gravité (données intégrées)
    # ============================================================
    print("\n[2/6] Chargement des catégories de gravité...")

    try:
        gravity_docs = loader.load_gravity_categories()
        print(f"   ✅ {len(gravity_docs)} catégories de gravité chargées")

        # Ajouter à la base
        vector_store.add_documents(gravity_docs)
        print("   ✅ Catégories indexées dans la base")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # ============================================================
    # ÉTAPE 3: Charger un PDF s'il existe
    # ============================================================
    print("\n[3/6] Recherche et chargement de PDFs médicaux...")

    pdf_paths = list((ROOT_DIR / "src" / "rag").glob("*.pdf"))

    if pdf_paths:
        for pdf_path in pdf_paths:
            try:
                print(f"   📄 Chargement de {pdf_path.name}...")
                docs = loader.load_from_file(str(pdf_path))
                print(f"      - {len(docs)} pages extraites")

                # Chunker les documents
                chunks = loader.chunk_documents(docs)
                print(f"      - {len(chunks)} chunks créés")

                # Ajouter à la base
                vector_store.add_documents(chunks)
                print(f"      ✅ Indexé dans la base")

            except Exception as e:
                print(f"      ❌ Erreur: {e}")
    else:
        print("   ⚠️ Aucun PDF trouvé dans src/rag/")
        print("   → Ajout de documents médicaux de test...")

        # Ajouter des documents médicaux de test
        test_docs = [
            {
                "text": "La douleur thoracique avec irradiation dans le bras gauche et des sueurs nécessite une prise en charge immédiate. Classification ROUGE.",
                "metadata": {"source": "test", "category": "cardiologie"},
            },
            {
                "text": "Fièvre supérieure à 39°C chez un enfant de moins de 3 mois: urgence pédiatrique, classification ORANGE minimum.",
                "metadata": {"source": "test", "category": "pediatrie"},
            },
            {
                "text": "Constantes vitales normales chez l'adulte: fréquence cardiaque 60-100 bpm, fréquence respiratoire 12-20/min, SpO2 > 95%, tension artérielle systolique 100-140 mmHg.",
                "metadata": {"source": "test", "category": "constantes"},
            },
            {
                "text": "L'entorse de cheville sans fracture associée, patient stable, peut être classée VERT avec orientation vers médecine de ville.",
                "metadata": {"source": "test", "category": "traumatologie"},
            },
        ]
        vector_store.add_documents(test_docs)
        print("   ✅ Documents de test ajoutés")

    # ============================================================
    # ÉTAPE 4: Vérifier les statistiques
    # ============================================================
    print("\n[4/6] Statistiques de la base...")

    try:
        stats = vector_store.get_collection_stats()
        print(f"   ✅ Total documents indexés: {stats['count']}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # ============================================================
    # ÉTAPE 5: Tester les recherches
    # ============================================================
    print("\n[5/6] Test des recherches RAG...")

    test_queries = [
        "patient avec douleur dans la poitrine",
        "enfant qui a de la fièvre",
        "classification urgence vitale",
        "entorse cheville",
    ]

    for query in test_queries:
        print(f"\n   🔍 Requête: '{query}'")
        try:
            # Recherche simple
            results = retriever.retrieve(query, top_k=2)
            print(f"      {len(results)} résultats trouvés")

            for i, doc in enumerate(results, 1):
                score = doc.get("score", 0)
                text = doc.get("text", "")[:80]
                print(f"      [{i}] Score: {score:.4f} | {text}...")

        except Exception as e:
            print(f"      ❌ Erreur: {e}")

    # ============================================================
    # ÉTAPE 6: Tester retrieve_and_format (pour le prompt LLM)
    # ============================================================
    print("\n[6/6] Test du formatage pour le prompt LLM...")

    try:
        context = retriever.retrieve_and_format(
            "patient avec douleur thoracique et essoufflement", top_k=3, max_tokens=500
        )
        print("   ✅ Contexte formaté:")
        print("-" * 50)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 50)

    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # ============================================================
    # BONUS: Test multi-query (plusieurs symptômes)
    # ============================================================
    print("\n[BONUS] Test multi-query (plusieurs symptômes)...")

    try:
        symptoms = ["douleur thoracique", "essoufflement", "sueurs"]
        results = retriever.multi_query_retrieve(symptoms, top_k_per_query=2)
        print(f"   ✅ {len(results)} documents uniques trouvés pour {len(symptoms)} symptômes")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("TESTS RAG TERMINÉS!")
    print("=" * 60)

    # Note: On ne nettoie pas la base pour que l'utilisateur puisse l'inspecter
    print(f"\n💡 Base de test conservée dans: {test_db_path}")
    print("   Pour la supprimer: rmdir /s /q data\\test_rag_db")


if __name__ == "__main__":
    test_rag_complete()
