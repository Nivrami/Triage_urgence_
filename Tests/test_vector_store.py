"""
Test du module VectorStore.
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


def test_vector_store():
    """Test complet du VectorStore."""

    print("=" * 60)
    print("TEST DU VECTORSTORE")
    print("=" * 60)

    # Chemin de test temporaire
    test_db_path = ROOT_DIR / "data" / "test_chroma_db"

    # Nettoyer si existe déjà
    if test_db_path.exists():
        shutil.rmtree(test_db_path)

    # Test 1: Initialisation
    print("\n[Test 1] Initialisation de l'EmbeddingProvider...")
    try:
        embedder = EmbeddingProvider("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ EmbeddingProvider créé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    print("\n[Test 2] Initialisation du VectorStore...")
    try:
        vector_store = VectorStore(
            persist_path=str(test_db_path),
            embedding_provider=embedder,
            collection_name="test_medical",
        )
        print(f"✅ VectorStore créé dans {test_db_path}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    # Test 3: Statistiques initiales
    print("\n[Test 3] Statistiques de la collection vide...")
    try:
        stats = vector_store.get_collection_stats()
        print(f"✅ Stats: {stats}")
        print(f"   - Nombre de documents: {stats['count']}")
        print(f"   - Nom de la collection: {stats['name']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Ajout de documents médicaux
    print("\n[Test 4] Ajout de documents médicaux...")
    try:
        medical_docs = [
            {
                "text": "Douleur thoracique avec irradiation nécessite une classification ROUGE et une prise en charge immédiate.",
                "metadata": {
                    "source": "protocole_CIMU.pdf",
                    "page": 5,
                    "category": "urgence_vitale",
                },
            },
            {
                "text": "Fièvre supérieure à 39°C chez un enfant de moins de 3 mois constitue une urgence pédiatrique.",
                "metadata": {"source": "protocole_CIMU.pdf", "page": 12, "category": "pediatrie"},
            },
            {
                "text": "Constantes vitales normales adulte: FC 60-100 bpm, FR 12-20, SpO2 > 95%, TA systolique 100-140 mmHg.",
                "metadata": {"source": "constantes.pdf", "page": 1, "category": "reference"},
            },
            {
                "text": "Traumatisme crânien avec perte de connaissance requiert un examen neurologique et un scanner cérébral.",
                "metadata": {
                    "source": "protocole_CIMU.pdf",
                    "page": 18,
                    "category": "traumatologie",
                },
            },
            {
                "text": "Douleur abdominale aiguë avec défense nécessite une consultation chirurgicale urgente.",
                "metadata": {"source": "protocole_CIMU.pdf", "page": 25, "category": "chirurgie"},
            },
        ]

        vector_store.add_documents(medical_docs)
        print(f"✅ {len(medical_docs)} documents ajoutés à la base")
    except Exception as e:
        print(f"❌ Erreur lors de l'ajout: {e}")
        return

    # Test 5: Statistiques après ajout
    print("\n[Test 5] Statistiques après ajout...")
    try:
        stats = vector_store.get_collection_stats()
        print(f"✅ Nombre de documents dans la base: {stats['count']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Recherche par similarité
    print("\n[Test 6] Recherche par similarité...")
    test_queries = [
        "patient avec douleur dans la poitrine",
        "enfant qui a de la fièvre",
        "valeurs normales du rythme cardiaque",
        "choc à la tête",
    ]

    for query in test_queries:
        print(f"\n   Requête: '{query}'")
        try:
            results = vector_store.search(query, top_k=2)
            print(f"   ✅ {len(results)} résultats trouvés:")
            for i, doc in enumerate(results, 1):
                print(f"      [{i}] Score: {doc['score']:.4f}")
                print(f"          Texte: {doc['text'][:80]}...")
                print(f"          Catégorie: {doc['metadata'].get('category', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

    # Test 7: Recherche avec filtre de métadonnées
    print("\n[Test 7] Recherche avec filtre (catégorie='urgence_vitale')...")
    try:
        results = vector_store.search(
            "patient urgent", top_k=5, filter_metadata={"category": "urgence_vitale"}
        )
        print(f"✅ {len(results)} résultats trouvés avec filtre")
        for doc in results:
            print(f"   - {doc['text'][:60]}...")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 8: Ajout de textes simples
    print("\n[Test 8] Ajout de textes simples sans métadonnées...")
    try:
        simple_texts = [
            "L'IOA doit poser des questions précises sur les symptômes.",
            "La classification par couleur facilite la priorisation des patients.",
        ]
        vector_store.add_texts(simple_texts)
        stats = vector_store.get_collection_stats()
        print(f"✅ Textes ajoutés. Total documents: {stats['count']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 9: Persistance
    print("\n[Test 9] Vérification de la persistance...")
    try:
        # Créer un nouveau VectorStore pointant vers le même chemin
        vector_store2 = VectorStore(
            persist_path=str(test_db_path),
            embedding_provider=embedder,
            collection_name="test_medical",
        )
        stats2 = vector_store2.get_collection_stats()
        print(f"✅ Données persistées! {stats2['count']} documents récupérés")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("TESTS TERMINÉS!")
    print("=" * 60)

    # Nettoyage
    print("\n[Nettoyage] Suppression de la base de test...")
    try:
        if test_db_path.exists():
            shutil.rmtree(test_db_path)
            print("✅ Base de test supprimée")
    except Exception as e:
        print(f"⚠️ Impossible de supprimer: {e}")


if __name__ == "__main__":
    test_vector_store()
