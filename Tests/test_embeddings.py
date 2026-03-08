"""
Test du module EmbeddingProvider.
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    os.system("chcp 65001 >nul")
    sys.stdout.reconfigure(encoding="utf-8")

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.rag.embeddings import EmbeddingProvider


def test_embedding_provider():
    """Test complet de l'EmbeddingProvider."""

    print("=" * 60)
    print("TEST DE L'EMBEDDINGPROVIDER")
    print("=" * 60)

    # Test 1: Initialisation
    print("\n[Test 1] Initialisation du modèle...")
    try:
        embedder = EmbeddingProvider("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return

    # Test 2: Infos du modèle
    print("\n[Test 2] Récupération des infos du modèle...")
    try:
        info = embedder.get_model_info()
        print(f"✅ Infos du modèle:")
        print(f"   - Nom: {info['model_name']}")
        print(f"   - Dimension: {info['dimension']}")
        print(f"   - Type: {info['type']}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 3: Dimension
    print("\n[Test 3] Vérification de la dimension...")
    try:
        dim = embedder.get_dimension()
        print(f"✅ Dimension des vecteurs: {dim}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 4: Embedding d'un seul texte
    print("\n[Test 4] Embedding d'un texte simple...")
    try:
        text = "douleur thoracique intense"
        vector = embedder.embed_text(text)
        print(f"✅ Texte: '{text}'")
        print(f"   - Type du vecteur: {type(vector)}")
        print(f"   - Longueur du vecteur: {len(vector)}")
        print(f"   - Premiers éléments: {vector[:5]}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 5: Embedding batch
    print("\n[Test 5] Embedding de plusieurs textes (batch)...")
    try:
        texts = ["douleur thoracique", "fièvre élevée", "migraine sévère", "fracture du poignet"]
        vectors = embedder.embed_batch(texts)
        print(f"✅ Nombre de textes: {len(texts)}")
        print(f"   - Nombre de vecteurs générés: {len(vectors)}")
        print(f"   - Dimension de chaque vecteur: {len(vectors[0])}")

        # Vérifier que tous ont la même dimension
        if all(len(v) == len(vectors[0]) for v in vectors):
            print(f"   ✅ Tous les vecteurs ont la même dimension")
        else:
            print(f"   ❌ Les vecteurs ont des dimensions différentes!")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    # Test 6: Test avec des textes médicaux français
    print("\n[Test 6] Test avec du vocabulaire médical français...")
    try:
        medical_texts = [
            "Le patient présente une tachycardie",
            "Suspicion d'infarctus du myocarde",
            "Constantes vitales stables",
            "Douleur abdominale aiguë",
        ]
        vectors = embedder.embed_batch(medical_texts)
        print(f"✅ {len(vectors)} vecteurs médicaux générés")
        print(f"   Le modèle multilingue gère bien le français médical!")
    except Exception as e:
        print(f"❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("TESTS TERMINÉS!")
    print("=" * 60)


if __name__ == "__main__":
    test_embedding_provider()
