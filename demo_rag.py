"""
Demo interactive du module RAG.

Lance ce script pour comprendre comment fonctionne le RAG:
    python demo_rag.py

Le script te guide étape par étape.
"""

import sys
from pathlib import Path

# Fix encodage Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ajouter le chemin du projet
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def pause():
    """Attend que l'utilisateur appuie sur Entrée."""
    input("\n👉 Appuie sur Entrée pour continuer...")
    print()


def main():
    print("=" * 60)
    print("   DEMO INTERACTIVE DU MODULE RAG")
    print("=" * 60)
    print()
    print("Ce script va te montrer comment fonctionne le RAG")
    print("(Retrieval-Augmented Generation) étape par étape.")
    pause()

    # =========================================================
    # ÉTAPE 1 : Les Embeddings
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 1 : LES EMBEDDINGS")
    print("=" * 60)
    print()
    print("Les embeddings transforment du texte en vecteurs numériques.")
    print("Des textes similaires auront des vecteurs proches.")
    print()
    print("Exemple: 'douleur thoracique' et 'mal à la poitrine'")
    print("         auront des vecteurs similaires.")
    pause()

    print("Chargement du modèle d'embeddings...")
    from src.rag.embeddings import EmbeddingProvider

    embedding_provider = EmbeddingProvider(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    print(f"✅ Modèle chargé: {embedding_provider.model_name}")
    print(f"   Dimension des vecteurs: {embedding_provider.get_dimension()}")
    pause()

    # Démonstration des embeddings
    print("Testons avec quelques phrases:")
    phrases = [
        "Le patient a mal à la poitrine",
        "Douleur thoracique intense",
        "J'ai faim, je veux manger une pizza",
    ]

    embeddings = embedding_provider.embed_batch(phrases)

    for i, phrase in enumerate(phrases):
        vec = embeddings[i][:5]  # Premiers 5 éléments
        print(f"\n'{phrase}'")
        print(f"   → Vecteur (5 premiers): {[round(v, 3) for v in vec]}...")

    # Calculer la similarité
    import numpy as np

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"\n📊 Similarité entre phrase 1 et 2: {sim_1_2:.3f} (proches!)")
    print(f"📊 Similarité entre phrase 1 et 3: {sim_1_3:.3f} (différentes)")
    pause()

    # =========================================================
    # ÉTAPE 2 : Le VectorStore
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 2 : LE VECTORSTORE (ChromaDB)")
    print("=" * 60)
    print()
    print("Le VectorStore stocke les vecteurs et permet de chercher")
    print("les documents les plus similaires à une requête.")
    pause()

    from src.rag.vector_store import VectorStore
    import shutil

    # Nettoyer la base de démo si elle existe
    demo_db_path = ROOT_DIR / "data" / "demo_rag_db"
    if demo_db_path.exists():
        shutil.rmtree(demo_db_path)

    vector_store = VectorStore(
        persist_path=str(demo_db_path),
        embedding_provider=embedding_provider,
        collection_name="demo_medical",
    )
    print(f"✅ VectorStore créé dans: {demo_db_path}")
    pause()

    # Ajouter des documents manuellement
    print("Ajoutons quelques documents médicaux à la base:")
    documents_demo = [
        {
            "text": "La douleur thoracique peut indiquer un infarctus du myocarde. Signes: douleur oppressive, irradiation bras gauche, sueurs.",
            "metadata": {"source": "cardiologie", "urgence": "haute"},
        },
        {
            "text": "La fièvre chez l'enfant nécessite une surveillance. Au-delà de 38.5°C, donner du paracétamol. Consulter si fièvre > 3 jours.",
            "metadata": {"source": "pédiatrie", "urgence": "moyenne"},
        },
        {
            "text": "L'entorse de cheville se traite par repos, glace, compression et élévation (protocole RICE). Immobilisation si nécessaire.",
            "metadata": {"source": "traumatologie", "urgence": "basse"},
        },
        {
            "text": "La dyspnée aiguë (difficulté à respirer) est une urgence. Causes: asthme, embolie pulmonaire, insuffisance cardiaque.",
            "metadata": {"source": "pneumologie", "urgence": "haute"},
        },
        {
            "text": "Les céphalées (maux de tête) sont souvent bénignes. Attention aux signes de gravité: raideur nuque, vomissements, confusion.",
            "metadata": {"source": "neurologie", "urgence": "variable"},
        },
    ]

    for doc in documents_demo:
        print(f"   📄 {doc['metadata']['source']}: {doc['text'][:50]}...")

    vector_store.add_documents(documents_demo)
    print(f"\n✅ {len(documents_demo)} documents indexés!")
    pause()

    # =========================================================
    # ÉTAPE 3 : La Recherche
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 3 : LA RECHERCHE SÉMANTIQUE")
    print("=" * 60)
    print()
    print("Maintenant on peut chercher des documents par similarité.")
    print("La recherche comprend le SENS, pas juste les mots-clés.")
    pause()

    # Recherches de démo
    requetes = ["j'ai mal au coeur", "mon enfant a de la température", "je me suis tordu le pied"]

    for requete in requetes:
        print(f"\n🔍 Requête: '{requete}'")
        print("-" * 40)

        resultats = vector_store.search(requete, top_k=2)

        for i, r in enumerate(resultats, 1):
            score = r.get("score", 0)
            source = r.get("metadata", {}).get("source", "inconnu")
            texte = r.get("text", "")[:80]
            print(f"   [{i}] Score: {score:.3f} | {source}")
            print(f"       {texte}...")

        pause()

    # =========================================================
    # ÉTAPE 4 : Le Retriever
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 4 : LE RETRIEVER")
    print("=" * 60)
    print()
    print("Le Retriever orchestre la recherche et formate les résultats")
    print("pour les injecter dans le prompt du LLM.")
    pause()

    from src.rag.retriever import Retriever

    retriever = Retriever(vector_store=vector_store, default_top_k=3)
    print("✅ Retriever créé")
    pause()

    # Démonstration retrieve_and_format
    print("Exemple: formater le contexte pour un prompt LLM")
    print("-" * 40)

    contexte = retriever.retrieve_and_format(
        query="patient avec douleur poitrine et essoufflement", top_k=3, max_tokens=500
    )

    print("Contexte formaté pour le LLM:")
    print("=" * 40)
    print(contexte)
    print("=" * 40)
    pause()

    # =========================================================
    # ÉTAPE 5 : Le DocumentLoader
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 5 : LE DOCUMENTLOADER")
    print("=" * 60)
    print()
    print("Le DocumentLoader charge des fichiers (PDF, TXT, JSON, CSV)")
    print("et les découpe en chunks pour l'indexation.")
    pause()

    from src.rag.document_loader import DocumentLoader

    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    print(f"✅ DocumentLoader créé")
    print(f"   Taille des chunks: {loader.chunk_size} caractères")
    print(f"   Chevauchement: {loader.chunk_overlap} caractères")
    pause()

    # Charger les catégories de gravité
    print("Chargement des catégories de gravité intégrées:")
    categories = loader.load_gravity_categories()

    for cat in categories:
        level = cat["metadata"]["level"]
        priority = cat["metadata"]["priority"]
        print(f"   {level} (priorité {priority})")
    pause()

    # =========================================================
    # ÉTAPE 6 : Mode interactif
    # =========================================================
    print("=" * 60)
    print("ÉTAPE 6 : MODE INTERACTIF")
    print("=" * 60)
    print()
    print("Maintenant, teste toi-même! Tape une requête médicale")
    print("et le RAG trouvera les documents pertinents.")
    print()
    print("Tape 'quit' pour quitter.")
    print()

    while True:
        requete = input("🔍 Ta requête: ").strip()

        if requete.lower() in ["quit", "exit", "q"]:
            break

        if not requete:
            continue

        resultats = retriever.retrieve(requete, top_k=3)

        print(f"\n📋 {len(resultats)} résultats trouvés:\n")

        for i, r in enumerate(resultats, 1):
            score = r.get("score", 0)
            metadata = r.get("metadata", {})
            source = metadata.get("source", "document")
            urgence = metadata.get("urgence", "")
            texte = r.get("text", "")

            print(f"[{i}] Score: {score:.3f} | Source: {source}")
            if urgence:
                print(f"    Urgence: {urgence}")
            print(f"    {texte[:150]}...")
            print()

        print("-" * 40)

    # Nettoyage
    print("\n🧹 Nettoyage de la base de démo...")
    if demo_db_path.exists():
        shutil.rmtree(demo_db_path)

    print("\n✅ Demo terminée! Tu as compris le fonctionnement du RAG.")
    print()
    print("Résumé du flux:")
    print("   1. DocumentLoader charge et découpe les documents")
    print("   2. EmbeddingProvider transforme le texte en vecteurs")
    print("   3. VectorStore stocke et indexe les vecteurs")
    print("   4. Retriever recherche et formate pour le LLM")
    print()


if __name__ == "__main__":
    main()
