"""
Logique de retrieval pour le RAG.
"""

from typing import Optional
from .vector_store import VectorStore


class Retriever:
    """
    Retriever pour le RAG.

    Responsabilités:
    - Rechercher les documents pertinents
    - Filtrer par score
    - Formater le contexte pour le prompt
    """

    def __init__(
        self, vector_store: VectorStore, default_top_k: int = 5, score_threshold: float = 0.7
    ) -> None:
        """
        Initialise le retriever.

        Args:
            vector_store: Base vectorielle
            default_top_k: Nombre de résultats par défaut
            score_threshold: Score minimum de similarité (distance cosinus)
                            Plus le score est bas, plus les documents sont similaires
        """
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Récupère les documents pertinents.

        Args:
            query: Requête de recherche
            top_k: Nombre de résultats (défaut si None)

        Returns:
            Liste de documents pertinents
        """
        if top_k is None:
            top_k = self.default_top_k

        return self.vector_store.search(query, top_k)

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> list[tuple[dict, float]]:
        """Récupère avec les scores de similarité."""
        documents = self.retrieve(query, top_k)
        return [(doc, doc.get("score", 0)) for doc in documents]

    def format_context(self, documents: list[dict], max_tokens: int = 1000) -> str:
        """
        Formate les documents pour injection dans le prompt.

        Args:
            documents: Documents à formater
            max_tokens: Limite de tokens (approximative: 1 token ≈ 4 chars)

        Returns:
            Contexte formaté en string
        """
        if not documents:
            return "Aucun document pertinent trouvé dans la base de connaissances."

        context_parts = []
        current_length = 0
        max_chars = max_tokens * 4  # Approximation

        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)

            # Construire l'entrée formatée
            source = metadata.get("source", "Document")
            if isinstance(source, str) and "/" in source:
                source = source.split("/")[-1]  # Garder juste le nom du fichier

            page_info = ""
            if "page" in metadata:
                page_info = f" (page {metadata['page']})"

            entry = f"[{i}] {source}{page_info}:\n{text}\n"

            # Vérifier si on dépasse la limite
            if current_length + len(entry) > max_chars:
                # Tronquer si nécessaire
                remaining = max_chars - current_length - 50  # Marge de sécurité
                if remaining > 100:
                    entry = entry[:remaining] + "...\n"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            current_length += len(entry)

        return "\n".join(context_parts)

    def filter_by_threshold(
        self, results: list[tuple[dict, float]], threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Filtre les résultats par score minimum.

        Note: En distance cosinus, un score plus BAS = plus similaire
              Donc on garde les scores EN-DESSOUS du seuil

        Args:
            results: Résultats avec scores
            threshold: Seuil (utilise le défaut si None)

        Returns:
            Documents au-dessous du seuil (plus similaires)
        """
        if threshold is None:
            threshold = self.score_threshold

        # En distance cosinus: score bas = plus similaire
        return [doc for doc, score in results if score <= threshold]

    def retrieve_and_format(
        self, query: str, top_k: Optional[int] = None, max_tokens: int = 1000
    ) -> str:
        """
        Raccourci: retrieve + format en une méthode.

        Returns:
            Contexte formaté prêt pour le prompt
        """
        documents = self.retrieve(query, top_k)
        return self.format_context(documents, max_tokens)

    def multi_query_retrieve(self, queries: list[str], top_k_per_query: int = 3) -> list[dict]:
        """
        Recherche avec plusieurs requêtes et fusionne les résultats.

        Utile quand le patient a plusieurs symptômes.

        Args:
            queries: Liste de requêtes (ex: ["douleur thoracique", "essoufflement"])
            top_k_per_query: Nombre de résultats par requête

        Returns:
            Liste de documents uniques, triés par score
        """
        all_results = {}

        for query in queries:
            documents = self.retrieve(query, top_k_per_query)

            for doc in documents:
                doc_id = doc.get("id", doc.get("text", "")[:50])

                # Garder le meilleur score pour chaque document
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                else:
                    # Garder le score le plus bas (plus similaire)
                    existing_score = all_results[doc_id].get("score", float("inf"))
                    new_score = doc.get("score", float("inf"))
                    if new_score < existing_score:
                        all_results[doc_id] = doc

        # Trier par score (croissant = plus similaire d'abord)
        sorted_results = sorted(all_results.values(), key=lambda x: x.get("score", float("inf")))

        return sorted_results

    def retrieve_for_triage(
        self, symptoms: list[str], vital_signs: dict = None, top_k: int = 5
    ) -> str:
        """
        Méthode spécialisée pour le triage médical.

        Args:
            symptoms: Liste des symptômes du patient
            vital_signs: Constantes vitales (optionnel)
            top_k: Nombre de résultats

        Returns:
            Contexte formaté pour le triage
        """
        # Construire les requêtes basées sur les symptômes
        queries = symptoms.copy()

        # Ajouter des requêtes basées sur les constantes si anormales
        if vital_signs:
            if vital_signs.get("fc", 0) > 100:
                queries.append("tachycardie fréquence cardiaque élevée")
            elif vital_signs.get("fc", 100) < 60:
                queries.append("bradycardie fréquence cardiaque basse")

            if vital_signs.get("spo2", 100) < 95:
                queries.append("désaturation hypoxie SpO2 basse")

            if vital_signs.get("temperature", 37) > 38.5:
                queries.append("fièvre hyperthermie température élevée")

            if vital_signs.get("ta_systolique", 120) > 140:
                queries.append("hypertension tension artérielle élevée")
            elif vital_signs.get("ta_systolique", 120) < 90:
                queries.append("hypotension choc tension basse")

        # Récupérer les documents
        documents = self.multi_query_retrieve(queries, top_k_per_query=3)

        # Limiter au top_k
        documents = documents[:top_k]

        # Formater le contexte
        context = self.format_context(documents)

        return context
