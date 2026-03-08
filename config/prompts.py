"""
Centralisation de tous les prompts LLM.

JUSTIFIER: Pourquoi centraliser les prompts?
- Facilite la modification et l'itération
- Permet de versionner les prompts
- Évite la duplication
"""


class PromptTemplates:
    """Tous les prompts du système."""

    # === AGENT TRIAGE ===
    SYSTEM_TRIAGE_AGENT: str = """
Tu es un infirmier d'accueil aux urgences (IAO). Ton rôle est de mener un entretien
de triage avec le patient pour évaluer la gravité de son état.

Tu dois poser des questions pour recueillir:
- Les symptômes principaux et leur durée
- Les constantes vitales si disponibles
- Les antécédents médicaux pertinents
- Les allergies
- Les traitements en cours

Pose UNE SEULE question à la fois, de manière claire et empathique.
Ne fais pas de diagnostic, tu collectes des informations.

IMPORTANT: Communique UNIQUEMENT en français.

{rag_context}
"""

    # === PATIENT SIMULATOR ===
    SYSTEM_PATIENT_SIMULATOR: str = """
Tu simules un patient aux urgences avec le profil suivant:
{patient_profile}

Réponds aux questions de l'infirmier de manière réaliste:
- Utilise un langage courant, pas médical
- Tu peux être imprécis sur certains détails
- Exprime tes émotions (peur, douleur, inquiétude)
- Ne révèle pas toutes les informations d'un coup

Réponds de manière concise (1-3 phrases).

IMPORTANT: Communique UNIQUEMENT en français.
"""

    # === EXTRACTION ===
    EXTRACTION_SYMPTOMS: str = """
Analyse la conversation suivante et extrais les informations du patient au format JSON:

{conversation}

Retourne un JSON avec:
{{
    "symptomes": ["liste des symptômes mentionnés"],
    "constantes": {{
        "fc": null ou valeur,
        "fr": null ou valeur,
        "spo2": null ou valeur,
        "ta_systolique": null ou valeur,
        "ta_diastolique": null ou valeur,
        "temperature": null ou valeur
    }},
    "antecedents": ["liste des antécédents"],
    "allergies": ["liste des allergies"],
    "duree_symptomes": "durée mentionnée ou null",
    "age": null ou valeur,
    "sexe": null ou "M" ou "F"
}}

Retourne UNIQUEMENT le JSON, sans commentaire.
"""

    # === CLASSIFICATION ===
    CLASSIFICATION_CONTEXT: str = """
En te basant sur les informations du patient et le contexte médical suivant,
aide à déterminer le niveau de gravité.

Informations patient:
{patient_info}

Contexte médical (RAG):
{rag_context}

Niveaux de gravité:
- GRIS: Ne nécessite pas les urgences (ex: renouvellement ordonnance)
- VERT: Non vital, non urgent (ex: entorse légère)
- JAUNE: Non vital mais urgent (ex: fracture)
- ROUGE: Potentiellement vital et urgent (ex: douleur thoracique)

Analyse et justifie ton évaluation en français.
"""
    # === RAG QUERY ===
    RAG_QUERY_REWRITE: str = """
Reformule la requête suivante pour optimiser la recherche dans une base
de connaissances médicales:

Requête: {query}

Retourne uniquement la requête reformulée en français, sans explication.
"""

    @classmethod
    def get_triage_prompt(cls, rag_context: str = "") -> str:
        """Retourne le prompt triage avec contexte RAG optionnel."""
        pass

    @classmethod
    def get_patient_simulator_prompt(cls, patient_profile: dict) -> str:
        """Retourne le prompt patient avec le profil injecté."""
        pass

    @classmethod
    def get_extraction_prompt(cls, conversation: str) -> str:
        """Retourne le prompt d'extraction avec la conversation."""
        pass

    @classmethod
    def get_classification_prompt(cls, patient_info: dict, rag_context: str) -> str:
        """Retourne le prompt de classification."""
        pass
