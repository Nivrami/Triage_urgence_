"""
Analyseur de conversation - Extrait les informations patient.

Analyse une conversation et extrait :
- Symptômes
- Constantes vitales
- Antécédents
- Durée des symptômes
"""

import json

from ..llm.base_llm import BaseLLMProvider
from ..models.conversation import ConversationHistory
from ..models.patient import Patient, Constantes


class ConversationAnalyzer:
    """
    Analyse une conversation et extrait les données patient.

    Usage:
        analyzer = ConversationAnalyzer(llm)
        patient = analyzer.extract_patient_info(conversation)
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """
        Initialise l'analyseur.

        Args:
            llm_provider: Provider LLM
        """
        self.llm = llm_provider

    def get_missing_fields(self, patient: Patient) -> list[str]:
        """Retourne la liste des champs manquants pour le triage."""
        missing = []

        # Identité
        if patient.age is None:
            missing.append("age")
        if patient.sexe is None:
            missing.append("sexe")

        # Symptômes
        if not patient.symptomes_exprimes:
            missing.append("symptomes_exprimes")
        if not patient.duree_symptomes:
            missing.append("duree_symptomes")

        # Antécédents
        if not patient.antecedents:
            missing.append("antecedents")

        # Constantes vitales
        c = patient.constantes
        if not c or c.fc is None:
            missing.append("fc")
        if not c or c.fr is None:
            missing.append("fr")
        if not c or c.spo2 is None:
            missing.append("spo2")
        if not c or c.ta_systolique is None or c.ta_diastolique is None:
            missing.append("ta")
        if not c or c.temperature is None:
            missing.append("temperature")

        return missing

    def extract_patient_info(self, conversation: ConversationHistory) -> Patient:
        """
        Extrait les informations patient de la conversation.

        Args:
            conversation: Historique de conversation

        Returns:
            Patient: Objet Patient rempli avec les infos extraites
        """
        # Construire le prompt d'extraction
        conversation_text = conversation.get_full_text()

        prompt = f"""Analyse cette conversation médicale entre un infirmier et un patient.
Contexte: urgences hospitalières.

CONVERSATION :
{conversation_text}

EXTRAIT LES INFORMATIONS SUIVANTES (mets null si non mentionné) :

**INFORMATIONS EXTRAITES :**
- **Symptômes** : Liste des symptômes exprimés par le patient (en langage simple)
- **Âge** : Âge du patient (nombre)
- **Sexe** : M ou F (si mentionné)
- **Durée des symptômes** : Depuis combien de temps ("depuis ce matin", "depuis 2 jours"...)
- **Antécédents médicaux** : Maladies, opérations, traitements (liste)
- **Allergies** : Si mentionnées
- **Traitements en cours** : Médicaments pris actuellement

**CONSTANTES VITALES** (si mentionnées ou déduisibles) :
- **Fréquence cardiaque (FC)** : battements/min
- **Fréquence respiratoire (FR)** : respirations/min
- **SpO2** : saturation oxygène en % (0-100)
- **Tension artérielle** : systolique/diastolique (ex: 120/80)
- **Température** : en °C

**INDICES POUR LES CONSTANTES :**
Si le patient dit :
- "mon cœur bat vite" → FC probablement >100
- "j'ai du mal à respirer" → SpO2 probablement <95%, FR >20
- "je me sens chaud" → Température probablement >38°C
- "je me sens étourdi" → TA probablement basse <90/60

RÉPONDS UNIQUEMENT AU FORMAT JSON (pas de texte avant/après) :

{{
  "age": nombre_ou_null,
  "sexe": "M_ou_F_ou_null",
  "symptomes_exprimes": ["liste", "des", "symptomes"],
  "duree_symptomes": "depuis_quand_ou_null",
  "antecedents": ["liste_ou_vide"],
  "allergies": ["liste_ou_vide"],
  "traitements_en_cours": ["liste_ou_vide"],
  "constantes": {{
    "fc": nombre_ou_null,
    "fr": nombre_ou_null,
    "spo2": nombre_ou_null,
    "ta_systolique": nombre_ou_null,
    "ta_diastolique": nombre_ou_null,
    "temperature": nombre_ou_null
  }}
}}

IMPORTANT :
- Extrais UNIQUEMENT ce qui est mentionné ou clairement déductible
- Pour les symptômes, utilise le langage du patient
- Mets null pour ce qui n'est PAS dans la conversation
"""

        # Appeler le LLM
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=800
        )

        # Parser le JSON
        try:
            data = self._extract_json_from_response(response)

            # Créer les constantes
            const_data = data.get("constantes", {})
            constantes = None
            if any(const_data.values()):  # Si au moins une constante
                constantes = Constantes(
                    fc=const_data.get("fc"),
                    fr=const_data.get("fr"),
                    spo2=const_data.get("spo2"),
                    ta_systolique=const_data.get("ta_systolique"),
                    ta_diastolique=const_data.get("ta_diastolique"),
                    temperature=const_data.get("temperature"),
                )

            # Créer le patient
            patient = Patient(
                age=data.get("age"),
                sexe=data.get("sexe"),
                symptomes_exprimes=data.get("symptomes_exprimes", []),
                duree_symptomes=data.get("duree_symptomes"),
                antecedents=data.get("antecedents", []),
                allergies=data.get("allergies", []),
                traitements_en_cours=data.get("traitements_en_cours", []),
                constantes=constantes,
            )

            return patient

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERREUR] Extraction : {e}")
            print(f"Réponse LLM : {response[:200]}...")
            # Retourner patient vide plutôt que crash
            return Patient()

    def _extract_json_from_response(self, response: str) -> dict:
        """Extrait JSON de la réponse."""
        # Nettoyer markdown
        response_clean = response.strip()

        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        elif response_clean.startswith("```"):
            response_clean = response_clean[3:]

        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]

        # Parser
        return json.loads(response_clean.strip())

    def get_completeness_score(self, patient: Patient) -> dict:
        """
        Calcule le score de complétude du patient.

        Args:
            patient: Objet Patient

        Returns:
            {
                "score": float (0-1),
                "missing": list[str],
                "has_critical_info": bool
            }
        """
        total_fields = 0
        filled_fields = 0
        missing = []

        # Champs critiques
        critical_fields = {
            "symptomes_exprimes": "Symptômes",
            "age": "Âge",
            "duree_symptomes": "Durée des symptômes",
        }

        # Champs importants
        important_fields = {
            "sexe": "Sexe",
            "antecedents": "Antécédents",
            "constantes": "Constantes vitales",
        }

        # Vérifier champs critiques
        for field, label in critical_fields.items():
            total_fields += 1
            value = getattr(patient, field)
            if value and (not isinstance(value, list) or len(value) > 0):
                filled_fields += 1
            else:
                missing.append(label)

        # Vérifier champs importants
        for field, label in important_fields.items():
            total_fields += 1
            value = getattr(patient, field)
            if value and (not isinstance(value, list) or len(value) > 0):
                filled_fields += 1

        # Score
        score = filled_fields / total_fields if total_fields > 0 else 0

        # Au moins les infos critiques ?
        has_critical = len(missing) == 0

        return {"score": round(score, 2), "missing": missing, "has_critical_info": has_critical}
