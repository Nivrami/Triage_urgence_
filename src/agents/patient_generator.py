from .base_agent import BaseAgent
from ..llm.base_llm import BaseLLMProvider
from ..models.patient import Patient, Constantes


class PatientGenerator(BaseAgent):
    """Générateur de patient réaliste."""

    def __init__(self, llm_provider: BaseLLMProvider):
        system_prompt = "Expert médical. Génère des profils patients réalistes. JSON uniquement."
        super().__init__(
            llm_provider=llm_provider, system_prompt=system_prompt, name="PatientGenerator"
        )

    def run(self, input_data: str) -> dict:
        patient = self.generate_from_description(input_data)
        return {"patient": patient}

    def generate_from_description(self, description: str) -> Patient:
        prompt = f"""Génère un patient COMPLET et RÉALISTE pour cette pathologie.

PATHOLOGIE : {description}

RÉPONDS UNIQUEMENT EN JSON (pas de texte avant/après) :

{{
  "prenom": "prénom français",
  "nom": "nom français",
  "age": nombre,
  "sexe": "M" ou "F",
  "symptomes_exprimes": ["symptôme en langage patient", ...],
  "duree_symptomes": "depuis quand",
  "antecedents": ["antécédent1", ...],
  "constantes": {{
    "fc": nombre (60-180),
    "fr": nombre (12-35),
    "spo2": nombre (85-100),
    "ta_systolique": nombre (80-200),
    "ta_diastolique": nombre (50-120),
    "temperature": nombre (36.0-40.5)
  }}
}}

RÈGLES CONSTANTES (IMPORTANT) :
- Infarctus → FC 100-130, SpO2 88-93%, TA basse
- Pneumonie → FC 100-120, FR 25-32, SpO2 85-92%, Temp 38.5-40°C
- Fracture → FC 85-95, tout normal
- Gastro → FC normal, Temp 37.5-38.5°C

JSON uniquement :"""

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=600
        )

        data = self._extract_json_from_response(response)

        # Sexe
        sexe_raw = str(data.get("sexe", "M")).upper()[0]
        sexe = "M" if sexe_raw == "M" else "F"

        # Constantes COMPLÈTES
        c = data.get("constantes", {})
        constantes = Constantes(
            fc=c.get("fc", 90),
            fr=c.get("fr", 18),
            spo2=c.get("spo2", 97),
            ta_systolique=c.get("ta_systolique", 120),
            ta_diastolique=c.get("ta_diastolique", 80),
            temperature=c.get("temperature", 37.0),
        )

        patient = Patient(
            prenom=data.get("prenom", "Jean"),
            nom=data.get("nom", "Dupont"),
            age=data.get("age", 50),
            sexe=sexe,
            symptomes_exprimes=data.get("symptomes_exprimes", []),
            constantes=constantes,
            antecedents=data.get("antecedents", []),
            duree_symptomes=data.get("duree_symptomes", "depuis quelques heures"),
        )
        return patient
