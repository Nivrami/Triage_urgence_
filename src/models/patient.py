"""
Modèles de données pour les patients.

JUSTIFIER: Pourquoi Pydantic?
- Validation automatique des données
- Sérialisation/désérialisation JSON facile
- Type hints intégrés
- Immutabilité optionnelle
"""

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GravityLevel(Enum):
    """
    Niveaux de gravité du triage.

    """

    GRIS = "GRIS"  # Ne nécessite pas les urgences
    VERT = "VERT"  # Non vital, non urgent
    JAUNE = "JAUNE"  # Non vital mais urgent
    ROUGE = "ROUGE"  # Potentiellement vital et urgent

    @classmethod
    def from_string(cls, value: str) -> "GravityLevel":
        """Convertit une string en GravityLevel."""
        return cls[value.upper()]

    def to_color_code(self) -> str:
        """Retourne le code couleur hexadécimal."""
        color_map = {
            "GRIS": "#808080",  # Gris
            "VERT": "#008000",  # Vert
            "JAUNE": "#FFFF00",  # Jaune
            "ROUGE": "#FF0000",  # Rouge
        }
        return color_map[self.value]

    def get_description(self) -> str:
        """Retourne une description textuelle du niveau de gravité."""
        descriptions = {
            "GRIS": "Ne nécessite pas les urgences",
            "VERT": "Non vital, non urgent",
            "JAUNE": "Non vital mais urgent",
            "ROUGE": "Potentiellement vital et urgent",
        }
        return descriptions[self.value]


class Constantes(BaseModel):
    """
    Constantes vitales du patient.

    Valeurs normales adulte (pour référence):
    - FC: 60-100 bpm
    - FR: 12-20/min
    - SpO2: >95%
    - TA: 120/80 mmHg
    - Température: 36.5-37.5°C
    """

    fc: Optional[int] = Field(default=None, description="Fréquence cardiaque (bpm)")
    fr: Optional[int] = Field(default=None, description="Fréquence respiratoire (/min)")
    spo2: Optional[int] = Field(default=None, description="Saturation O2 (%)")
    ta_systolique: Optional[int] = Field(default=None, description="Tension systolique (mmHg)")
    ta_diastolique: Optional[int] = Field(default=None, description="Tension diastolique (mmHg)")
    temperature: Optional[float] = Field(default=None, description="Température (°C)")

    def is_complete(self) -> bool:
        """Vérifie si toutes les constantes sont renseignées."""
        if (
            self.fc is not None
            and self.fr is not None
            and self.spo2 is not None
            and self.ta_systolique is not None
            and self.ta_diastolique is not None
            and self.temperature is not None
        ):
            return True
        return False

    def get_missing_fields(self) -> list[str]:
        """Retourne la liste des constantes manquantes."""
        missing = []
        if self.fc is None:
            missing.append("fc")
        if self.fr is None:
            missing.append("fr")
        if self.spo2 is None:
            missing.append("spo2")
        if self.ta_systolique is None:
            missing.append("ta_systolique")
        if self.ta_diastolique is None:
            missing.append("ta_diastolique")
        if self.temperature is None:
            missing.append("temperature")
        return missing

    def to_feature_vector(self) -> list[float]:
        """
        Convertit en vecteur numérique pour le ML.
        Remplace les None par -1.
        """
        vector = []
        vector.append(self.fc if self.fc is not None else -1)
        vector.append(self.fr if self.fr is not None else -1)
        vector.append(self.spo2 if self.spo2 is not None else -1)
        vector.append(self.ta_systolique if self.ta_systolique is not None else -1)
        vector.append(self.ta_diastolique if self.ta_diastolique is not None else -1)
        vector.append(self.temperature if self.temperature is not None else -1)
        return vector

    def has_critical_values(self) -> bool:
        """Vérifie si des valeurs sont critiques (hors normes)."""
        #
        pass

    def get_anomalies(self) -> list[str]:
        """Retourne la liste des constantes anormales."""
        pass


class Patient(BaseModel):
    """Modèle complet d'un patient aux urgences."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Identifiant unique du patient
    prenom: Optional[str] = None
    nom: Optional[str] = None
    age: Optional[int] = None
    sexe: Optional[str] = Field(default=None, pattern="^(M|F)$")

    # Informations cliniques
    symptomes_exprimes: list[str] = Field(default_factory=list)
    constantes: Optional[Constantes] = None
    antecedents: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    traitements_en_cours: list[str] = Field(default_factory=list)
    duree_symptomes: Optional[str] = None

    # Résultat du triage
    gravite_predite: Optional[GravityLevel] = None
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)

    def is_ready_for_classification(self) -> bool:
        """
        Vérifie si on a assez d'informations pour classifier.
        Minimum requis: au moins 1 symptôme.
        """
        if len(self.symptomes_exprimes) > 0:
            return True
        return False

    def get_completeness_score(self) -> float:
        """
        Retourne un score de complétude des informations (0-1).
        Utile pour décider si continuer l'entretien.
        """
        total_fields = 6  # Nombre total de champs à vérifier
        filled_fields = 0
        if self.prenom is not None:
            filled_fields += 1
        if self.nom is not None:
            filled_fields += 1
        if self.age is not None:
            filled_fields += 1
        if self.sexe is not None:
            filled_fields += 1
        if len(self.symptomes_exprimes) > 0:
            filled_fields += 1
        if self.constantes is not None and self.constantes.is_complete():
            filled_fields += 1
        return filled_fields / total_fields

    def get_missing_critical_info(self) -> list[str]:
        """Retourne les informations critiques manquantes."""
        missing = []
        if len(self.symptomes_exprimes) == 0:
            missing.append("symptomes_exprimes")
        if self.constantes is None or not self.constantes.is_complete():
            missing.append("constantes")
        return missing

    def to_dict(self) -> dict:
        """Sérialisation en dictionnaire."""
        return self.dict()

    def to_summary_string(self) -> str:
        """Résumé textuel pour injection dans un prompt."""
        summary = f"Patient ID: {self.id}\n"
        if self.prenom:
            summary += f"Prénom: {self.prenom}\n"
        if self.nom:
            summary += f"Nom: {self.nom}\n"
        if self.age:
            summary += f"Âge: {self.age}\n"
        if self.sexe:
            summary += f"Sexe: {self.sexe}\n"
        if self.symptomes_exprimes:
            summary += f"Symptômes: {', '.join(self.symptomes_exprimes)}\n"
        if self.constantes:
            summary += "Constantes:\n"
            if self.constantes.fc is not None:
                summary += f"  - FC: {self.constantes.fc} bpm\n"
            if self.constantes.fr is not None:
                summary += f"  - FR: {self.constantes.fr} /min\n"
            if self.constantes.spo2 is not None:
                summary += f"  - SpO2: {self.constantes.spo2} %\n"
            if (
                self.constantes.ta_systolique is not None
                and self.constantes.ta_diastolique is not None
            ):
                ta_sys = self.constantes.ta_systolique
                ta_dia = self.constantes.ta_diastolique
                summary += f"  - TA: {ta_sys}/{ta_dia} mmHg\n"
            if self.constantes.temperature is not None:
                summary += f"  - Température: {self.constantes.temperature} °C\n"
        return summary

    def merge_with(self, other_data: dict) -> "Patient":
        """Fusionne avec de nouvelles données extraites."""
        pass
