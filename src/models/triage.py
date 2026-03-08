"""
Modèle pour les résultats de triage.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .patient import GravityLevel


class TriageResult(BaseModel):
    """Résultat complet d'un triage."""

    patient_id: str
    gravity_level: GravityLevel
    confidence: float = Field(ge=0, le=1)

    # Explications
    reasoning: str = Field(description="Explication du choix de gravité")
    key_factors: list[str] = Field(default_factory=list, description="Facteurs clés de décision")
    recommended_actions: list[str] = Field(default_factory=list)

    # Traçabilité
    sources_used: list[str] = Field(default_factory=list, description="Documents RAG utilisés")
    ml_prediction: Optional[GravityLevel] = Field(default=None, description="Prédiction ML seule")
    llm_prediction: Optional[GravityLevel] = Field(default=None, description="Prédiction LLM seule")

    # Métadonnées
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(default="")
    processing_time_ms: Optional[float] = None

    def to_display_dict(self) -> dict:
        """Format pour affichage dans l'interface."""
        pass

    def get_color(self) -> str:
        """Retourne la couleur associée au niveau de gravité."""
        pass

    def get_priority_score(self) -> int:
        """Retourne un score de priorité (1-4)."""
        pass

    def is_high_priority(self) -> bool:
        """Vérifie si le cas est prioritaire (JAUNE ou ROUGE)."""
        pass

    def to_summary_string(self) -> str:
        """Résumé textuel du résultat."""
        pass


class TriageEvaluation(BaseModel):
    """Évaluation d'un triage (pour la simulation)."""

    predicted: GravityLevel
    actual: GravityLevel
    is_correct: bool

    # Métriques d'erreur
    severity_error: int = Field(description="Différence de sévérité (-3 à +3)")
    is_under_triage: bool = Field(description="Sous-estimation de gravité")
    is_over_triage: bool = Field(description="Sur-estimation de gravité")

    def calculate_severity_error(self) -> int:
        """Calcule l'erreur de sévérité."""
        pass

    def to_dict(self) -> dict:
        """Sérialisation."""
        pass
