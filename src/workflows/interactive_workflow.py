"""
Test du workflow - SIMPLE
"""

from src.llm.llm_factory import LLMFactory
from src.workflows.simulation_workflow import SimulationWorkflow
import json
from typing import Optional

# from ..agents.triage_agent import PatientSimulator
from ..metrics.metrics_aggregator import MetricsAggregator
from ..models.patient import Patient
from ..models.conversation import ConversationHistory, MessageRole
from ..models.triage import TriageResult


def test_simulation():
    """Test simple."""

    print("=" * 70)
    print("🏥 TEST SIMULATION")
    print("=" * 70)
    print()

    # Init
    llm = LLMFactory.create("mistral", "mistral-small-latest")
    workflow = SimulationWorkflow(llm, max_turns=8)

    # Run
    result = workflow.run_simulation()

    # Afficher extraction
    print("\n" + workflow.format_for_display())

    # Export ML
    print("\n" + "=" * 60)
    print("[INFO] DONNEES POUR ML")
    print("=" * 60)
    ml_data = workflow.export_for_ml()
    print(json.dumps(ml_data, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("[OK] TEST TERMINE")
    print("=" * 70)

    return result


if __name__ == "__main__":
    test_simulation()
