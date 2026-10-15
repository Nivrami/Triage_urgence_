"""
Workflow de simulation AMÉLIORÉ avec questions pertinentes
"""

from typing import Optional

from src.agents.patient_generator import PatientGenerator
from src.agents.patient_simulator import PatientSimulator
from src.agents.nurse_agent import NurseAgent
from src.agents.conversation_analyzer import ConversationAnalyzer
from src.llm.base_llm import BaseLLMProvider
from src.models.patient import Patient, GravityLevel
from src.models.conversation import ConversationHistory


class SimulationWorkflow:
    """Workflow : conversation intelligente + extraction ML."""

    def __init__(self, llm_provider: BaseLLMProvider, max_turns: int = 10):
        self.llm = llm_provider
        self.max_turns = max(max_turns, 8)  # Minimum 8 questions
        self.patient_generator = PatientGenerator(llm_provider)
        self.analyzer = ConversationAnalyzer(llm_provider)
        self.conversation = None
        self.original_patient = None
        self.extracted_patient = None
        self.pathology = None

    def run_simulation(self, pathology: Optional[str] = None) -> dict:
        """Simulation complète avec questions PERTINENTES."""

        # 1. Pathologie
        if pathology is None:
            self.pathology = self._generate_random_pathology()
        else:
            self.pathology = pathology

        print(f"🎲 Pathologie : {self.pathology}\n")

        # 2. Patient (AVEC constantes)
        self.original_patient = self.patient_generator.generate_from_description(self.pathology)
        print(
            f"👤 {self.original_patient.prenom} {self.original_patient.nom}, {self.original_patient.age} ans\n"
        )

        # 3. Agents
        patient_sim = PatientSimulator(self.llm, self.original_patient)
        nurse = NurseAgent(self.llm, max_questions=self.max_turns)
        self.conversation = ConversationHistory()

        # 4. Plainte initiale
        print("💬 Conversation :")
        print("-" * 60)
        initial = patient_sim.get_initial_complaint()
        self.conversation.add_assistant_message(initial)
        print(f"🤒 {initial}\n")

        # 5. Questions INTELLIGENTES et PERTINENTES
        turn = 0
        basic_info_collected = {"age": False, "sexe": False}

        while turn < self.max_turns:
            # Analyser ce qu'on a déjà
            self.extracted_patient = self.analyzer.extract_patient_info(self.conversation)

            # ⭐ LOGIQUE DE QUESTIONS AMÉLIORÉE
            question = None

            # Vérifier si on a âge/sexe (demander UNE SEULE FOIS si manquant)
            if turn < 2:  # Seulement dans les 2 premières questions
                if not self.extracted_patient.age and not basic_info_collected["age"]:
                    # Vérifier si l'âge est dans la plainte initiale
                    if not any(str(i) in initial.lower() for i in range(10, 100)):
                        question = nurse.ask_basic_info_question("age")
                        basic_info_collected["age"] = True

                elif not self.extracted_patient.sexe and not basic_info_collected["sexe"]:
                    # Déduire le sexe du prénom si possible
                    prenom = self.original_patient.prenom.lower()
                    male_names = ["jean", "pierre", "paul", "marc", "louis", "michel"]
                    female_names = ["marie", "jeanne", "sophie", "claire", "isabelle"]

                    is_obvious = any(n in prenom for n in male_names + female_names)

                    if not is_obvious:
                        question = nurse.ask_basic_info_question("sexe")
                    basic_info_collected["sexe"] = True

            if not question:
                question = nurse.generate_contextual_question(self.conversation)

            # Poser la question
            self.conversation.add_user_message(question)
            print(f"👨‍⚕️ {question}")

            # Réponse patient
            response = patient_sim.respond(question)
            self.conversation.add_assistant_message(response)
            print(f"🤒 {response}\n")

            turn += 1

            # Arrêt si on a assez d'infos critiques
            if turn >= 6:  # Minimum 6 questions
                completeness = self.analyzer.get_completeness_score(self.extracted_patient)
                if completeness["score"] > 0.7:  # 70% d'infos
                    print("[OK] Informations suffisantes collectees.")
                    break

        # 6. L'infirmier MESURE les constantes
        print("\n[INFO] L'infirmier mesure les constantes vitales...")
        print(f"   FC : {self.original_patient.constantes.fc} bpm")
        print(f"   FR : {self.original_patient.constantes.fr} /min")
        print(f"   SpO2 : {self.original_patient.constantes.spo2}%")
        print(
            f"   TA : {self.original_patient.constantes.ta_systolique}/{self.original_patient.constantes.ta_diastolique} mmHg"
        )
        print(f"   Temp : {self.original_patient.constantes.temperature}°C")

        # 7. Extraction finale
        print("\n" + "=" * 60)
        print("[EXTRACTION]")
        print("=" * 60)
        self.extracted_patient = self.analyzer.extract_patient_info(self.conversation)
        completeness = self.analyzer.get_completeness_score(self.extracted_patient)

        print(f"[OK] Completude : {completeness['score']*100:.0f}%")
        if completeness["missing"]:
            print(f"   Infos manquantes : {', '.join(completeness['missing'])}")
        else:
            print("   Toutes les infos collectées !")

        return {
            "pathology": self.pathology,
            "original_patient": self.original_patient,
            "conversation": self.conversation,
            "extracted_patient": self.extracted_patient,
            "completeness": completeness,
        }

    def _generate_random_pathology(self) -> str:
        """Génère pathologie propre."""
        prompt = """Génère UNE SEULE pathologie aléatoire réaliste pour les urgences.

FORMAT EXACT : "[Sexe] de [âge] ans avec [pathologie]"

EXEMPLES :
- Homme de 62 ans avec infarctus du myocarde
- Femme de 35 ans avec appendicite aiguë
- Homme de 45 ans avec pneumonie
- Femme de 78 ans avec fracture du col du fémur
- Homme de 55 ans avec accident vasculaire cérébral

Une ligne seulement :"""

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=1.0, max_tokens=50
        )

        clean = response.strip().split("\n")[0]
        clean = clean.strip('"').strip("'").strip("-").strip()

        # Enlever préfixes inutiles
        prefixes = ["Voici", "Pathologie", "Une pathologie", "Cas", "Exemple"]
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean.split(":", 1)[-1].strip()

        return clean

    def format_for_display(self) -> str:
        """Affichage AVEC constantes originales."""
        if not self.extracted_patient or not self.original_patient:
            return "Aucune simulation."

        lines = []
        lines.append("=" * 60)
        lines.append("📋 DONNÉES PATIENT COMPLÈTES")
        lines.append("=" * 60)

        extr = self.extracted_patient
        orig = self.original_patient

        # Infos extraites conversation
        lines.append(f"Âge : {extr.age or orig.age} ans")
        lines.append(f"Sexe : {extr.sexe or orig.sexe}")

        if extr.symptomes_exprimes:
            lines.append(f"\nSymptômes (déclarés) :")
            for s in extr.symptomes_exprimes[:8]:
                lines.append(f"  • {s}")

        if extr.duree_symptomes:
            lines.append(f"\nDurée : {extr.duree_symptomes}")

        # Constantes MESURÉES (patient original)
        lines.append(f"\nConstantes (mesurées) :")
        c = orig.constantes
        if c:
            if c.fc:
                lines.append(f"  • FC : {c.fc} bpm")
            if c.fr:
                lines.append(f"  • FR : {c.fr} /min")
            if c.spo2:
                lines.append(f"  • SpO2 : {c.spo2}%")
            if c.ta_systolique and c.ta_diastolique:
                lines.append(f"  • TA : {c.ta_systolique}/{c.ta_diastolique} mmHg")
            if c.temperature:
                lines.append(f"  • Temp : {c.temperature}°C")

        if extr.antecedents:
            lines.append(f"\nAntécédents :")
            for a in extr.antecedents[:5]:
                lines.append(f"  • {a}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def export_for_ml(self) -> dict:
        """Export ML avec constantes ORIGINALES (mesurées)."""
        if not self.extracted_patient or not self.original_patient:
            return {}

        extr = self.extracted_patient
        orig = self.original_patient

        data = {
            "pathology": self.pathology,
            # Infos conversationnelles (extraites)
            "age": extr.age or orig.age,
            "sexe": extr.sexe or orig.sexe,
            "symptomes": extr.symptomes_exprimes,
            "nb_symptomes": len(extr.symptomes_exprimes or []),
            "duree": extr.duree_symptomes,
            "antecedents": extr.antecedents,
            "nb_antecedents": len(extr.antecedents or []),
            # Constantes MESURÉES (patient original)
            "fc": orig.constantes.fc if orig.constantes else None,
            "fr": orig.constantes.fr if orig.constantes else None,
            "spo2": orig.constantes.spo2 if orig.constantes else None,
            "ta_systolique": orig.constantes.ta_systolique if orig.constantes else None,
            "ta_diastolique": orig.constantes.ta_diastolique if orig.constantes else None,
            "temperature": orig.constantes.temperature if orig.constantes else None,
        }

        return data

    def reset(self):
        self.conversation = None
        self.original_patient = None
        self.extracted_patient = None
        self.pathology = None
