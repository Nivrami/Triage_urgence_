from typing import Any
from .base_agent import BaseAgent
from ..llm.base_llm import BaseLLMProvider
from ..models.patient import Patient
from ..models.conversation import ConversationHistory


class PatientSimulator(BaseAgent):
    """Simulateur de patient."""

    def __init__(self, llm_provider: BaseLLMProvider, patient: Patient):
        self.patient = patient
        self.conversation = ConversationHistory()
        system_prompt = self._build_system_prompt()
        super().__init__(
            llm_provider, system_prompt=system_prompt, name=f"Patient_{patient.prenom}"
        )

    def run(self, input_data: Any) -> dict:
        question = str(input_data)
        response = self.respond(question)
        return {"response": response}

    def respond(self, question: str) -> str:
        """Répond à une question de l'infirmier."""
        self.conversation.add_user_message(question)

        prompt = f"""Tu es {self.patient.prenom} {self.patient.nom}, {self.patient.age} ans.

RÈGLES STRICTES :
- Réponse COURTE (1-3 phrases max)
- Langage SIMPLE (comme un vrai patient)
- PAS de notes entre parenthèses
- PAS de "Réponse de..." ou "Note :"
- Cohérent avec tes symptômes

Question : {question}

Ta réponse (courte et simple) :"""

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=150
        ).strip()

        # Nettoyer les notes
        response = response.split("(Note :")[0]
        response = response.split("*(Note :")[0]
        response = response.replace("**", "")
        response = response.replace("*", "")
        response = response.strip()

        self.conversation.add_assistant_message(response)
        return response

    def get_initial_complaint(self) -> str:
        """Plainte initiale du patient."""
        prompt = f"""Tu es {self.patient.prenom} {self.patient.nom}, {self.patient.age} ans.

Exprime ta plainte en 1-2 phrases SIMPLES :

RÈGLES :
- Langage patient (pas médical)
- Court et direct
- PAS de notes ou parenthèses

Ta plainte :"""

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=100
        ).strip()

        # Nettoyer
        response = response.split("(")[0]
        response = response.replace("**", "").replace("*", "")
        response = response.strip('"')

        self.conversation.add_assistant_message(response)
        return response

    def _build_system_prompt(self) -> str:
        """Construit le prompt système."""
        prompt = f"Tu es {self.patient.prenom} {self.patient.nom}, {self.patient.age} ans.\n\n"

        if self.patient.symptomes_exprimes:
            prompt += "Symptômes :\n"
            for s in self.patient.symptomes_exprimes:
                prompt += f"- {s}\n"

        if self.patient.antecedents:
            prompt += f"Antécédents : {', '.join(self.patient.antecedents)}\n"
        else:
            prompt += "Antécédents : Aucun\n"

        prompt += "\nRÈGLES :\n"
        prompt += "- Parle simplement\n"
        prompt += "- Sois cohérent\n"
        prompt += "- Pas de jargon médical\n"

        return prompt

    def get_conversation_history(self) -> ConversationHistory:
        return self.conversation

    def reset_conversation(self) -> None:
        self.conversation = ConversationHistory()
