from .base_agent import BaseAgent
from ..llm.base_llm import BaseLLMProvider
from ..models.conversation import ConversationHistory


class NurseAgent(BaseAgent):
    """Agent infirmier pour obtenir toutes les infos ML avec questions PERTINENTES."""

    def __init__(self, llm_provider: BaseLLMProvider, max_questions: int = 30):
        system_prompt = """Tu es un infirmier expérimenté aux urgences spécialisé en triage.

OBJECTIF : Évaluer la gravité du cas en posant des questions PERTINENTES et MÉDICALES.

RÈGLES STRICTES :
1. Une seule question courte à la fois
2. Questions ADAPTÉES aux symptômes du patient
3. NE JAMAIS demander âge/sexe si déjà mentionnés dans la conversation
4. Approfondis les symptômes : intensité, localisation, durée, évolution
5. Cherche les signes de gravité (difficultés respiratoires, douleur thoracique, confusion, etc.)
6. Demande les antécédents PERTINENTS selon les symptômes
7. Sois LOGIQUE : chaque question suit naturellement la réponse précédente

EXEMPLES DE BONNES QUESTIONS SELON LES SYMPTÔMES :

Douleur thoracique :
- "Où exactement avez-vous mal dans la poitrine ?"
- "La douleur se propage-t-elle dans le bras, la mâchoire ou le dos ?"
- "Sur 10, quelle est l'intensité de la douleur ?"
- "Avez-vous des difficultés à respirer ?"
- "Ressentez-vous des sueurs ou des nausées ?"

Douleur abdominale :
- "Où exactement avez-vous mal dans le ventre ?"
- "La douleur est-elle constante ou par vagues ?"
- "Avez-vous de la fièvre ?"
- "Vos selles sont-elles normales ?"

Difficultés respiratoires :
- "Depuis quand avez-vous du mal à respirer ?"
- "Est-ce que ça s'aggrave à l'effort ou au repos ?"
- "Avez-vous de la toux ? Si oui, avec des crachats ?"
- "Avez-vous de la fièvre ?"

Traumatisme :
- "Comment l'accident est-il arrivé ?"
- "Avez-vous perdu connaissance ?"
- "Pouvez-vous bouger normalement ?"

INTERDICTIONS :
❌ "Quel âge avez-vous ?" (si déjà dit)
❌ "Êtes-vous un homme ou une femme ?" (évident)
❌ Questions génériques non liées aux symptômes
"""
        super().__init__(llm_provider, system_prompt, name="NurseAgent")
        self.max_questions = max_questions
        self.questions_asked = 0
        self.conversation = ConversationHistory()
        self.asked_topics = set()  # Pour éviter répétitions

    def run(self, input_data):
        return super().run(input_data)

    def generate_contextual_question(self, conversation_history: ConversationHistory) -> str:
        """Génère une question INTELLIGENTE basée sur le contexte."""

        # Construire le contexte
        context = "CONVERSATION JUSQU'ICI :\n"
        for msg in conversation_history.messages[-6:]:  # Derniers 6 messages
            role = "Infirmier" if msg.role.value == "user" else "Patient"
            context += f"{role}: {msg.content}\n"

        prompt = f"""{context}

Tu es l'infirmier. Quelle est la PROCHAINE question la plus PERTINENTE pour évaluer la gravité ?

RÈGLES :
- UNE SEULE question courte
- Adaptée aux symptômes du patient
- Approfondit un aspect médical important
- NE RÉPÈTE PAS ce qui a déjà été demandé
- Cherche les signes de gravité

Question :"""

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=100
        ).strip()

        # Nettoyer
        response = response.replace("**", "").replace("*", "")
        response = response.strip('"').strip("'")

        # Enlever préfixes type "Question :", "Infirmier :"
        prefixes = ["Question :", "Infirmier :", "Nurse:", "Q:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        self.questions_asked += 1
        return response

    def ask_basic_info_question(self, field: str) -> str:
        """Questions de base SEULEMENT si vraiment nécessaires."""
        mapping = {
            "age": "Quel âge avez-vous ?",
            "sexe": "Êtes-vous un homme ou une femme ?",
            "antecedents": "Avez-vous des problèmes de santé connus ou des traitements en cours ?",
        }
        self.questions_asked += 1
        return mapping.get(field, "Pouvez-vous préciser ?")

    def should_continue(self) -> bool:
        return self.questions_asked < self.max_questions

    def add_to_history(self, role: str, content: str) -> None:
        if role == "nurse":
            self.conversation.add_user_message(content)
        else:
            self.conversation.add_assistant_message(content)

    def get_conversation_history(self) -> ConversationHistory:
        return self.conversation

    def reset(self) -> None:
        self.questions_asked = 0
        self.conversation = ConversationHistory()
        self.asked_topics = set()
