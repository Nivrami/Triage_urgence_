"""
Chatbot Final - Mistral API ROBUSTE avec Monitoring
"""

import re
import time
import os
from typing import Dict
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()


class TriageChatbotAPI:
    """Chatbot Mistral API robuste avec tracking complet."""

    def __init__(self, api_key: str = None, retriever=None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.retriever = retriever  # RAG retriever
        if self.api_key:
            self.client = Mistral(api_key=self.api_key)
            self.use_api = True

            print("[OK] Mistral API activee")
        else:
            self.use_api = False
            print("[WARN] Mode regles (sans API)")

        self.reset()

    def start(self) -> str:
        return "Bonjour. Indiquez : prénom, âge, sexe\nExemple : Jean, 25 ans, homme"

    def chat(self, msg: str) -> str:
        """Chat principal avec tracking."""
        start = time.time()
        self.data["messages"].append({"role": "user", "content": msg})

        # Extraire données
        self._extract(msg)

        # Déterminer étape suivante
        next_step = self._get_next_step()

        # Générer réponse
        if self.use_api and self.data.get("age") and next_step != "identity":
            response = self._ask_with_api(next_step)
        else:
            response = self._ask_with_rules(next_step)

        self.data["messages"].append({"role": "assistant", "content": response})

        # Track latence
        self._track_latency(time.time() - start)

        return response

    def _get_next_step(self) -> str:
        """Détermine prochaine étape unique."""
        if not self.data.get("age") or not self.data.get("sex"):
            return "identity"

        if not self.data.get("symptoms"):
            return "symptoms"

        v = self.data["vitals"]

        if "Temperature" not in v:
            return "temperature"
        if "FC" not in v:
            return "fc"
        if "TA_systolique" not in v:
            return "ta"
        if "SpO2" not in v:
            return "spo2"
        if "FR" not in v:
            return "fr"

        return "done"

    def _ask_with_api(self, step: str) -> str:
        """Appel Mistral avec enrichissement RAG."""
        try:
            # Récupérer contexte RAG si symptômes présents
            rag_context = ""
            if self.retriever and self.data.get("symptoms"):
                try:
                    query = " ".join(self.data["symptoms"])
                    # Supporte les deux interfaces (RAGRetriever et Retriever)
                    if hasattr(self.retriever, "retrieve_context"):
                        rag_context = self.retriever.retrieve_context(query=query, top_k=3)
                    elif hasattr(self.retriever, "retrieve_and_format"):
                        rag_context = self.retriever.retrieve_and_format(
                            query=query, top_k=3, max_tokens=500
                        )
                    else:
                        rag_context = ""
                except Exception as e:
                    print(f"RAG Error: {e}")
                    rag_context = ""

            # Prompts système
            prompts = {
                "symptoms": """Tu es un assistant médical empathique. Le patient a déjà donné son identité.
Demande maintenant son symptôme principal de manière naturelle et rassurante.
Réponds en 1-2 phrases maximum.""",
                "temperature": """Le patient a décrit ses symptômes.
Demande maintenant sa température corporelle de manière claire.
IMPORTANT: L'exemple DOIT inclure l'unité °C pour que le système reconnaisse la valeur.
Exemple à donner: "38.5°C" ou "38.5 degrés"
Sois bref et précis.""",
                "fc": """Demande la fréquence cardiaque (pouls) du patient.
IMPORTANT: L'exemple DOIT inclure l'unité bpm pour que le système reconnaisse la valeur.
Exemple à donner: "80 bpm" ou "80 battements par minute"
Reste concis.""",
                "ta": """Demande la tension artérielle.
Format attendu: deux nombres séparés par un slash (systolique/diastolique).
Exemple à donner: "120/80"
Une seule phrase.""",
                "spo2": """Demande la saturation en oxygène (SpO2).
IMPORTANT: L'exemple DOIT inclure le symbole % pour que le système reconnaisse la valeur.
Exemple à donner: "97%" ou "saturation 97"
Sois direct.""",
                "fr": """Demande la fréquence respiratoire.
IMPORTANT: L'exemple DOIT inclure l'unité /min pour que le système reconnaisse la valeur.
Exemple à donner: "16/min" ou "16 respirations par minute"
Concis et clair.""",
                "done": """Toutes les informations sont collectées.
Informe le patient que son dossier est complet et qu'il peut obtenir une prédiction.
Sois rassurant et professionnel.
Une phrase courte.""",
            }

            system_prompt = prompts.get(step, "Guide le patient avec empathie.")

            # Ajouter le contexte RAG au prompt si disponible
            if rag_context:
                system_prompt += f"""

Contexte médical de référence (utilise ces informations pour guider tes questions):
{rag_context}"""

            # Contexte des données déjà collectées
            context = self._build_context()

            # Appel API Mistral
            start = time.time()
            resp = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Contexte patient: {context}\n\nQuelle est ta question ?",
                    },
                ],
                temperature=0.4,
                max_tokens=100,
            )

            # Track API call
            self._track_api(
                resp.usage.prompt_tokens, resp.usage.completion_tokens, time.time() - start
            )

            response = resp.choices[0].message.content.strip()

            # Nettoyer la réponse si trop longue
            if len(response) > 200:
                response = response[:200] + "..."

            return response

        except Exception as e:
            print(f"API Error: {e}")
            return self._ask_with_rules(step)

    def _build_context(self) -> str:
        """Construit contexte pour Mistral."""
        parts = []

        if self.data.get("name"):
            parts.append(f"Prénom: {self.data['name']}")
        if self.data.get("age"):
            parts.append(f"Âge: {self.data['age']} ans")
        if self.data.get("sex"):
            sex = "Homme" if self.data["sex"] == "H" else "Femme"
            parts.append(f"Sexe: {sex}")
        if self.data.get("symptoms"):
            parts.append(f"Symptômes: {', '.join(self.data['symptoms'])}")

        v = self.data["vitals"]
        vitals_collected = []
        if "Temperature" in v:
            vitals_collected.append(f"Temp: {v['Temperature']}°C")
        if "FC" in v:
            vitals_collected.append(f"FC: {v['FC']} bpm")
        if "TA_systolique" in v:
            vitals_collected.append(f"TA: {v['TA_systolique']}/{v.get('TA_diastolique', '?')}")
        if "SpO2" in v:
            vitals_collected.append(f"SpO2: {v['SpO2']}%")
        if "FR" in v:
            vitals_collected.append(f"FR: {v['FR']}/min")

        if vitals_collected:
            parts.append(f"Constantes: {', '.join(vitals_collected)}")

        return " | ".join(parts) if parts else "Nouveau patient"

    def _ask_with_rules(self, step: str) -> str:
        """Fallback règles."""
        responses = {
            "identity": "Précisez votre âge et sexe.\nExemple : 25 ans homme",
            "symptoms": "Quel est votre symptôme ?\nExemple : mal de tête / fièvre",
            "temperature": "Quelle est votre température ?\nExemple : 38.5°C",
            "fc": "Quel est votre pouls ?\nExemple : 80 bpm",
            "ta": "Quelle est votre tension ?\nExemple : 120/80",
            "spo2": "Quelle est votre saturation ?\nExemple : 97%",
            "fr": "Quelle est votre fréquence respiratoire ?\nExemple : 16/min",
            "done": "Dossier complet. Cliquez 'Obtenir prediction ML'.",
        }
        return responses.get(step, "Erreur")

    def _extract(self, msg: str):
        """Extraction robuste."""
        ml = msg.lower()

        # Âge
        if not self.data.get("age"):
            m = re.search(r"(\d{1,3})\s*ans?", ml)
            if m:
                age = int(m.group(1))
                if 0 < age < 120:
                    self.data["age"] = age

        # Sexe
        if not self.data.get("sex"):
            if "homme" in ml:
                self.data["sex"] = "H"
            elif "femme" in ml:
                self.data["sex"] = "F"

        # Prénom
        if not self.data.get("name"):
            words = msg.split(",")
            if words:
                first = words[0].strip()
                if len(first) >= 2 and first[0].isupper():
                    self.data["name"] = first

        # Symptômes - dictionnaire étendu par catégorie médicale
        symp = {
            # ══════════════════════════════════════════════════════════
            # NEUROLOGIQUE
            # ══════════════════════════════════════════════════════════
            r"t[eéèê]te|c[ée]phal|migraine": "Céphalée",
            r"vertige|tournis|[ée]tourdissement": "Vertiges",
            r"syncope|[ée]vanoui|perte\s*de\s*connaissance": "Syncope",
            r"convulsion|crise|[ée]pilep": "Convulsions",
            r"confusion|d[ée]sorient": "Confusion",
            r"paralys|faiblesse.*(bras|jambe|visage)": "Déficit neurologique",
            r"trouble.*parole|difficult.*parler": "Trouble de la parole",
            r"trouble.*vision|voi[rt]\s*flou|double": "Trouble visuel",
            # ══════════════════════════════════════════════════════════
            # CARDIOVASCULAIRE
            # ══════════════════════════════════════════════════════════
            r"poitrine|thorax|oppression": "Douleur thoracique",
            r"palpitation|cœur\s*bat|tachycardie": "Palpitations",
            r"jambe.*gonfl|œd[èe]me|enfl[ée]": "Œdème",
            # ══════════════════════════════════════════════════════════
            # RESPIRATOIRE
            # ══════════════════════════════════════════════════════════
            r"essouffl[éeè]|dyspn[ée]e|respir.*difficile": "Dyspnée",
            r"toux": "Toux",
            r"crachats?|expectoration": "Expectorations",
            r"[ée]touff|suffoqu": "Détresse respiratoire",
            # ══════════════════════════════════════════════════════════
            # DIGESTIF
            # ══════════════════════════════════════════════════════════
            r"ventre|abdomen|estomac": "Douleur abdominale",
            r"naus[ée]e|envie\s*de\s*vomir|mal\s*au\s*cœur": "Nausées",
            r"vomi|r[ée]gurgit": "Vomissements",
            r"diarrh[ée]e|selles?\s*liquides?": "Diarrhée",
            r"constip|bloqu[ée]|transit": "Constipation",
            r"sang.*selles|rectorragie": "Rectorragie",
            r"br[uû]lure.*estomac|reflux|acidit": "Reflux gastrique",
            r"difficult[ée].*avaler|dysphagie": "Dysphagie",
            # ══════════════════════════════════════════════════════════
            # URINAIRE
            # ══════════════════════════════════════════════════════════
            r"br[uû]l.*urin|cystite": "Brûlures mictionnelles",
            r"sang.*urine|h[ée]maturie": "Hématurie",
            r"envie.*fr[ée]quente|pollakiurie": "Pollakiurie",
            r"difficult[ée].*uriner|r[ée]tention": "Dysurie",
            # ══════════════════════════════════════════════════════════
            # MUSCULO-SQUELETTIQUE
            # ══════════════════════════════════════════════════════════
            r"dos|lombaire|lumbago|sciatique": "Lombalgie",
            r"genou": "Gonalgie",
            r"hanche": "Coxalgie",
            r"cheville|entorse": "Douleur cheville",
            r"[ée]paule": "Omalgie",
            r"nuque|cervical|torticolis": "Cervicalgie",
            r"articulation|arthr": "Arthralgie",
            r"fracture|cass[ée]": "Traumatisme osseux",
            r"bras": "Douleur membre supérieur",
            r"jambe|mollet": "Douleur membre inférieur",
            # ══════════════════════════════════════════════════════════
            # ORL / OPHTALMOLOGIE
            # ══════════════════════════════════════════════════════════
            r"gorge|angine|pharyn": "Odynophagie",
            r"oreille|otite|acouph[èe]ne": "Otalgie",
            r"nez.*bouch|rhume|sinusite": "Rhinite/Sinusite",
            r"saign.*nez|[ée]pistaxis": "Épistaxis",
            r"œil.*rouge|conjonctiv": "Conjonctivite",
            r"œil.*douleur": "Douleur oculaire",
            # ══════════════════════════════════════════════════════════
            # DERMATOLOGIE
            # ══════════════════════════════════════════════════════════
            r"[ée]ruption|bouton|rash|plaques?": "Éruption cutanée",
            r"d[ée]mangeaison|prurit|gratt": "Prurit",
            r"br[uû]lure(?!.*estomac)": "Brûlure",
            r"plaie|coupure|blessure": "Plaie",
            r"abc[èe]s|furoncle": "Abcès",
            # ══════════════════════════════════════════════════════════
            # GÉNÉRAL / PSYCHIATRIQUE
            # ══════════════════════════════════════════════════════════
            r"fi[eéèê]vre|temp[éeè]rature|frisson": "Fièvre",
            r"fatigue|[ée]puis[ée]|asth[ée]nie": "Asthénie",
            r"perte.*poids|amaigri": "Amaigrissement",
            r"sueur|transpir": "Sueurs",
            r"insomnie|dort.*mal|sommeil": "Trouble du sommeil",
            r"anxi[ée]t[ée]|stress|angoiss|panique": "Anxiété",
            r"allergi|r[ée]action|urticaire": "Réaction allergique",
            r"d[ée]prim|triste|moral": "Syndrome dépressif",
        }

        for p, s in symp.items():
            if s and re.search(p, ml):  # Ignorer les patterns avec valeur vide
                if s not in self.data["symptoms"]:
                    self.data["symptoms"].append(s)

        # Température - exiger contexte (°, degré, température, temp)
        if "Temperature" not in self.data["vitals"]:
            m = re.search(r"(\d{2}[.,]?\d?)\s*(?:°|degr|temp)", ml)
            if m:
                t = float(m.group(1).replace(",", "."))
                if 35 <= t <= 43:
                    self.data["vitals"]["Temperature"] = t

        # FC - exiger contexte (bpm, battement, pouls, cardiaque, fc)
        if "FC" not in self.data["vitals"]:
            m = re.search(r"(\d{2,3})\s*(?:bpm|battement|pouls|cardiaque|fc)", ml)
            if m:
                fc = int(m.group(1))
                if 30 <= fc <= 220:
                    self.data["vitals"]["FC"] = fc

        # TA - format explicite avec slash
        if "TA_systolique" not in self.data["vitals"]:
            m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", ml)
            if m:
                self.data["vitals"]["TA_systolique"] = int(m.group(1))
                self.data["vitals"]["TA_diastolique"] = int(m.group(2))

        # SpO2 - exiger contexte (%, sat, spo, oxygène)
        if "SpO2" not in self.data["vitals"]:
            m = re.search(r"(\d{2,3})\s*(?:%|sat|spo|oxyg)", ml)
            if m:
                spo2 = int(m.group(1))
                if 50 <= spo2 <= 100:
                    self.data["vitals"]["SpO2"] = spo2

        # FR - exiger contexte (respir, /min, fr)
        if "FR" not in self.data["vitals"]:
            m = re.search(r"(\d{1,2})\s*(?:respir|/min|fr\b)", ml)
            if m:
                fr = int(m.group(1))
                if 5 <= fr <= 60:
                    self.data["vitals"]["FR"] = fr

    def _track_latency(self, duration: float):
        """Track latence chatbot."""
        try:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.monitoring.metrics_tracker import get_tracker

            get_tracker().track_latency("Chatbot", "chat", duration)
        except:
            pass

    def _track_api(self, tokens_in: int, tokens_out: int, latency: float):
        """Track appel API Mistral."""
        try:
            import sys
            from pathlib import Path

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.monitoring.metrics_tracker import get_tracker

            get_tracker().track_api_call(
                service="mistral",
                model="mistral-small-latest",
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                latency=latency,
                success=True,
            )
        except:
            pass

    def is_ready_for_prediction(self) -> bool:
        """Vérifie si 5/5 constantes."""
        v = self.data["vitals"]
        return (
            self.data.get("age")
            and self.data.get("sex")
            and self.data.get("symptoms")
            and all(k in v for k in ["Temperature", "FC", "TA_systolique", "SpO2", "FR"])
        )

    def get_summary(self) -> Dict:
        """Résumé pour prédiction ML."""
        return {
            "patient_info": {
                "name": self.data.get("name"),
                "age": self.data.get("age"),
                "sex": "Femme" if self.data.get("sex") == "F" else "Homme",
            },
            "symptoms": self.data.get("symptoms", []),
            "vitals": self.data["vitals"],
            "messages": [
                m.get("content", m) if isinstance(m, dict) else m for m in self.data["messages"]
            ],
        }

    def reset(self):
        """Reset complet."""
        self.data = {
            "name": None,
            "age": None,
            "sex": None,
            "symptoms": [],
            "vitals": {},
            "messages": [],
        }
