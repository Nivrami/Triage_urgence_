import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F


class ClinicalTriageClassifier:
    def __init__(self):
        # On utilise une version de ClinicalBERT fine-tunée pour la classification de texte
        # ou un pipeline de Zero-Shot si on n'a pas de dataset d'entraînement spécifique.
        self.model_name = "medicalai/ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Chargement du modèle avec une tête de classification pour 4 classes
        # Note: En production, ce modèle devrait être fine-tuné sur des données de triage
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4,
            problem_type="single_label_classification",
        )

        # Labels demandés
        self.labels = ["GRIS", "VERT", "JAUNE", "ROUGE"]
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels)}
        self.label_desc = {
            "GRIS": "Ne nécessite pas les urgences, ni de voir rapidement un médecin généraliste. Situation stable.",
            "VERT": "Pathologie non vitale et non urgente. Consultation classique.",
            "JAUNE": "Pathologie non vitale mais urgente. Nécessite une prise en charge rapide.",
            "ROUGE": "Pathologie potentiellement vitale et urgente. Détresse vitale suspectée.",
        }
        self.model.eval()  # Mode évaluation

    def _prepare_input_text(self, id_data, const_data, symptoms_json):
        """
        Concatène les données hétérogènes en un paragraphe textuel
        compréhensible par ClinicalBERT.
        """
        text = f"Patient: {id_data.get('genre')}, {id_data.get('age')} ans. "

        # Ajout des constantes (critique pour ClinicalBERT)
        text += f"Constantes: FC {const_data.get('fc')}bpm, "
        text += f"SpO2 {const_data.get('spo2')}%, "
        text += f"Temp {const_data.get('temp')}°C, "
        text += f"TA {const_data.get('tas')}/{const_data.get('tad')}. "

        # Ajout des symptômes issus du JSON
        symptomes = ", ".join(symptoms_json.get("symptomes_principaux", []))
        text += f"Symptômes: {symptomes}. "
        text += f"Localisation: {symptoms_json.get('localisation')}. "
        text += f"Intensité douleur: {symptoms_json.get('intensite_douleur')}/10."

        return text

    def _check_vital_emergency_rules(self, id_data, const_data):
        """
        Vérifie les critères d'alerte rouge basés sur les constantes et l'âge.
        Retourne True si une condition 'ROUGE' est rencontrée.
        Sources : Ameli.fr et Vidal.fr
        """
        age = id_data.get("age", 30)
        temp = const_data.get("temp", 37.0)
        fc = const_data.get("fc", 75)
        tas = const_data.get("tas", 120)  # TA Systolique
        spo2 = const_data.get("spo2", 98)

        # 1. Condition d'origine : Hypoxie ou Tachycardie extrême
        if spo2 < 90 or fc > 130:
            return True

        # 2. Nourrissons et fièvre (Age <= 1 an et Temp > 38)
        if age <= 1 and temp > 38.0:
            return True

        # 3. Jeunes enfants et forte fièvre (Age <= 3 ans et Temp > 38.5)
        if age <= 3 and temp > 38.5:
            return True

        # 4. Hypertension sévère (TA Systolique >= 170 mmHg)
        if tas >= 170:
            return True

        # 5. État de choc / Bradycardie (FC <= 50 et TA Systolique <= 90)
        tas_val = tas if tas > 20 else tas * 10  # Normalisation auto 9 -> 90
        if fc <= 50 and tas_val <= 90:
            return True

        return False

    def classify_emergency(self, id_data, const_data, symptoms_json):
        """
        Réalise la classification via ClinicalBERT.
        """
        input_text = self._prepare_input_text(id_data, const_data, symptoms_json)

        # Tokenisation du texte pour ClinicalBERT
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Inférence avec ClinicalBERT
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)

        # Récupération de la prédiction
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item()
        score_final = self.id_to_label[predicted_class_id]

        # Application du corpus de conditions prioritaires (Hard Rules)
        if self._check_vital_emergency_rules(id_data, const_data):
            score_final = "ROUGE"
            confidence = 1.0  # Force la confiance au maximum car règle médicale absolue

        return {
            "niveau": score_final,
            "confiance": confidence,
            "resume_analyse": input_text,
            "probabilites": {
                label: probabilities[0][idx].item() for label, idx in self.label_to_id.items()
            },
        }
