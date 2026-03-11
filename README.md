# Triage Urgences — Système d'aide au triage hospitalier par LLM

Système conversationnel d'aide au triage aux urgences combinant un agent LLM (Mistral), un module RAG sur protocoles médicaux, et un classifieur Random Forest pour produire une évaluation de gravité structurée.

---

## Aperçu

![Accueil](screenshots/accueil.png)
![Simulation](screenshots/simulation.png)
![Chatbot](screenshots/chatbot.png)
![Prediction](screenshots/chatbot_ML.png)
---

## Fonctionnalités

- **Conversation naturelle** : un agent infirmier (LLM Mistral) interroge le patient en langage naturel, une question à la fois, de façon adaptée aux symptômes déclarés
- **Simulation de patients** : génération automatique de cas cliniques variés pour tester et évaluer le système
- **Prédiction ML** : classifieur Random Forest entraîné sur 18 features (constantes vitales + symptômes binaires) pour une évaluation objective du niveau de gravité
- **RAG médical** : enrichissement des réponses par retrieval sur une base documentaire de protocoles (ChromaDB + sentence-transformers)
- **4 niveaux de triage** :
  | Niveau | Signification | Action |
  |--------|--------------|--------|
  | 🔴 ROUGE | Urgence vitale | Appeler le 15 immédiatement |
  | 🟡 JAUNE | Urgence | Prise en charge dans l'heure |
  | 🟢 VERT | Non urgent | Consultation sous 24-48h |
  | ⚪ GRIS | Pas d'urgence | Rendez-vous médecin traitant |
- **Monitoring** : suivi des coûts API, latences et métriques de prédiction en temps réel
- **Interface Streamlit** : mode conversation interactive et mode simulation automatique

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE STREAMLIT                        │
│              app/app.py  — Mode interactif & simulation         │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│       NurseAgent        │   │       PatientSimulator          │
│  (src/agents/)          │◄──│  (src/agents/)                  │
│  - Pose les questions   │   │  - Génère des cas cliniques     │
│  - Gère la conversation │   │  - Simule les réponses patient  │
└────────────┬────────────┘   └─────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         COUCHE LLM                              │
│          MistralProvider  (src/llm/mistral_provider.py)         │
│   Prompt = System + Contexte RAG + Historique + Message patient │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│        MODULE RAG       │   │          MODULE ML              │
│  (src/rag/)             │   │  (src/rag/predictor.py)         │
│                         │   │                                 │
│  DocumentLoader         │   │  Random Forest (sklearn)        │
│    → chunking docs      │   │  18 features :                  │
│  EmbeddingProvider      │   │   - 8 constantes vitales        │
│    → sentence-transform │   │     FC, FR, SpO2, TA, Temp, Age │
│  VectorStore (ChromaDB) │   │   - 10 symptômes binaires       │
│    → stockage vecteurs  │   │  + enrichissement RAG           │
│  Retriever              │   │  → niveau ROUGE/JAUNE/VERT/GRIS │
│    → top-k pertinents   │   │                                 │
└─────────────────────────┘   └─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MONITORING                               │
│              (src/monitoring/)                                  │
│   MetricsTracker · CostCalculator · Latences · Prédictions     │
└─────────────────────────────────────────────────────────────────┘
```

### Flux d'un triage

```
Patient : "J'ai une douleur dans la poitrine depuis ce matin"
    │
    ▼
NurseAgent (LLM Mistral)
  → Recherche RAG : protocoles "douleur thoracique"
  → Pose des questions ciblées (intensité, irradiation, dyspnée…)
    │
    ▼  [après collecte des infos]
MLTriagePredictor
  → Constantes vitales + symptômes encodés → 18 features
  → Random Forest → probabilités par classe
  → Enrichissement RAG : recommandations protocoles
    │
    ▼
Résultat structuré
  🔴 URGENCE VITALE — Appeler le 15
  Confiance : 92% | Red flags : Tachycardie (125 bpm), Douleur thoracique
  Recommandations RAG : protocole ROUGE…
```

---

## Structure du projet

```
Triage_urgence_bis/
│
├── app/
│   └── app.py                        # Application Streamlit (interactif + simulation)
│
├── src/
│   ├── agents/
│   │   ├── nurse_agent.py            # Agent infirmier LLM
│   │   ├── patient_simulator.py      # Simulateur de patient
│   │   ├── patient_generator.py      # Générateur de cas cliniques
│   │   └── conversation_analyzer.py  # Analyse de conversation
│   │
│   ├── llm/
│   │   ├── mistral_provider.py       # Provider Mistral API
│   │   ├── llm_factory.py            # Factory LLM
│   │   └── base_llm.py               # Interface abstraite
│   │
│   ├── rag/
│   │   ├── chatbot.py                # Chatbot Mistral + RAG
│   │   ├── predictor.py              # Prédicteur ML + RAG (Random Forest)
│   │   ├── embeddings.py             # Texte → vecteurs (sentence-transformers)
│   │   ├── vector_store.py           # VectorStore + Retriever (ChromaDB)
│   │   └── document_loader.py        # Chargement et chunking des documents
│   │
│   ├── models/
│   │   ├── patient.py                # Modèle de données patient (Pydantic)
│   │   └── conversation.py           # Historique de conversation
│   │
│   ├── monitoring/
│   │   ├── metrics_tracker.py        # Tracking prédictions et latences
│   │   └── cost_calculator.py        # Calcul coûts API Mistral
│   │
│   ├── config/
│   │   ├── settings.py               # Configuration globale
│   │   └── prompts.py                # Prompts système
│   │
│   └── simulation_workflow.py        # Orchestration simulation complète
│
├── data/
│   ├── models/
│   │   ├── random_forest_v2.pkl      # Modèle entraîné (18 features)
│   │   └── triage_dataset_v2.pkl     # Dataset d'entraînement
│   ├── rag_document/                 # Documents médicaux (PDF, Markdown)
│   │   ├── protocole_CIMU.pdf
│   │   ├── categories_triage.pdf
│   │   ├── constantes_vitales.md
│   │   ├── criteres_classification.md
│   │   ├── protocoles_action.md
│   │   └── signes_alerte.md
│   └── vector_db/                    # Base vectorielle ChromaDB (persistante)
│
└── notebooks/
    ├── generate_dataset_v2.ipynb     # Génération du dataset v2
    ├── train_model_v2.ipynb          # Entraînement Random Forest v2
    ├── feature_extraction.ipynb      # Exploration des features
    └── add_manual_cases.ipynb        # Ajout de cas manuels
```

---

## Installation

### Prérequis

- Python 3.10+
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### Étapes

```bash
# Cloner le projet
git clone <https://github.com/Nivrami/Triage_urgence_bis>
cd Triage_urgence_bis

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Editer .env et renseigner MISTRAL_API_KEY=<votre_clé>
```

### Lancement

```bash
streamlit run app/app.py
```

---

## Modules

### Agent infirmier (`src/agents/nurse_agent.py`)

Agent LLM basé sur Mistral qui conduit la conversation de triage. Il pose des questions médicales pertinentes et adaptées aux symptômes déclarés (une seule question à la fois), approfondit les signes de gravité (intensité, durée, irradiation, antécédents) et construit progressivement le résumé clinique transmis au prédicteur ML.

### Simulateur de patients (`src/agents/patient_simulator.py` + `patient_generator.py`)

Génère des profils patients synthétiques couvrant les 4 niveaux de gravité, avec constantes vitales et symptômes cohérents. Permet de tester et évaluer le système en mode automatique sans patient réel.

### Module RAG (`src/rag/`)

Pipeline de Retrieval-Augmented Generation sur une base documentaire de protocoles médicaux :

- **DocumentLoader** : charge et découpe les documents (PDF, Markdown) en chunks
- **EmbeddingProvider** : encode les textes en vecteurs (384 dimensions, `sentence-transformers`)
- **VectorStore** : stockage et recherche de similarité via ChromaDB (persistant)
- **Retriever** : sélectionne les top-k passages pertinents et les formate en contexte

Le RAG est utilisé à deux niveaux : enrichissement des prompts du NurseAgent pendant la conversation, et enrichissement des recommandations du prédicteur ML au moment du résultat final.

### Prédicteur ML (`src/rag/predictor.py`)

Classifieur Random Forest (`sklearn`) entraîné sur 18 features :

| # | Feature | Type |
|---|---------|------|
| 1-6 | FC, FR, SpO2, TA systolique, TA diastolique, Température | Constante vitale |
| 7-8 | Âge, Sexe | Info patient |
| 9-18 | Douleur thoracique, Dyspnée, Perte de connaissance, Hémorragie, Fracture, Fièvre élevée, Douleur abdominale, Nausée/vomissement, Symptôme mineur, Pas d'urgence | Binaire (symptôme présent/absent) |

Produit : niveau de gravité (ROUGE/JAUNE/VERT/GRIS), probabilités par classe, red flags détectés, recommandations RAG.

### Monitoring (`src/monitoring/`)

- **MetricsTracker** : suit les prédictions (niveau, âge, sexe, red flags, confiance) et les latences par composant (LLM, RAG, Predictor)
- **CostCalculator** : estime le coût des appels API Mistral en temps réel

---

## Performances du modèle

Modèle : `Random Forest` — 200 arbres, `class_weight='balanced'`, pipeline avec `StandardScaler`
Dataset : 129 cas équilibrés sur 4 classes (≈ 25% chacune)

| Métrique | Valeur |
|----------|--------|
| Accuracy (test set) | 73.1% |
| F1 weighted (test) | 0.722 |
| Accuracy (CV 5-fold) | **83.8% ± 8.8%** |
| F1 weighted (CV) | **0.832 ± 0.091** |

Performances par classe (test set) :

| Classe | Précision | Rappel | F1 |
|--------|-----------|--------|-----|
| 🔴 ROUGE | 1.00 | 0.86 | **0.92** |
| 🟡 JAUNE | 0.86 | 1.00 | **0.92** |
| 🟢 VERT | 0.50 | 0.33 | 0.40 |
| ⚪ GRIS | 0.56 | 0.71 | 0.62 |

> Le modèle est particulièrement fiable sur les cas critiques (ROUGE, F1=0.92), ce qui est l'objectif prioritaire en contexte d'urgences. La classe VERT est la moins bien discriminée (confusion avec GRIS et JAUNE).

---

## Points d'amélioration

### Données
- **Dataset limité (129 cas)** : principale limite du modèle. L'augmenter permettrait de réduire la variance (±8.8% en CV) et d'améliorer la classe VERT
- Mieux équilibrer les features binaires (`douleur abdominale` et `nausée vomissement` sont sur-représentées à ~73%)
- Ajouter des cas réels anonymisés ou des cas synthétiques plus variés

### Modèle ML
- Tester d'autres classifieurs (Gradient Boosting, XGBoost) ou un ensemble de modèles
- Optimisation des hyperparamètres par recherche sur grille (GridSearchCV)
- Seuils de décision ajustables selon le niveau de risque clinique acceptable

### RAG
- Enrichir la base documentaire (plus de protocoles, guidelines HAS, SFMU)
- Évaluer la pertinence des passages récupérés (métriques RAG : précision, rappel)
- Expérimenter avec un modèle d'embedding médical spécialisé (ex. `ClinicalBERT`)

### Application
- Authentification et gestion multi-utilisateurs
- Export des rapports de triage (PDF)
- Intégration d'un système de feedback pour améliorer le modèle en continu
- Tests unitaires et tests d'intégration à compléter
