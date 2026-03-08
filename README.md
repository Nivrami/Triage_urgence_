# Agent Triage Urgences - Projet LLM

## Description
Système d'aide au triage aux urgences hospitalières utilisant des LLM, RAG et ML.

## Structure du projet
```
triage_urgences/
├── config/          # Configuration et prompts
├── src/             # Code source
│   ├── models/      # Modèles de données (Pydantic)
│   ├── agents/      # Agents LLM
│   ├── llm/         # Providers LLM
│   ├── rag/         # Retrieval Augmented Generation
│   ├── ml/          # Machine Learning
│   ├── metrics/     # Tracking métriques
│   ├── workflows/   # Orchestration
│   └── utils/       # Utilitaires
├── data/            # Données
├── notebooks/       # Exploration
├── tests/           # Tests unitaires
└── app/             # Interface Streamlit
```

## Installation
```bash
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run app/main.py
```
