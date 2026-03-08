# Architecture du Projet Triage Urgence

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTERFACE UTILISATEUR                              │
│                              (Streamlit)                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   01_simulation.py  │  │  02_interactive.py  │  │     dashboard       │  │
│  │   (Mode Auto)       │  │  (Mode Manuel)      │  │     (Métriques)     │  │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  │
└─────────────┼────────────────────────┼────────────────────────┼─────────────┘
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COUCHE AGENTS                                   │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐           │
│  │      NurseAgent            │  │    PatientSimulator         │           │
│  │  (Agent Infirmier)         │◄─┤  (Simule un patient)        │           │
│  │                            │  │                             │           │
│  │  - Pose des questions      │  │  - Génère des symptômes     │           │
│  │  - Évalue la gravité       │  │  - Répond aux questions     │           │
│  │  - Utilise le RAG          │  │  - Simule des cas cliniques │           │
│  └──────────┬─────────────────┘  └─────────────────────────────┘           │
└─────────────┼───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COUCHE LLM                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MistralProvider                                 │    │
│  │                                                                      │    │
│  │   Prompt = Instructions + Contexte RAG + Historique conversation    │    │
│  │                                                                      │    │
│  └──────────┬──────────────────────────────────────────────────────────┘    │
└─────────────┼───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODULE RAG                                      │
│                    (Retrieval-Augmented Generation)                          │
│                                                                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │DocumentLoader │───►│EmbeddingProvider──►│  VectorStore  │                │
│  │               │    │               │    │  (ChromaDB)   │                │
│  │ - Charge PDF  │    │ - Texte→Vecteur   │               │                │
│  │ - Chunk docs  │    │ - 384 dimensions  │ - Stockage    │                │
│  └───────────────┘    └───────────────┘    │ - Recherche   │                │
│                                            └───────┬───────┘                │
│                                                    │                        │
│                       ┌───────────────┐            │                        │
│                       │   Retriever   │◄───────────┘                        │
│                       │               │                                     │
│                       │ - Recherche   │                                     │
│                       │ - Formate     │                                     │
│                       │ - Multi-query │                                     │
│                       └───────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODULE ML (À faire)                               │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐                │
│  │FeatureExtractor──►│ Preprocessor  │───►│  Classifier   │                │
│  │               │    │               │    │               │                │
│  │ Extraire les  │    │ Normaliser    │    │ Prédire la    │                │
│  │ features      │    │ les données   │    │ gravité       │                │
│  └───────────────┘    └───────────────┘    └───────────────┘                │
│                                                                              │
│                    Rôle: "Deuxième avis" pour valider le triage             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODULE METRICS                                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│  │ CostTracker   │  │LatencyTracker │  │ CarbonTracker │                    │
│  │               │  │               │  │               │                    │
│  │ Coût API ($)  │  │ Temps (ms)    │  │ CO2 (g)       │                    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                    │
│          │                  │                  │                            │
│          └──────────────────┼──────────────────┘                            │
│                             ▼                                               │
│                   ┌───────────────────┐                                     │
│                   │ MetricsAggregator │                                     │
│                   └───────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Flux de données : Triage d'un patient

```
┌──────────────┐
│   Patient    │
│  "J'ai mal   │
│   au ventre" │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  1. RÉCEPTION DE LA REQUÊTE                                  │
│     NurseAgent reçoit le message du patient                  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  2. RECHERCHE RAG                                            │
│                                                              │
│  "mal au ventre" ──► EmbeddingProvider ──► Vecteur [0.2,...]│
│                                                  │           │
│                                                  ▼           │
│                                            VectorStore       │
│                                            (recherche)       │
│                                                  │           │
│                                                  ▼           │
│                                     Documents pertinents:    │
│                                     - Appendicite            │
│                                     - Gastro-entérite        │
│                                     - Colique néphrétique    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  3. CONSTRUCTION DU PROMPT                                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ System: Tu es un infirmier aux urgences...             │  │
│  │                                                        │  │
│  │ Contexte RAG:                                          │  │
│  │ [1] Appendicite: douleur FID, fièvre, défense...      │  │
│  │ [2] Gastro: diarrhée, vomissements, déshydratation... │  │
│  │                                                        │  │
│  │ Historique: [messages précédents]                      │  │
│  │                                                        │  │
│  │ Patient: "J'ai mal au ventre"                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  4. APPEL LLM (Mistral)                                      │
│                                                              │
│     Prompt ──────► API Mistral ──────► Réponse              │
│                                                              │
│     "Depuis combien de temps avez-vous cette douleur?       │
│      Où exactement? Avez-vous de la fièvre?"                │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  5. ÉVALUATION FINALE                                        │
│                                                              │
│  Après plusieurs échanges:                                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  NIVEAU DE GRAVITÉ: ORANGE                             │  │
│  │                                                        │  │
│  │  Motif: Suspicion appendicite                          │  │
│  │  - Douleur FID depuis 12h                              │  │
│  │  - Fièvre 38.2°C                                       │  │
│  │  - Défense abdominale                                  │  │
│  │                                                        │  │
│  │  Action: Consultation chirurgien dans l'heure          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Structure des fichiers

```
triage_urgence/
│
├── app/                          # Interface Streamlit
│   ├── main.py                   # Point d'entrée
│   ├── pages/
│   │   ├── 01_simulation.py      # Mode simulation auto
│   │   └── 02_interactive.py     # Mode conversation
│   └── components/
│       ├── chat_interface.py     # Composant chat
│       ├── dashboard.py          # Métriques temps réel
│       └── patient_card.py       # Carte patient
│
├── src/                          # Code source
│   ├── agents/
│   │   ├── nurse_agent.py        # ✅ Agent infirmier
│   │   └── patient_simulator.py  # ✅ Simulateur patient
│   │
│   ├── llm/
│   │   ├── base_provider.py      # Interface abstraite
│   │   └── mistral_provider.py   # ✅ Provider Mistral
│   │
│   ├── rag/                      # ✅ MODULE COMPLET
│   │   ├── embeddings.py         # Texte → Vecteur
│   │   ├── vector_store.py       # Stockage ChromaDB
│   │   ├── document_loader.py    # Chargement documents
│   │   └── retriever.py          # Recherche + formatage
│   │
│   ├── ml/                       # ❌ À IMPLÉMENTER
│   │   ├── feature_extractor.py  # Extraction features
│   │   ├── preprocessor.py       # Prétraitement
│   │   └── classifier.py         # Classification gravité
│   │
│   └── metrics/                  # ❌ À IMPLÉMENTER
│       ├── cost_tracker.py       # Suivi coûts
│       ├── latency_tracker.py    # Suivi latence
│       ├── carbon_tracker.py     # Suivi carbone
│       └── metrics_aggregator.py # Agrégation
│
├── data/                         # Données
│   └── documents/                # PDFs médicaux
│
└── tests/                        # Tests
    ├── test_embeddings.py
    ├── test_vector_store.py
    └── test_rag_complete.py
```

---

## Légende des niveaux de gravité

```
┌─────────┬────────────────────────────────────────────────────────┐
│  ROUGE  │ Urgence vitale - Prise en charge immédiate            │
│         │ Ex: Arrêt cardiaque, détresse respiratoire            │
├─────────┼────────────────────────────────────────────────────────┤
│ ORANGE  │ Urgence vraie - Consultation rapide (< 1h)            │
│         │ Ex: Douleur thoracique, dyspnée modérée               │
├─────────┼────────────────────────────────────────────────────────┤
│  JAUNE  │ Urgence relative - Peut attendre (1-2h)               │
│         │ Ex: Fièvre élevée, fracture simple                    │
├─────────┼────────────────────────────────────────────────────────┤
│  VERT   │ Consultation non urgente                              │
│         │ Ex: Plaie superficielle, entorse légère               │
├─────────┼────────────────────────────────────────────────────────┤
│  BLEU   │ Orientation médecine de ville                         │
│         │ Ex: Renouvellement ordonnance, certificat             │
└─────────┴────────────────────────────────────────────────────────┘
```
