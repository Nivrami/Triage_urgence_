"""
Application principale - Système de Triage Intelligent aux Urgences.
"""

import sys
import json
import io
import time
import pickle
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Dict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import find_dotenv, load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv(find_dotenv())

from src.llm.llm_factory import LLMFactory
from src.agents.patient_generator import PatientGenerator
from src.agents.patient_simulator import PatientSimulator
from src.agents.nurse_agent import NurseAgent
from src.models.conversation import ConversationHistory
from src.rag.chatbot import TriageChatbotAPI
from src.rag.predictor import MLTriagePredictor
from src.rag.vector_store import VectorStore, RAGRetriever
from src.simulation_workflow import SimulationWorkflow
from src.monitoring.metrics_tracker import get_tracker
from src.monitoring.cost_calculator import get_calculator

# ---------------------------------------------------------------------------
# Config globale
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Triage Urgences - IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Palette médicale */
:root {
    --medical-blue: #1a5f7a;
    --medical-blue-light: #2d8ab5;
    --medical-teal: #57c5b6;
    --medical-bg: #f0f7fa;
    --medical-white: #ffffff;
    --medical-gray: #6b7280;
    --rouge: #dc2626;
    --jaune: #d97706;
    --vert: #16a34a;
    --gris: #6b7280;
}

/* Fond général */
.stApp {
    background-color: var(--medical-bg);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5f7a 0%, #0f3d52 100%);
}
[data-testid="stSidebar"] * {
    color: #e0f2fe !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #e0f2fe !important;
    font-weight: 500;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2);
}

/* Titres */
h1 { color: #1a5f7a !important; border-bottom: 3px solid #57c5b6; padding-bottom: 0.4rem; }
h2 { color: #1a5f7a !important; }
h3 { color: #2d8ab5 !important; }

/* Boutons principaux */
.stButton > button {
    background-color: #1a5f7a;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover {
    background-color: #2d8ab5;
    color: white;
}
.stButton > button[kind="primary"] {
    background-color: #16a34a;
}
.stButton > button[kind="secondary"] {
    background-color: #6b7280;
}

/* Métriques */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    border-left: 4px solid #57c5b6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* Messages chat */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.5rem;
}

/* Info / Warning / Success */
.stAlert {
    border-radius: 8px;
}

/* Expander */
.streamlit-expanderHeader {
    background: white;
    border-radius: 8px;
    color: #1a5f7a !important;
    font-weight: 600;
}

/* Cards de niveau de gravité */
.severity-badge {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

/* Divider */
hr { border-color: #c7dde8; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #1a5f7a;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #57c5b6;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# PAGE : ACCUEIL
# ===========================================================================

def page_accueil():
    st.title("🏥 Système de Triage Intelligent aux Urgences")
    st.markdown("*Aide à la décision médicale par intelligence artificielle*")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div style="background:white; border-radius:12px; padding:24px; border-top:4px solid #57c5b6; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
<h3 style="color:#1a5f7a;">🎲 Génération de conversations</h3>
<ul>
<li>Conversations automatiques infirmier / patient</li>
<li>Extraction automatique des informations médicales</li>
<li>Constantes vitales cohérentes avec la pathologie</li>
<li>Export des données pour machine learning</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style="background:white; border-radius:12px; padding:24px; border-top:4px solid #2d8ab5; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
<h3 style="color:#1a5f7a;">👤 Mode interactif</h3>
<ul>
<li>Jouez le rôle de l'infirmier de triage</li>
<li>Posez vos propres questions au patient IA</li>
<li>Collecte automatique des constantes vitales</li>
<li>Prédiction ML du niveau de gravité</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div style="background:white; border-radius:12px; padding:24px; border-left:4px solid #1a5f7a; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
<h3 style="color:#1a5f7a;">📊 Données générées</h3>
<p>Les conversations générées contiennent :</p>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
<div>• <b>Informations patient</b> : âge, sexe, symptômes, antécédents</div>
<div>• <b>Constantes vitales</b> : FC, FR, SpO2, TA, température</div>
<div>• <b>Historique complet</b> de la conversation</div>
<div>• <b>Format ML</b> : prêt pour l'entraînement de modèles</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_r, col_j, col_v, col_g = st.columns(4)
    with col_r:
        st.markdown('<div style="background:#dc2626;color:white;border-radius:10px;padding:16px;text-align:center;font-weight:bold;">🔴 ROUGE<br><small>Urgence vitale</small></div>', unsafe_allow_html=True)
    with col_j:
        st.markdown('<div style="background:#d97706;color:white;border-radius:10px;padding:16px;text-align:center;font-weight:bold;">🟡 JAUNE<br><small>Urgence relative</small></div>', unsafe_allow_html=True)
    with col_v:
        st.markdown('<div style="background:#16a34a;color:white;border-radius:10px;padding:16px;text-align:center;font-weight:bold;">🟢 VERT<br><small>Non urgent</small></div>', unsafe_allow_html=True)
    with col_g:
        st.markdown('<div style="background:#6b7280;color:white;border-radius:10px;padding:16px;text-align:center;font-weight:bold;">⚪ GRIS<br><small>Pas une urgence</small></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ À propos")
        st.markdown("Application de triage intelligent aux urgences.")
        st.markdown("**Modèle LLM** : Mistral AI")
        st.markdown("**Framework** : Streamlit")

    st.markdown(
        "<br><div style='text-align:center;color:#6b7280;'><small>🏥 Système de Triage Intelligent — 2025</small></div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE : SIMULATION AUTOMATIQUE
# ===========================================================================

PREDEFINED_PROFILES = {
    "Cas Grave - Infarctus": "Homme de 62 ans avec suspicion d'infarctus du myocarde",
    "Cas Modéré - Fracture": "Femme de 55 ans avec fracture suspectée du poignet après chute",
    "Cas Léger - Entorse": "Homme de 30 ans avec entorse de cheville",
    "Cas Critique - AVC": "Femme de 68 ans avec suspicion d'AVC, troubles de la parole et paralysie faciale",
    "Cas Urgent - Pneumonie": "Homme de 45 ans avec dyspnée importante et fièvre élevée",
}


def _init_simulation_state():
    defaults = {
        "simulation_started": False,
        "simulation_paused": False,
        "conversation_history": [],
        "patient": None,
        "patient_simulator": None,
        "nurse_agent": None,
        "conversation": None,
        "question_count": 0,
        "simulation_complete": False,
        "llm": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_simulation():
    st.session_state.simulation_started = False
    st.session_state.simulation_paused = False
    st.session_state.conversation_history = []
    st.session_state.patient = None
    st.session_state.patient_simulator = None
    st.session_state.nurse_agent = None
    st.session_state.conversation = None
    st.session_state.question_count = 0
    st.session_state.simulation_complete = False


def _start_simulation(pathology_description: str):
    try:
        if st.session_state.llm is None:
            with st.spinner("Initialisation du LLM Mistral..."):
                st.session_state.llm = LLMFactory.create("mistral", "mistral-small-latest")

        llm = st.session_state.llm

        with st.spinner("Génération du patient..."):
            generator = PatientGenerator(llm)
            patient = generator.generate_from_description(pathology_description)
            st.session_state.patient = patient

        with st.spinner("Création des agents..."):
            st.session_state.patient_simulator = PatientSimulator(llm, patient)
            st.session_state.nurse_agent = NurseAgent(llm, max_questions=6)

        initial_complaint = st.session_state.patient_simulator.get_initial_complaint()
        st.session_state.conversation = ConversationHistory()
        st.session_state.conversation.add_assistant_message(initial_complaint)
        st.session_state.conversation_history = [
            {"role": "patient", "content": initial_complaint, "type": "initial"}
        ]
        st.session_state.simulation_started = True
        st.session_state.question_count = 0
        st.success("Simulation démarrée avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du démarrage de la simulation: {e}")


def _step_simulation():
    if not st.session_state.simulation_started:
        st.warning("Veuillez démarrer la simulation d'abord.")
        return
    if st.session_state.simulation_complete:
        st.info("La simulation est terminée.")
        return

    nurse = st.session_state.nurse_agent
    patient_sim = st.session_state.patient_simulator

    if not nurse.should_continue():
        st.session_state.simulation_complete = True
        st.info("Simulation terminée - Nombre maximum de questions atteint.")
        return

    try:
        with st.spinner("L'infirmier réfléchit..."):
            question = nurse.ask_next_question()

        st.session_state.conversation_history.append(
            {"role": "nurse", "content": question, "question_num": st.session_state.question_count + 1}
        )
        st.session_state.conversation.add_user_message(question)
        nurse.add_to_history("nurse", question)

        with st.spinner("Le patient répond..."):
            response = patient_sim.respond(question)

        st.session_state.conversation_history.append({"role": "patient", "content": response})
        st.session_state.conversation.add_assistant_message(response)
        nurse.add_to_history("patient", response)
        st.session_state.question_count += 1

        if not nurse.should_continue():
            st.session_state.simulation_complete = True
            st.success("Simulation terminée!")
    except Exception as e:
        st.error(f"Erreur lors de l'étape: {e}")


def _run_all_simulation():
    if not st.session_state.simulation_started:
        st.warning("Veuillez démarrer la simulation d'abord.")
        return

    nurse = st.session_state.nurse_agent
    progress_bar = st.progress(0)
    status_text = st.empty()
    iteration = 0

    while nurse.should_continue() and iteration < 10:
        status_text.text(f"Question {st.session_state.question_count + 1}/{nurse.max_questions}")
        _step_simulation()
        iteration += 1
        progress_bar.progress(min(iteration / nurse.max_questions, 1.0))

    progress_bar.empty()
    status_text.empty()
    st.success("Simulation complète terminée!")


def _extract_symptoms_from_history() -> Optional[Dict]:
    if not st.session_state.llm or not st.session_state.conversation_history:
        return None

    history_text = ""
    for msg in st.session_state.conversation_history:
        role = "Patient" if msg["role"] == "patient" else "Infirmier"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""
Analyse la conversation médicale suivante entre un infirmier de triage et un patient.
Extrait les symptômes mentionnés par le patient sous forme de liste structurée.

Format de sortie attendu (JSON uniquement) :
{{
    "symptomes_principaux": ["symptome1", "symptome2"],
    "localisation": "zone géographique du corps",
    "intensite_douleur": "0-10 ou inconnue",
    "duree": "depuis combien de temps",
    "facteurs_aggravants": ["facteur1"]
}}

Conversation :
{history_text}
"""
    try:
        response = st.session_state.llm.generate(prompt)
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des symptômes : {e}")
        return None


def page_simulation():
    st.title("🤖 Simulation Automatique de Triage")
    st.markdown("*Dialogue automatique entre un agent infirmier IA et un patient virtuel.*")

    _init_simulation_state()

    # Formulaires identité + constantes
    st.info("Saisie des informations cliniques initiales")
    col1, col2 = st.columns(2)

    with col1:
        with st.form("form_identite"):
            st.subheader("Identité")
            num_patient = st.text_input("Numéro de patient", placeholder="Ex: PAT-2024-001")
            age = st.number_input("Âge du patient", min_value=0, max_value=120, value=30)
            genre = st.selectbox("Genre", ["Homme", "Femme", "Autre"])
            if st.form_submit_button("Enregistrer l'identité"):
                st.session_state.id_data = {"num": num_patient, "age": age, "genre": genre}
                st.toast("Identité enregistrée")

    with col2:
        with st.form("form_constantes"):
            st.subheader("Constantes")
            c1, c2 = st.columns(2)
            with c1:
                fc = st.number_input("FC (bpm)", min_value=0, max_value=250, value=75)
                fr = st.number_input("FR (/min)", min_value=0, max_value=60, value=16)
                temp = st.number_input("T° (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
            with c2:
                tas = st.number_input("TA Systolique", min_value=40, max_value=250, value=120)
                tad = st.number_input("TA Diastolique", min_value=30, max_value=150, value=80)
                spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)
            if st.form_submit_button("Enregistrer les constantes"):
                st.session_state.const_data = {"fc": fc, "fr": fr, "temp": temp, "tas": tas, "tad": tad, "spo2": spo2}
                st.toast("Constantes enregistrées")

    st.divider()

    # Sélection profil patient
    st.subheader("Sélection du Profil Patient")
    profile_type = st.radio("Type de profil", ["Profil prédéfini", "Profil personnalisé"], horizontal=True)
    pathology_description = None

    if profile_type == "Profil prédéfini":
        selected = st.selectbox("Choisissez un profil", list(PREDEFINED_PROFILES.keys()))
        pathology_description = PREDEFINED_PROFILES[selected]
        st.info(f"Description: {pathology_description}")
    else:
        pathology_description = st.text_area(
            "Description du cas médical",
            placeholder="Ex: Femme de 35 ans avec migraine sévère et vomissements",
            height=100,
        )

    st.divider()

    # Boutons de contrôle
    col1, col2, col3, col4 = st.columns(4)
    action = None
    with col1:
        if st.button("Démarrer", disabled=st.session_state.simulation_started, use_container_width=True):
            action = "start"
    with col2:
        if st.button("Étape suivante", disabled=not st.session_state.simulation_started or st.session_state.simulation_complete, use_container_width=True):
            action = "step"
    with col3:
        if st.button("Tout exécuter", disabled=not st.session_state.simulation_started or st.session_state.simulation_complete, use_container_width=True):
            action = "run_all"
    with col4:
        if st.button("Réinitialiser", use_container_width=True):
            action = "reset"

    if action == "start" and pathology_description:
        st.session_state.pop("extracted_symptoms", None)
        _start_simulation(pathology_description)
        st.rerun()
    elif action == "start":
        st.warning("Veuillez saisir une description de cas médical.")
    elif action == "step":
        _step_simulation()
        st.rerun()
    elif action == "run_all":
        _run_all_simulation()
        st.rerun()
    elif action == "reset":
        _reset_simulation()
        st.session_state.pop("extracted_symptoms", None)
        st.rerun()

    st.divider()

    if st.session_state.patient:
        patient = st.session_state.patient
        with st.expander("Informations du Patient", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Identité:**")
                st.write(f"• Nom: {patient.prenom} {patient.nom}")
                st.write(f"• Âge: {patient.age} ans")
                st.write(f"• Sexe: {patient.sexe}")
                st.write(f"• Depuis: {patient.duree_symptomes}")
            with c2:
                st.markdown("**Symptômes:**")
                for s in patient.symptomes_exprimes:
                    st.write(f"• {s}")
            if patient.constantes:
                c = patient.constantes
                st.markdown("**Constantes Vitales:**")
                cc1, cc2, cc3, cc4, cc5 = st.columns(5)
                cc1.metric("FC", f"{c.fc} bpm")
                cc2.metric("FR", f"{c.fr} /min")
                cc3.metric("SpO2", f"{c.spo2}%")
                cc4.metric("TA", f"{c.ta_systolique}/{c.ta_diastolique}")
                cc5.metric("T°", f"{c.temperature}°C")
            if patient.antecedents:
                st.markdown("**Antécédents:**")
                st.write(", ".join(patient.antecedents))

    st.subheader("Conversation")
    messages = st.session_state.conversation_history
    if not messages:
        st.info("La conversation apparaîtra ici une fois démarrée.")
    else:
        for msg in messages:
            if msg["role"] == "patient":
                with st.chat_message("assistant", avatar="🤕"):
                    prefix = "**Plainte initiale:**\n\n" if msg.get("type") == "initial" else ""
                    st.markdown(f"{prefix}{msg['content']}")
            else:
                with st.chat_message("user", avatar="👨‍⚕️"):
                    st.markdown(f"**Question {msg.get('question_num', '')}:**\n\n{msg['content']}")

    if st.session_state.simulation_started:
        st.divider()
        st.subheader("Métriques de la Simulation")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Questions posées", st.session_state.question_count)
        mc2.metric("Messages échangés", len(messages))
        max_q = st.session_state.nurse_agent.max_questions if st.session_state.nurse_agent else 6
        mc3.metric("Progression", f"{(st.session_state.question_count / max_q) * 100:.0f}%")

    if st.session_state.simulation_complete:
        st.divider()
        st.subheader("Analyse Automatique des Symptômes")
        if st.button("Extraire les symptômes pour classification"):
            with st.spinner("Analyse du dialogue en cours..."):
                extracted = _extract_symptoms_from_history()
                if extracted:
                    st.session_state.extracted_symptoms = extracted

        if "extracted_symptoms" in st.session_state:
            data = st.session_state.extracted_symptoms
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Symptômes détectés :**")
                for s in data.get("symptomes_principaux", []):
                    st.markdown(f"- {s}")
            with c2:
                st.write(f"**Localisation :** {data.get('localisation', 'N/A')}")
                st.write(f"**Intensité :** {data.get('intensite_douleur', 'N/A')}")
            st.info("Format JSON prêt pour l'algorithme de classification :")
            st.json(data)

        st.success("La simulation est terminée. Vous pouvez maintenant procéder au triage final.")


# ===========================================================================
# PAGE : CHAT INTERACTIF
# ===========================================================================

def page_chat_interactif():
    st.title("💬 Chatbot de Triage des Urgences")
    st.markdown("*Assistant ML pour aide à la décision — joue le rôle de l'infirmier*")

    if "chatbot" not in st.session_state:
        retriever = None
        with st.spinner("Chargement RAG..."):
            try:
                vector_db_path = ROOT_DIR / "data" / "vector_db"
                vector_store = VectorStore(
                    persist_directory=str(vector_db_path), collection_name="triage_medical"
                )
                retriever = RAGRetriever(vector_store=vector_store)
                st.session_state.predictor = MLTriagePredictor(rag_retriever=retriever)
                st.success("RAG chargé avec succès")
            except Exception as e:
                st.warning(f"RAG non chargé: {e}")
                st.session_state.predictor = MLTriagePredictor()

        st.session_state.chatbot = TriageChatbotAPI(retriever=retriever)
        st.session_state.messages = []
        st.session_state.started = False
        st.session_state.prediction = None
        st.session_state.chat_max_questions = 5

    bot = st.session_state.chatbot
    predictor = st.session_state.predictor
    data = bot.data

    # ------------------------------------------------------------------
    # Formulaires identité + constantes en haut de page
    # ------------------------------------------------------------------
    st.info("📋 Saisie des informations cliniques initiales")
    col1, col2 = st.columns(2)

    with col1:
        with st.form("chat_form_identite"):
            st.subheader("👤 Identité du patient")
            prenom = st.text_input("Prénom", placeholder="Ex: Jean")
            age_val = st.number_input("Âge", min_value=1, max_value=120, value=40)
            genre = st.selectbox("Genre", ["Homme", "Femme", "Autre"])
            if st.form_submit_button("✅ Enregistrer l'identité", use_container_width=True):
                st.session_state.chat_id_data = {"prenom": prenom, "age": age_val, "genre": genre}
                st.rerun()
        if "chat_id_data" in st.session_state:
            d = st.session_state.chat_id_data
            st.success(f"✓ {d['prenom']}, {d['age']} ans, {d['genre']}")

    with col2:
        with st.form("chat_form_constantes"):
            st.subheader("🌡️ Constantes vitales")
            ca, cb = st.columns(2)
            with ca:
                fc = st.number_input("FC (bpm)", min_value=0, max_value=250, value=75)
                fr = st.number_input("FR (/min)", min_value=0, max_value=60, value=16)
                temp = st.number_input("T° (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
            with cb:
                tas = st.number_input("TA Systolique", min_value=40, max_value=250, value=120)
                tad = st.number_input("TA Diastolique", min_value=30, max_value=150, value=80)
                spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)
            if st.form_submit_button("✅ Enregistrer les constantes", use_container_width=True):
                st.session_state.chat_const_data = {
                    "FC": fc, "FR": fr, "Temperature": temp,
                    "TA_systolique": tas, "TA_diastolique": tad, "SpO2": spo2,
                }
                st.rerun()
        if "chat_const_data" in st.session_state:
            c = st.session_state.chat_const_data
            st.success(f"✓ FC {c['FC']} bpm | T° {c['Temperature']}°C | SpO2 {c['SpO2']}% | TA {c['TA_systolique']}/{c['TA_diastolique']}")

    st.divider()

    with st.sidebar:
        st.header("Paramètres")
        max_questions = st.slider(
            "Nombre de questions",
            min_value=1,
            max_value=10,
            value=st.session_state.get("chat_max_questions", 5),
            help="Nombre de questions posées par le chatbot avant de clôturer le dossier",
        )
        if max_questions != st.session_state.get("chat_max_questions", 5):
            st.session_state.chat_max_questions = max_questions
            bot.max_questions = max_questions

        st.divider()
        st.header("Dossier patient")
        st.subheader("Identité")
        st.write(f"**Prénom:** {data.get('name') or '—'}")
        st.write(f"**Âge:** {data.get('age') or '—'}")
        sex = "Homme" if data.get("sex") == "H" else "Femme" if data.get("sex") == "F" else "—"
        st.write(f"**Sexe:** {sex}")
        st.divider()

        st.subheader("Symptômes")
        if data["symptoms"]:
            for s in data["symptoms"]:
                st.write(f"• {s}")
        else:
            st.write("—")
        st.divider()

        st.subheader("Constantes vitales")
        v = data["vitals"]
        count = len([k for k in ["Temperature", "FC", "TA_systolique", "SpO2", "FR"] if k in v])
        st.write(f"**Progression: {count}/5**")
        if v:
            if "Temperature" in v:
                st.write(f"Temp: {v['Temperature']}°C")
            if "FC" in v:
                st.write(f"FC: {v['FC']} bpm")
            if "TA_systolique" in v:
                st.write(f"TA: {v['TA_systolique']}/{v.get('TA_diastolique', '?')}")
            if "SpO2" in v:
                st.write(f"SpO2: {v['SpO2']}%")
            if "FR" in v:
                st.write(f"FR: {v['FR']}/min")
        else:
            st.write("—")
        st.divider()

        if st.button("🔄 Nouvelle conversation", use_container_width=True):
            bot.reset()
            st.session_state.messages = []
            st.session_state.started = False
            st.session_state.prediction = None
            st.session_state.pop("chat_id_data", None)
            st.session_state.pop("chat_const_data", None)
            st.rerun()

        ready = bot.is_ready_for_prediction()
        if st.button("Obtenir prédiction ML", use_container_width=True, disabled=not ready, help="5/5 constantes requises"):
            with st.spinner("Analyse ML en cours..."):
                summary = bot.get_summary()
                st.session_state.prediction = predictor.predict(summary)
            st.rerun()

    col1, col2 = st.columns([2, 1])

    forms_complete = "chat_id_data" in st.session_state and "chat_const_data" in st.session_state

    with col1:
        st.subheader("💬 Conversation")
        if not st.session_state.started:
            if not forms_complete:
                st.warning("Veuillez remplir et enregistrer l'identité et les constantes vitales avant de démarrer.")
            if st.button("🚀 Démarrer la consultation", use_container_width=True, disabled=not forms_complete):
                d = st.session_state.chat_id_data
                bot.data["name"] = d["prenom"]
                bot.data["age"] = d["age"]
                bot.data["sex"] = "H" if d["genre"] == "Homme" else "F"
                bot.data["vitals"].update(st.session_state.chat_const_data)
                bot.max_questions = st.session_state.get("chat_max_questions", 5)
                msg = bot.start()
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.session_state.started = True
                st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.started and not st.session_state.prediction:
            user_input = st.chat_input("Votre message...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                response = bot.chat(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    with col2:
        st.subheader("Résultat ML")
        if st.session_state.prediction:
            r = st.session_state.prediction
            st.markdown(
                f"""<div style="background-color:{r['color']};padding:20px;border-radius:10px;
                text-align:center;font-size:22px;font-weight:bold;
                color:{'white' if r['severity_level'] in ['ROUGE','GRIS'] else 'black'};
                margin-bottom:20px;">{r['label']}</div>""",
                unsafe_allow_html=True,
            )
            st.info(f"**Action:** {r['action']}")

            if r.get("red_flags"):
                st.warning("**Signes de gravité:**")
                for f in r["red_flags"]:
                    st.markdown(f"- {f}")

            st.subheader("Probabilités ML")
            for lvl in ["ROUGE", "JAUNE", "VERT", "GRIS"]:
                p = r["probabilities"].get(lvl, 0)
                st.progress(p, text=f"{lvl}: {p*100:.1f}%")

            st.metric("Confiance", f"{r['confidence']*100:.1f}%")

            if r.get("features_used"):
                with st.expander("Features ML"):
                    st.json(r["features_used"])

            if r.get("rag_sources"):
                with st.expander("Sources RAG"):
                    for source in r["rag_sources"]:
                        st.markdown(f"- {source}")

            with st.expander("Justification ML + RAG", expanded=True):
                st.markdown(r["justification"])

            st.divider()
            if st.button("Exporter rapport", use_container_width=True):
                export = f"# RAPPORT TRIAGE ML\n\n## {r['label']}\n"
                export += f"**Action:** {r['action']}\n**Confiance:** {r['confidence']*100:.1f}%\n\n"
                if r.get("red_flags"):
                    export += "## Drapeaux rouges\n" + "".join(f"- {f}\n" for f in r["red_flags"])
                export += "\n## Conversation\n\n"
                for m in st.session_state.messages:
                    role = "Assistant" if m["role"] == "assistant" else "Patient"
                    export += f"**{role}:** {m['content']}\n\n"
                st.download_button("Télécharger", export, "rapport_triage_ml.md", "text/markdown", use_container_width=True)
        else:
            st.info("Complétez la conversation puis cliquez 'Obtenir prédiction ML'")
            with st.expander("Aide"):
                st.markdown(
                    """
**Constantes requises (5/5):**
- Température
- Fréquence cardiaque (FC)
- Tension artérielle (TA)
- Saturation oxygène (SpO2)
- Fréquence respiratoire (FR)

Le bot vous guidera étape par étape.
"""
                )

    st.divider()
    st.caption("ML (Random Forest) + RAG (Documents médicaux) + Mistral API | Outil d'aide - ne remplace pas avis médical")


# ===========================================================================
# PAGE : GÉNÉRATION DE DONNÉES
# ===========================================================================

def page_generation():
    st.title("🎲 Génération de Conversations")
    st.markdown("*Générez des conversations automatiques pour constituer un dataset de triage médical.*")

    with st.sidebar:
        st.header("Paramètres")
        max_turns = st.slider("Nombre max de questions", min_value=3, max_value=15, value=8)
        st.markdown("---")
        st.markdown("**Astuce** : Laissez la pathologie vide pour une génération aléatoire !")

    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuration")
        pathology_input = st.text_input(
            "Pathologie (optionnel)",
            placeholder="Ex: Homme de 65 ans avec infarctus",
        )

    with col2:
        st.subheader("Actions")

        if st.button("Générer 1 conversation", type="primary"):
            with st.spinner("Génération en cours..."):
                try:
                    llm = LLMFactory.create("mistral", "mistral-large-latest")
                    workflow = SimulationWorkflow(llm, max_turns=max_turns)
                    pathology = pathology_input.strip() or None
                    log_stream = io.StringIO()
                    start_time = time.time()
                    with redirect_stdout(log_stream):
                        result = workflow.run_simulation(pathology=pathology)
                    duration = time.time() - start_time

                    try:
                        tracker = get_tracker()
                        tracker.track_latency("Generation", "conversation", duration)
                        tracker.track_api_call("mistral", "mistral-large-latest", 500, 300, duration, True)
                    except Exception:
                        pass

                    st.session_state.current_result = result
                    st.session_state.conversations.append(workflow.export_for_ml())
                    st.success(f"Conversation générée en {duration:.2f}s!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")

        if st.button("Générer 10 conversations"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                llm = LLMFactory.create("mistral", "mistral-large-latest")
                workflow = SimulationWorkflow(llm, max_turns=max_turns)
                total_duration = 0
                for i in range(10):
                    status_text.text(f"Génération {i+1}/10...")
                    progress_bar.progress((i + 1) / 10)
                    log_stream = io.StringIO()
                    start_time = time.time()
                    with redirect_stdout(log_stream):
                        workflow.run_simulation()
                    duration = time.time() - start_time
                    total_duration += duration
                    st.session_state.conversations.append(workflow.export_for_ml())
                    workflow.reset()
                    try:
                        tracker = get_tracker()
                        tracker.track_latency("Generation", "conversation", duration)
                        tracker.track_api_call("mistral", "mistral-large-latest", 500, 300, duration, True)
                    except Exception:
                        pass
                st.success(f"10 conversations générées en {total_duration:.1f}s!")
                progress_bar.empty()
                status_text.empty()
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown("---")

    if st.session_state.current_result:
        result = st.session_state.current_result
        st.subheader("Dernière conversation générée")
        st.info(f"**Pathologie :** {result['pathology']}")

        extr = result["extracted_patient"]
        orig = result["original_patient"]

        st.markdown("---")
        st.markdown("### Mesure des Constantes Vitales")

        with st.expander("Comment sont générées les constantes ?"):
            st.markdown(
                """
**Processus automatique :**
1. L'IA analyse la pathologie simulée
2. Génère des constantes **cohérentes** avec cette pathologie
3. L'infirmier "mesure" ces valeurs avec ses appareils
"""
            )

        if orig.constantes:
            c = orig.constantes
            c1, c2, c3 = st.columns(3)

            with c1:
                fc_note = "Normale" if 60 <= c.fc <= 100 else "Légèrement élevée" if c.fc > 100 else "Basse"
                st.metric("Fréquence Cardiaque (FC)", f"{c.fc} bpm", help=fc_note)
                fr_note = "Normale" if 12 <= c.fr <= 20 else "Élevée" if c.fr > 20 else "Basse"
                st.metric("Fréquence Respiratoire (FR)", f"{c.fr} /min", help=fr_note)
            with c2:
                spo2_note = "Normale" if c.spo2 >= 95 else "Basse - Hypoxie" if c.spo2 < 90 else "Légèrement basse"
                st.metric("Saturation Oxygène (SpO2)", f"{c.spo2}%", help=spo2_note)
                temp_note = "Normale" if 36.5 <= c.temperature <= 37.5 else "Fièvre" if c.temperature > 37.5 else "Hypothermie"
                st.metric("Température", f"{c.temperature}°C", help=temp_note)
            with c3:
                ta_note = "Normale" if 100 <= c.ta_systolique <= 140 else "Élevée (HTA)" if c.ta_systolique > 140 else "Basse"
                st.metric("Tension Artérielle (TA)", f"{c.ta_systolique}/{c.ta_diastolique} mmHg", help=ta_note)
                st.metric("Patient", f"{extr.age or orig.age} ans, {extr.sexe or orig.sexe}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Conversation", "Données Patient", "Export ML"])

        with tab1:
            st.markdown("### Conversation complète")
            for msg in result["conversation"].messages:
                content = msg.content
                if '"' in content:
                    parts = content.split('"')
                    if len(parts) > 1:
                        content = parts[0] + '"'
                if "Explication" in content:
                    content = content.split("Explication")[0].strip()
                if msg.role.value == "user":
                    st.markdown(f"**Infirmier :** {content}")
                else:
                    st.markdown(f"**Patient :** {content}")
                st.markdown("")

        with tab2:
            st.markdown("### Informations extraites de la conversation")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Identité**")
                st.write(f"- Âge : {extr.age} ans")
                st.write(f"- Sexe : {extr.sexe}")
                st.markdown("**Symptômes déclarés**")
                for s in (extr.symptomes_exprimes or ["Aucun symptôme extrait"]):
                    st.write(f"- {s}")
                if extr.duree_symptomes:
                    st.markdown(f"**Durée :** {extr.duree_symptomes}")
            with cb:
                st.markdown("**Antécédents médicaux**")
                for a in (extr.antecedents or ["Aucun antécédent déclaré"]):
                    st.write(f"- {a}")
                completeness = result["completeness"]
                st.markdown("**Complétude de l'information**")
                st.progress(completeness["score"])
                st.write(f"{completeness['score']*100:.0f}% des informations collectées")
                if completeness["missing"]:
                    st.warning(f"Manquant : {', '.join(completeness['missing'])}")

        with tab3:
            st.markdown("### Données pour Machine Learning")
            llm = LLMFactory.create("mistral", "mistral-large-latest")
            workflow = SimulationWorkflow(llm)
            workflow.original_patient = result["original_patient"]
            workflow.extracted_patient = result["extracted_patient"]
            workflow.pathology = result["pathology"]
            workflow.conversation = result["conversation"]
            ml_data = workflow.export_for_ml()
            st.json(ml_data)
            st.download_button(
                "Télécharger (JSON)",
                json.dumps(ml_data, indent=2, ensure_ascii=False),
                "conversation_triage.json",
                "application/json",
            )

        st.markdown("---")
        st.subheader("Prédiction de Gravité")

        if st.button("Prédire le niveau de gravité", type="primary"):
            model_path = ROOT_DIR / "data" / "models" / "random_forest_simple.pkl"
            if not model_path.exists():
                st.error("Modèle non trouvé")
            else:
                with st.spinner("Chargement du modèle et prédiction..."):
                    start_time = time.time()
                    with open(model_path, "rb") as f:
                        clf = pickle.load(f)

                    llm = LLMFactory.create("mistral", "mistral-large-latest")
                    workflow = SimulationWorkflow(llm)
                    workflow.original_patient = result["original_patient"]
                    workflow.extracted_patient = result["extracted_patient"]
                    workflow.pathology = result["pathology"]
                    workflow.conversation = result["conversation"]
                    ml_data = workflow.export_for_ml()

                    fc = ml_data.get("fc", 80)
                    fr = ml_data.get("fr", 16)
                    spo2 = ml_data.get("spo2", 98)
                    ta_sys = ml_data.get("ta_systolique", 120)
                    ta_dia = ml_data.get("ta_diastolique", 80)
                    temp = ml_data.get("temperature", 37.0)
                    age = ml_data.get("age", 50)
                    sexe = ml_data.get("sexe", "M")

                    features = np.array([[
                        (fc - 70) / 30, (fr - 16) / 5, (spo2 - 95) / 5,
                        (ta_sys - 120) / 20, (ta_dia - 80) / 10,
                        (temp - 37) / 2, (age - 50) / 25,
                        1 if sexe == "M" else 0,
                    ]])

                    prediction = clf.predict(features)[0]
                    probas = clf.predict_proba(features)[0]
                    proba_dict = dict(zip(clf.classes_, probas))
                    duration = time.time() - start_time

                    try:
                        tracker = get_tracker()
                        tracker.track_prediction(prediction, age, sexe[0], [], [], max(probas))
                        tracker.track_latency("Predictor_ML", "predict", duration)
                    except Exception:
                        pass

                    colors = {"ROUGE": "🔴", "JAUNE": "🟡", "VERT": "🟢", "GRIS": "⚪"}
                    descriptions = {
                        "ROUGE": "**Urgence vitale immédiate** - Pronostic vital engagé",
                        "JAUNE": "**Urgent mais non vital** - Prise en charge rapide nécessaire",
                        "VERT": "**Non urgent** - Peut attendre",
                        "GRIS": "**Ne nécessite pas les urgences**",
                    }

                    st.success(f"### {colors[prediction]} Gravité prédite : **{prediction}**")
                    st.info(descriptions[prediction])

                    st.markdown("#### Probabilités par classe")
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    for col, label in zip([pc1, pc2, pc3, pc4], ["ROUGE", "JAUNE", "VERT", "GRIS"]):
                        with col:
                            prob = proba_dict.get(label, 0)
                            col.metric(f"{colors[label]} {label}", f"{prob*100:.1f}%")
                            col.progress(prob)

                    with st.expander("Détails de la prédiction"):
                        dc1, dc2, dc3 = st.columns(3)
                        with dc1:
                            st.write(f"- FC : {fc} bpm")
                            st.write(f"- FR : {fr} /min")
                            st.write(f"- SpO2 : {spo2}%")
                        with dc2:
                            st.write(f"- TA : {ta_sys}/{ta_dia} mmHg")
                            st.write(f"- Temp : {temp}°C")
                        with dc3:
                            st.write(f"- Âge : {age} ans")
                            st.write(f"- Sexe : {sexe}")

    st.markdown("---")
    st.subheader("Dataset complet")

    if st.session_state.conversations:
        st.info(f"**{len(st.session_state.conversations)} conversations** générées")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.download_button(
                "Télécharger dataset (JSON)",
                json.dumps(st.session_state.conversations, indent=2, ensure_ascii=False),
                "dataset_triage.json",
                "application/json",
            )
        with dc2:
            df = pd.DataFrame(st.session_state.conversations)
            st.download_button("Télécharger dataset (CSV)", df.to_csv(index=False), "dataset_triage.csv", "text/csv")
        with dc3:
            if st.button("Effacer le dataset"):
                st.session_state.conversations = []
                st.session_state.current_result = None
                st.rerun()

        with st.expander("Aperçu du dataset"):
            st.dataframe(pd.DataFrame(st.session_state.conversations), use_container_width=True)
    else:
        st.info("Aucune conversation générée. Cliquez sur 'Générer' pour commencer !")


# ===========================================================================
# PAGE : MONITORING
# ===========================================================================

def page_monitoring():
    st.title("📊 Monitoring du Système")
    st.markdown("*Suivi des coûts API, performances et statistiques de prédiction*")

    tracker = get_tracker()
    calculator = get_calculator()

    with st.sidebar:
        st.header("Actions")
        if st.button("Rafraîchir", use_container_width=True):
            st.rerun()
        if st.button("Export CSV", use_container_width=True):
            export_path = tracker.export_csv()
            st.success(f"Exporté vers: {export_path}")
        if st.button("Reset métriques", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_reset"):
                tracker.reset()
                st.success("Métriques réinitialisées")
                st.session_state.confirm_reset = False
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Cliquez à nouveau pour confirmer")

    api_stats = tracker.get_api_stats()
    cost_data = calculator.calculate_total_cost(tracker.api_calls)
    days_elapsed = max(1, (datetime.now() - datetime.fromisoformat(tracker.api_calls[0]["timestamp"])).days) if tracker.api_calls else 1
    monthly_estimate = calculator.estimate_monthly_cost(cost_data["total_cost"], days_elapsed)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Coût Total", calculator.format_cost(cost_data["total_cost"]))
    mc2.metric("Appels API", api_stats["total_calls"])
    mc3.metric("Latence Moyenne", f"{api_stats['avg_latency']:.2f}s" if api_stats["avg_latency"] > 0 else "N/A")
    pred_stats = tracker.get_prediction_stats()
    mc4.metric(
        "Prédictions",
        pred_stats["total"],
        delta=f"{pred_stats['avg_confidence']*100:.0f}% confiance" if pred_stats["total"] > 0 else None,
    )

    st.divider()
    st.header("Analyse des Coûts")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Répartition par Service")
        if cost_data["total_cost"] > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Mistral API", "Embeddings"],
                values=[cost_data["mistral"]["cost"], cost_data["embeddings"]["cost"]],
                hole=0.4,
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de coût disponible")
        st.markdown("**Mistral API**")
        st.write(f"• Appels: {cost_data['mistral']['calls']}")
        st.write(f"• Tokens input: {cost_data['mistral']['tokens_input']:,}")
        st.write(f"• Tokens output: {cost_data['mistral']['tokens_output']:,}")
        st.write(f"• Coût: {calculator.format_cost(cost_data['mistral']['cost'])}")

    with c2:
        st.subheader("Évolution des Coûts")
        if tracker.api_calls:
            costs_over_time = []
            cumulative_cost = 0
            for call in tracker.api_calls:
                if call.get("service") == "mistral":
                    cumulative_cost += calculator.calculate_mistral_cost(
                        call.get("model", "mistral-small-latest"),
                        call["tokens_input"],
                        call["tokens_output"],
                    )["cost_total"]
                costs_over_time.append({"timestamp": datetime.fromisoformat(call["timestamp"]), "cost": cumulative_cost})
            df_cost = pd.DataFrame(costs_over_time)
            fig2 = px.line(df_cost, x="timestamp", y="cost", title="Coût Cumulé")
            fig2.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée disponible")

    st.divider()
    st.header("Performances")
    latency_stats = tracker.get_latency_stats()

    if latency_stats:
        lc1, lc2 = st.columns(2)
        with lc1:
            st.subheader("Latences par Composant")
            df_lat = pd.DataFrame([
                {"Composant": comp, "Moyenne (s)": f"{s['avg']:.3f}", "Min (s)": f"{s['min']:.3f}", "Max (s)": f"{s['max']:.3f}", "Appels": s["count"]}
                for comp, s in latency_stats.items()
            ])
            st.dataframe(df_lat, use_container_width=True, hide_index=True)
        with lc2:
            st.subheader("Distribution des Latences")
            latency_data = [{"Composant": l["component"], "Durée (s)": l["duration"]} for l in tracker.latencies]
            if latency_data:
                fig3 = px.box(pd.DataFrame(latency_data), x="Composant", y="Durée (s)", title="Distribution")
                fig3.update_layout(height=300)
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Aucune donnée de latence disponible")

    st.divider()
    st.header("Analyse des Prédictions")

    if pred_stats["total"] > 0:
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.subheader("Répartition par Gravité")
            fig4 = px.pie(
                pd.DataFrame([{"Niveau": k, "Nombre": v} for k, v in pred_stats["by_severity"].items()]),
                values="Nombre", names="Niveau",
                color="Niveau",
                color_discrete_map={"ROUGE": "#FF0000", "JAUNE": "#FFD700", "VERT": "#00FF00", "GRIS": "#808080"},
            )
            fig4.update_layout(height=300)
            st.plotly_chart(fig4, use_container_width=True)
        with pc2:
            st.subheader("Statistiques")
            for severity, count in pred_stats["by_severity"].items():
                st.write(f"**{severity}**: {count} ({count/pred_stats['total']*100:.1f}%)")
            st.write(f"**Confiance moyenne**: {pred_stats['avg_confidence']*100:.1f}%")
        with pc3:
            st.subheader("Dernières Prédictions")
            for pred in tracker.predictions[-5:][::-1]:
                ts = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M:%S")
                st.write(f"**{ts}** - {pred['severity']} ({pred['confidence']*100:.0f}%)")
    else:
        st.info("Aucune prédiction disponible")

    st.divider()
    with st.expander("Détails Techniques"):
        t1, t2, t3 = st.tabs(["Appels API", "Latences", "Prédictions"])
        with t1:
            if tracker.api_calls:
                df_api = pd.DataFrame(tracker.api_calls)
                df_api["timestamp"] = pd.to_datetime(df_api["timestamp"])
                st.dataframe(df_api, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun appel API enregistré")
        with t2:
            if tracker.latencies:
                df_lat2 = pd.DataFrame(tracker.latencies)
                df_lat2["timestamp"] = pd.to_datetime(df_lat2["timestamp"])
                st.dataframe(df_lat2, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune latence enregistrée")
        with t3:
            if tracker.predictions:
                df_pred = pd.DataFrame([{
                    "Timestamp": p["timestamp"], "Gravité": p["severity"],
                    "Âge": p["patient"]["age"], "Sexe": p["patient"]["sex"],
                    "Symptômes": ", ".join(p["symptoms"]),
                    "Drapeaux": len(p["red_flags"]),
                    "Confiance": f"{p['confidence']*100:.0f}%",
                } for p in tracker.predictions])
                st.dataframe(df_pred, use_container_width=True, hide_index=True)
            else:
                st.info("Aucune prédiction enregistrée")

    st.divider()
    st.caption("Note: Les coûts sont estimés selon les tarifs Mistral officiels.")


# ===========================================================================
# NAVIGATION
# ===========================================================================

pages = {
    "🏠 Accueil": page_accueil,
    "🤖 Simulation automatique": page_simulation,
    "💬 Chat interactif": page_chat_interactif,
    "🎲 Génération de données": page_generation,
    "📊 Monitoring": page_monitoring,
}

with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
    <div style="font-size:2.5rem;">🏥</div>
    <div style="font-size:1rem; font-weight:700; color:#e0f2fe; letter-spacing:0.5px;">Triage IA</div>
    <div style="font-size:0.75rem; color:#93c5fd;">Aide à la décision médicale</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    st.markdown("---")

pages[selected]()
