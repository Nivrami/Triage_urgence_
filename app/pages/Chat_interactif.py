"""
Page Streamlit  - Chatbot + ML Predictor + Monitoring
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.chatbot import TriageChatbotAPI
from src.rag.predictor import MLTriagePredictor

# Config
st.set_page_config(page_title="Chatbot Triage ML", page_icon="🏥", layout="wide")

st.title("🏥 Chatbot de Triage des Urgences")
st.markdown("*Assistant ML pour aide à la décision*")

# Session
if "chatbot" not in st.session_state:
    # Charger RAG
    retriever = None
    with st.spinner("🔄 Chargement RAG..."):
        try:
            project_root = Path(__file__).resolve().parents[2]
            vector_db_path = project_root / "data" / "vector_db"

            from src.rag.vector_store import VectorStore, RAGRetriever

            vector_store = VectorStore(
                persist_directory=str(vector_db_path), collection_name="triage_medical"
            )
            retriever = RAGRetriever(vector_store=vector_store)

            st.session_state.predictor = MLTriagePredictor(rag_retriever=retriever)
            st.success("✅ RAG chargé avec succès")
        except Exception as e:
            st.warning(f"⚠️ RAG non chargé: {e}")
            st.session_state.predictor = MLTriagePredictor()

    # Créer chatbot AVEC le retriever RAG
    st.session_state.chatbot = TriageChatbotAPI(retriever=retriever)
    st.session_state.messages = []
    st.session_state.started = False
    st.session_state.prediction = None

bot = st.session_state.chatbot
predictor = st.session_state.predictor
data = bot.data

# Sidebar
with st.sidebar:
    st.header("🧑‍⚕️ Dossier patient")

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
            st.write(f"🌡️ Temp: {v['Temperature']}°C")
        if "FC" in v:
            st.write(f"❤️ FC: {v['FC']} bpm")
        if "TA_systolique" in v:
            st.write(f"💉 TA: {v['TA_systolique']}/{v.get('TA_diastolique', '?')}")
        if "SpO2" in v:
            st.write(f"🫁 SpO2: {v['SpO2']}%")
        if "FR" in v:
            st.write(f"🌬️ FR: {v['FR']}/min")
    else:
        st.write("—")
    st.divider()

    # Actions
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        bot.reset()
        st.session_state.messages = []
        st.session_state.started = False
        st.session_state.prediction = None
        st.rerun()

    ready = bot.is_ready_for_prediction()

    if st.button(
        "🎯 Obtenir prédiction ML",
        use_container_width=True,
        disabled=not ready,
        help="5/5 constantes requises",
    ):
        with st.spinner("🔮 Analyse ML en cours..."):
            summary = bot.get_summary()
            st.session_state.prediction = predictor.predict(summary)
        st.rerun()

# Conversation
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Conversation")

    if not st.session_state.started:
        if st.button("🚀 Démarrer", use_container_width=True):
            msg = bot.start()
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.session_state.started = True
            st.rerun()

    # Messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    if st.session_state.started and not st.session_state.prediction:
        user_input = st.chat_input("Votre message...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = bot.chat(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Prédiction
with col2:
    st.subheader("🎯 Résultat ML")

    if st.session_state.prediction:
        r = st.session_state.prediction

        # Badge niveau
        st.markdown(
            f"""
        <div style="
            background-color:{r['color']};
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:22px;
            font-weight:bold;
            color:{'white' if r['severity_level'] in ['ROUGE','GRIS'] else 'black'};
            margin-bottom:20px;
        ">{r['label']}</div>
        """,
            unsafe_allow_html=True,
        )

        st.info(f"**Action:** {r['action']}")

        # Drapeaux rouges
        if r.get("red_flags"):
            st.warning("⚠️ **Signes de gravité:**")
            for f in r["red_flags"]:
                st.markdown(f"- {f}")

        # Probabilités
        st.subheader("📊 Probabilités ML")
        for lvl in ["ROUGE", "JAUNE", "VERT", "GRIS"]:
            p = r["probabilities"].get(lvl, 0)
            st.progress(p, text=f"{lvl}: {p*100:.1f}%")

        st.metric("Confiance", f"{r['confidence']*100:.1f}%")

        # Features
        if r.get("features_used"):
            with st.expander("🔍 Features ML"):
                st.json(r["features_used"])

        # RAG Sources
        if r.get("rag_sources"):
            with st.expander("📚 Sources RAG"):
                for source in r["rag_sources"]:
                    st.markdown(f"- {source}")

        # Justification
        with st.expander("📋 Justification ML + RAG", expanded=True):
            st.markdown(r["justification"])

        # Export
        st.divider()
        if st.button("💾 Exporter rapport", use_container_width=True):
            export = f"# RAPPORT TRIAGE ML\n\n"
            export += f"## {r['label']}\n"
            export += f"**Action:** {r['action']}\n"
            export += f"**Confiance:** {r['confidence']*100:.1f}%\n\n"

            if r.get("red_flags"):
                export += "## Drapeaux rouges\n"
                for f in r["red_flags"]:
                    export += f"- {f}\n"

            export += "\n## Conversation\n\n"
            for m in st.session_state.messages:
                role = "Assistant" if m["role"] == "assistant" else "Patient"
                export += f"**{role}:** {m['content']}\n\n"

            st.download_button(
                "📥 Télécharger",
                export,
                "rapport_triage_ml.md",
                "text/markdown",
                use_container_width=True,
            )
    else:
        st.info(
            "💡 Complétez la conversation (5/5 constantes) puis cliquez 'Obtenir prédiction ML'"
        )

        with st.expander("❓ Aide"):
            st.markdown(
                """
            **Constantes requises (5/5):**
            - ✅ Température
            - ✅ Fréquence cardiaque (FC)
            - ✅ Tension artérielle (TA)
            - ✅ Saturation oxygène (SpO2)
            - ✅ Fréquence respiratoire (FR)

            Le bot vous guidera étape par étape.
            """
            )

st.divider()
st.caption(
    "🤖 ML (Random Forest) + 📚 RAG (Documents médicaux) + 🔮 Mistral API | ⚠️ Outil d'aide - ne remplace pas avis médical"
)
