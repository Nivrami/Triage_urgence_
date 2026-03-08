"""
Page Monitoring - Suivi coûts et performances
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ajouter le chemin src au PYTHONPATH
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from src.monitoring.metrics_tracker import get_tracker
from src.monitoring.cost_calculator import get_calculator

# Configuration
st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

# Titre
st.title("📊 Monitoring du Système")
st.markdown("Suivi des coûts, performances et utilisation")

# Initialisation
tracker = get_tracker()
calculator = get_calculator()

# Sidebar
with st.sidebar:
    st.header("Actions")

    if st.button("🔄 Rafraîchir", use_container_width=True):
        st.rerun()

    if st.button("📥 Export CSV", use_container_width=True):
        export_path = tracker.export_csv()
        st.success(f"Exporté vers: {export_path}")

    if st.button("🗑️ Reset métriques", use_container_width=True, type="secondary"):
        if st.session_state.get("confirm_reset"):
            tracker.reset()
            st.success("Métriques réinitialisées")
            st.session_state.confirm_reset = False
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.warning("Cliquez à nouveau pour confirmer")

# Stats API
api_stats = tracker.get_api_stats()
cost_data = calculator.calculate_total_cost(tracker.api_calls)


if tracker.api_calls:
    first_call = datetime.fromisoformat(tracker.api_calls[0]["timestamp"])
    days_elapsed = max(1, (datetime.now() - first_call).days)
else:
    days_elapsed = 1

monthly_estimate = calculator.estimate_monthly_cost(cost_data["total_cost"], days_elapsed)

# Métriques principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Coût Total",
        calculator.format_cost(cost_data["total_cost"]),
    )

with col2:
    st.metric(
        "🔢 Appels API",
        f"{api_stats['total_calls']}",
    )

with col3:
    st.metric(
        "⏱️ Latence Moyenne",
        f"{api_stats['avg_latency']:.2f}s" if api_stats["avg_latency"] > 0 else "N/A",
    )

with col4:
    pred_stats = tracker.get_prediction_stats()
    st.metric(
        "🎯 Prédictions",
        f"{pred_stats['total']}",
        delta=(
            f"{pred_stats['avg_confidence']*100:.0f}% confiance"
            if pred_stats["total"] > 0
            else None
        ),
    )

st.divider()

# Section Coûts
st.header("💰 Analyse des Coûts")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Répartition par Service")

    if cost_data["total_cost"] > 0:
        fig_cost = go.Figure(
            data=[
                go.Pie(
                    labels=["Mistral API", "Embeddings"],
                    values=[cost_data["mistral"]["cost"], cost_data["embeddings"]["cost"]],
                    hole=0.4,
                )
            ]
        )
        fig_cost.update_layout(height=300)
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("Aucune donnée de coût disponible")

    # Détails Mistral
    st.markdown("**Mistral API**")
    st.write(f"• Appels: {cost_data['mistral']['calls']}")
    st.write(f"• Tokens input: {cost_data['mistral']['tokens_input']:,}")
    st.write(f"• Tokens output: {cost_data['mistral']['tokens_output']:,}")
    st.write(f"• Coût: {calculator.format_cost(cost_data['mistral']['cost'])}")

with col2:
    st.subheader("Évolution des Coûts")

    if tracker.api_calls:
        # Créer DataFrame des coûts cumulés
        costs_over_time = []
        cumulative_cost = 0

        for call in tracker.api_calls:
            if call.get("service") == "mistral":
                cost = calculator.calculate_mistral_cost(
                    call.get("model", "mistral-small-latest"),
                    call["tokens_input"],
                    call["tokens_output"],
                )
                cumulative_cost += cost["cost_total"]

            costs_over_time.append(
                {"timestamp": datetime.fromisoformat(call["timestamp"]), "cost": cumulative_cost}
            )

        if costs_over_time:
            df_cost = pd.DataFrame(costs_over_time)
            fig_timeline = px.line(df_cost, x="timestamp", y="cost", title="Coût Cumulé")
            fig_timeline.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Aucune donnée disponible")

st.divider()

# Section Performances
st.header("⚡ Performances")

latency_stats = tracker.get_latency_stats()

if latency_stats:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Latences par Composant")

        # Tableau
        latency_df = pd.DataFrame(
            [
                {
                    "Composant": comp,
                    "Moyenne (s)": f"{stats['avg']:.3f}",
                    "Min (s)": f"{stats['min']:.3f}",
                    "Max (s)": f"{stats['max']:.3f}",
                    "Appels": stats["count"],
                }
                for comp, stats in latency_stats.items()
            ]
        )
        st.dataframe(latency_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Distribution des Latences")

        # Graphique latences
        latency_data = []
        for lat in tracker.latencies:
            latency_data.append({"Composant": lat["component"], "Durée (s)": lat["duration"]})

        if latency_data:
            df_lat = pd.DataFrame(latency_data)
            fig_lat = px.box(df_lat, x="Composant", y="Durée (s)", title="Distribution")
            fig_lat.update_layout(height=300)
            st.plotly_chart(fig_lat, use_container_width=True)
else:
    st.info("Aucune donnée de latence disponible")

st.divider()

# Section Prédictions
st.header("🎯 Analyse des Prédictions")

if pred_stats["total"] > 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Répartition par Gravité")

        severity_df = pd.DataFrame(
            [{"Niveau": k, "Nombre": v} for k, v in pred_stats["by_severity"].items()]
        )

        fig_sev = px.pie(
            severity_df,
            values="Nombre",
            names="Niveau",
            color="Niveau",
            color_discrete_map={
                "ROUGE": "#FF0000",
                "JAUNE": "#FFD700",
                "VERT": "#00FF00",
                "GRIS": "#808080",
            },
        )
        fig_sev.update_layout(height=300)
        st.plotly_chart(fig_sev, use_container_width=True)

    with col2:
        st.subheader("Statistiques")

        for severity, count in pred_stats["by_severity"].items():
            pct = (count / pred_stats["total"]) * 100
            st.write(f"**{severity}**: {count} ({pct:.1f}%)")

        st.write(f"**Confiance moyenne**: {pred_stats['avg_confidence']*100:.1f}%")

    with col3:
        st.subheader("Dernières Prédictions")

        recent = tracker.predictions[-5:][::-1]
        for pred in recent:
            ts = datetime.fromisoformat(pred["timestamp"]).strftime("%H:%M:%S")
            st.write(f"**{ts}** - {pred['severity']} ({pred['confidence']*100:.0f}%)")
else:
    st.info("Aucune prédiction disponible")

st.divider()

# Section Détails
with st.expander("📋 Détails Techniques"):
    tab1, tab2, tab3 = st.tabs(["Appels API", "Latences", "Prédictions"])

    with tab1:
        if tracker.api_calls:
            df_api = pd.DataFrame(tracker.api_calls)
            df_api["timestamp"] = pd.to_datetime(df_api["timestamp"])
            st.dataframe(df_api, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun appel API enregistré")

    with tab2:
        if tracker.latencies:
            df_lat_full = pd.DataFrame(tracker.latencies)
            df_lat_full["timestamp"] = pd.to_datetime(df_lat_full["timestamp"])
            st.dataframe(df_lat_full, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune latence enregistrée")

    with tab3:
        if tracker.predictions:
            df_pred = pd.DataFrame(
                [
                    {
                        "Timestamp": p["timestamp"],
                        "Gravité": p["severity"],
                        "Âge": p["patient"]["age"],
                        "Sexe": p["patient"]["sex"],
                        "Symptômes": ", ".join(p["symptoms"]),
                        "Drapeaux": len(p["red_flags"]),
                        "Confiance": f"{p['confidence']*100:.0f}%",
                    }
                    for p in tracker.predictions
                ]
            )
            st.dataframe(df_pred, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune prédiction enregistrée")

# Footer
st.divider()
st.caption(
    "💡 Note: Les coûts sont estimés selon les tarifs Mistral officiels, même avec clé gratuite."
)
