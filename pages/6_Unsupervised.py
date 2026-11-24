import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from io import BytesIO
import base64
import joblib
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="Modèles Non Supervisés",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Apprentissage Non Supervisé")

# -----------------------------
# Vérification du dataset
# -----------------------------
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    data_source = "Données Prétraitées"
elif 'data' in st.session_state and st.session_state.data is not None:
    df = st.session_state.data
    data_source = "Données Originales"
else:
    st.warning("⚠️ Aucune donnée trouvée ! Veuillez importer et prétraiter les données d'abord.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Retour à l’importation des données"):
            st.switch_page("pages/1_Upload.py")
    with col2:
        if st.button("🔧 Aller au prétraitement"):
            st.switch_page("pages/2_Preprocessing.py")
    st.stop()

st.info(f"📊 Source utilisée : **{data_source}** | Dimensions : {df.shape}")

# Conserver uniquement les colonnes numériques
X = df.select_dtypes(include="number")

if X.empty:
    st.error("Le dataset ne contient aucune variable numérique.")
    st.stop()

# -----------------------------
# Paramètres du modèle K-Means
# -----------------------------
st.sidebar.header("Paramètres K-Means")
n_clusters = st.sidebar.slider("Nombre de clusters", 2, 10, 3)

# -----------------------------
# Entraînement du modèle
# -----------------------------
if st.button("🚀 Entraîner le modèle K-Means"):
    with st.spinner("Entraînement de K-Means en cours..."):
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X)

        st.session_state["unsup_model"] = model
        st.session_state["unsup_labels"] = labels
        st.session_state["unsup_model_name"] = "K-Means"

        st.success("✅ K-Means entraîné avec succès !")

# -----------------------------
# Évaluation
# -----------------------------
if "unsup_model" in st.session_state:
    st.subheader("📊 Métriques d’évaluation")

    labels = st.session_state["unsup_labels"]
    metrics = {}

    if labels is not None and len(set(labels)) > 1:
        metrics["Silhouette"] = silhouette_score(X, labels)
        metrics["Calinski-Harabasz"] = calinski_harabasz_score(X, labels)
        metrics["Davies-Bouldin"] = davies_bouldin_score(X, labels)

    for k, v in metrics.items():
        st.metric(k, f"{v:.4f}")
        st.session_state["unsup_metrics"] = metrics

# -----------------------------
# Navigation vers le tableau de résultats
# -----------------------------
if "unsup_model" in st.session_state:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📊 Voir le tableau de résultats", use_container_width=True, type="primary"):
            st.switch_page("pages/7_Results.py")

