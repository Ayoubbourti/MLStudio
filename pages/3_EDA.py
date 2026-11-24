import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Étape 3 : Analyse Exploratoire des Données (EDA)")

# -----------------------------
# Chargement des données
# -----------------------------
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    data_source = "Données prétraitées"
elif 'data' in st.session_state and st.session_state.data is not None:
    df = st.session_state.data
    data_source = "Données originales"
else:
    st.warning("⚠️ Aucune donnée trouvée ! Veuillez d’abord importer un jeu de données.")
    st.stop()

st.info(f"📁 Source utilisée : **{data_source}** | Dimensions : {df.shape}")

# -----------------------------
# Barre latérale
# -----------------------------
st.sidebar.header("📊 Options EDA")
viz_type = st.sidebar.selectbox(
    "Sélectionner le type de visualisation",
    [
        "📈 Vue d’ensemble",
        "📉 Distributions",
        "🔗 Corrélations",
        "🎯 Cible vs Variables"
    ]
)

# -----------------------------
# VUE D’ENSEMBLE
# -----------------------------
if viz_type == "📈 Vue d’ensemble":
    st.header("Vue d’ensemble du jeu de données")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Nombre d’échantillons", len(df))
    with col2: st.metric("Nombre de variables", df.shape[1])
    with col3: st.metric("Variables numériques", len(df.select_dtypes(include=np.number).columns))
    with col4: st.metric("Variables catégorielles", len(df.select_dtypes(include=['object']).columns))

    st.subheader("Répartition des types de données")
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Types de variables"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Valeurs manquantes (NaN)")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        missing_percent = (missing / len(df)) * 100
        fig = px.bar(
            x=missing_percent.index,
            y=missing_percent.values,
            labels={"x": "Variable", "y": "% Manquant"},
            title="Pourcentage de valeurs manquantes par variable"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ Aucune valeur manquante détectée")

# -----------------------------
# DISTRIBUTIONS
# -----------------------------
elif viz_type == "📉 Distributions":
    st.header("Distribution des variables")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    columns_to_plot = st.multiselect(
        "Sélectionner les variables",
        df.columns.tolist(),
        default=numeric_cols[:4]
    )

    if columns_to_plot:
        n_cols = 2
        n_rows = (len(columns_to_plot) + 1) // 2
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=columns_to_plot)

        for i, col in enumerate(columns_to_plot):
            r = i // n_cols + 1
            c = i % n_cols + 1
            if pd.api.types.is_numeric_dtype(df[col]):
                fig.add_trace(go.Histogram(x=df[col], nbinsx=30), row=r, col=c)
            else:
                counts = df[col].value_counts().head(10)
                fig.add_trace(go.Bar(x=counts.index, y=counts.values), row=r, col=c)

        fig.update_layout(height=300 * n_rows, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CORRÉLATIONS
# -----------------------------
elif viz_type == "🔗 Corrélations":
    st.header("Analyse des corrélations")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Matrice de corrélation"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CIBLE vs VARIABLES
# -----------------------------
elif viz_type == "🎯 Cible vs Variables":
    st.header("Relation entre la variable cible et les variables explicatives")

    if "target_column" not in st.session_state:
        st.warning("⚠️ Aucune variable cible sélectionnée.")
        st.stop()

    target = st.session_state.target_column

    if target not in df.columns:
        st.error("❌ La variable cible est introuvable dans le jeu de données.")
        st.stop()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    if not numeric_cols:
        st.warning("⚠️ Aucune variable explicative numérique disponible.")
        st.stop()

    st.markdown(
        """
        Cette section permet d’analyser la relation entre la **variable cible**
        et les **variables explicatives** à l’aide de :
        - 📌 Graphiques de dispersion (scatter plots)
        - 📌 Distributions conditionnelles selon la classe cible  

        Ces visualisations aident à identifier les **variables les plus influentes**
        pour la prédiction.
        """
    )

    selected_feature = st.selectbox(
        "Sélectionner une variable explicative",
        numeric_cols
    )

    st.subheader("🔹 Graphique de dispersion")
    fig_scatter = px.scatter(
        df,
        x=selected_feature,
        y=target,
        color=df[target].astype(str),
        title=f"{selected_feature} vs {target}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("🔹 Distributions conditionnelles")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            df,
            x=selected_feature,
            color=df[target].astype(str),
            nbins=30,
            barmode="overlay",
            title=f"Distribution de {selected_feature} selon la cible"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_box = px.box(
            df,
            x=df[target].astype(str),
            y=selected_feature,
            title=f"{selected_feature} par classe de la cible"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# Navigation
# -----------------------------
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("➡️ Passer à la sélection du modèle", use_container_width=True, type="primary"):
        st.switch_page("pages/4_Models.py")
