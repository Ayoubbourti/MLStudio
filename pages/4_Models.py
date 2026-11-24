import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sélection du Modèle",
    page_icon="🤖",
    layout="wide"
)

# ------------------------
# Custom CSS for dark mode cards
# ------------------------
st.markdown("""
<style>
    /* Card styling */
    .simple-card {
        background: #000000;  /* black card */
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        border: 2px solid #1F1F1F;
        transition: all 0.3s ease;
        cursor: pointer;
        color: #F9FAFB;
    }
    .simple-card:hover {
        border-color: #3B82F6;
        transform: translateY(-2px);
    }
    .simple-card.selected {
        border-color: #3B82F6;
        background-color: #1F1F1F;
        color: #F3F4F6;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-classification { background-color: #059669; color: #F3F4F6; }
    .badge-regression { background-color: #2563EB; color: #F3F4F6; }
    .badge-clustering { background-color: #7C3AED; color: #F3F4F6; }

    .simple-card h4 { margin-bottom: 0.5rem; font-size: 1.05rem; }
    .simple-card p { font-size: 0.8rem; line-height: 1.3; }

    button[kind="primary"] {
        font-size: 0.8rem !important;
        padding: 0.3rem 0.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Étape 4 : Sélection du Modèle")

# ------------------------
# Check if data is available
# ------------------------
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    data_source = "Données Prétraitées"
elif 'data' in st.session_state and st.session_state.data is not None:
    df = st.session_state.data
    data_source = "Données Originales"
else:
    st.warning("⚠️ Aucune donnée trouvée ! Veuillez importer et prétraiter les données.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Aller à l’importation des données"):
            st.switch_page("pages/1_Upload.py")
    with col2:
        if st.button("🔧 Aller au prétraitement"):
            st.switch_page("pages/2_Preprocessing.py")
    st.stop()

st.info(f"📁 Source utilisée : **{data_source}** | Dimensions : {df.shape}")

# ------------------------
# Check target column
# ------------------------
if not hasattr(st.session_state, 'target_column') or st.session_state.target_column is None:
    st.error("❌ Variable cible non sélectionnée ! Veuillez revenir au prétraitement.")
    if st.button("🔧 Aller au prétraitement"):
        st.switch_page("pages/2_Preprocessing.py")
    st.stop()

target_col = st.session_state.target_column

# ------------------------
# Problem type detection
# ------------------------
st.header("🔍 Analyse du type de problème")
target_series = df[target_col]
unique_values = target_series.nunique()
is_categorical = target_series.dtype == 'object' or unique_values < 10
problem_type = "Classification" if is_categorical else "Régression"
st.session_state.problem_type = problem_type
badge_class = "badge-classification" if problem_type == "Classification" else "badge-regression"
icon = "🏷️" if problem_type == "Classification" else "📈"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div style='text-align:center; font-size:2rem'>{icon}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<span class='badge {badge_class}'>{problem_type}</span>", unsafe_allow_html=True)
with col3:
    if problem_type == "Classification":
        st.metric("Nombre de classes", unique_values)
    else:
        st.metric("Plage de la cible", f"{target_series.min():.2f} - {target_series.max():.2f}")
with col4:
    if problem_type == "Classification":
        class_dist = target_series.value_counts(normalize=True)
        imbalance_ratio = class_dist.min() / class_dist.max()
        if imbalance_ratio < 0.3:
            balance_status = "⚠️ Déséquilibré"
        elif imbalance_ratio < 0.6:
            balance_status = "⚠️ Modéré"
        else:
            balance_status = "✓ Équilibré"
        st.metric("Équilibre des classes", balance_status)
    else:
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((target_series < (Q1 - 1.5 * IQR)) | (target_series > (Q3 + 1.5 * IQR))).sum()
        st.metric("Valeurs aberrantes", outliers)

# ------------------------
# Target visualization
# ------------------------
st.subheader("📊 Distribution de la variable cible")
if problem_type == "Classification":
    fig = px.pie(
        values=target_series.value_counts().values,
        names=target_series.value_counts().index,
        title=f"Distribution de {target_col}",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x=target_col, nbins=30, title=f"Distribution de {target_col}")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=target_series, name=target_col))
        fig2.update_layout(title=f"Boîte à moustaches de {target_col}")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# Model selection
# ------------------------
st.header("🎯 Sélection du type d’apprentissage")
model_type = st.radio(
    "Choisissez le type de tâche en Machine Learning :",
    ["Supervised Learning", "Unsupervised Learning"],
    horizontal=True
)
st.session_state.selected_model_type = "Supervised" if model_type == "Supervised Learning" else "Unsupervised"

# ------------------------
# Supervised Learning Cards
# ------------------------
if model_type == "Supervised Learning":
    st.subheader("🤖 Algorithmes disponibles")

    st.markdown("### 📌 Algorithmes de base")
    algorithms = []

    if problem_type == "Classification":
        algorithms = [
            {"name":"Logistic Regression","desc":"Modèle linéaire pour la classification binaire ou multiclasse","badge":"badge-classification"},
            {"name":"Naive Bayes","desc":"Classificateur probabiliste rapide basé sur le théorème de Bayes","badge":"badge-classification"},
            {"name":"K-Nearest Neighbors (KNN)","desc":"Apprentissage par instance, vote majoritaire","badge":"badge-classification"},
        ]
    else:
        algorithms = [
            {"name":"Linear Regression","desc":"Relation linéaire simple entre les variables et la cible","badge":"badge-regression"},
            {"name":"K-Nearest Neighbors (KNN)","desc":"Prédit à partir de la moyenne des voisins proches","badge":"badge-regression"},
            {"name":"Support Vector Regression (SVR)","desc":"Régression non linéaire utilisant des noyaux","badge":"badge-regression"},
        ]

    cols = st.columns(3)
    for i, alg in enumerate(algorithms):
        with cols[i % 3]:
            selected = st.session_state.get("selected_algorithm") == alg["name"]
            card_class = "simple-card selected" if selected else "simple-card"
            if st.button("", key=f"alg_{alg['name']}", use_container_width=True):
                st.session_state.selected_algorithm = alg["name"]
                st.rerun()
            st.markdown(f"""
            <div class="{card_class}" style="text-align:center;">
                <h4>{alg['name']}</h4>
                <span class="badge {alg['badge']}">{problem_type}</span>
                <p>{alg['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🌲 Algorithmes basés sur les arbres")
    tree_algorithms = []

    if problem_type == "Classification":
        tree_algorithms = [
            {"name":"Decision Tree","desc":"Arbre de décision avec règles if-then, très interprétable","badge":"badge-classification"},
            {"name":"Random Forest","desc":"Ensemble d’arbres de décision, réduit le sur-apprentissage","badge":"badge-classification"},
        ]
    else:
        tree_algorithms = [
            {"name":"Decision Tree Regressor","desc":"Modèle de régression non linéaire basé sur les arbres","badge":"badge-regression"},
            {"name":"Random Forest Regressor","desc":"Méthode d’ensemble moyennant plusieurs arbres","badge":"badge-regression"},
        ]

    cols = st.columns(3)
    for i, alg in enumerate(tree_algorithms):
        with cols[i % 3]:
            selected = st.session_state.get("selected_algorithm") == alg["name"]
            card_class = "simple-card selected" if selected else "simple-card"
            if st.button("", key=f"alg_{alg['name']}", use_container_width=True):
                st.session_state.selected_algorithm = alg["name"]
                st.rerun()
            st.markdown(f"""
            <div class="{card_class}" style="text-align:center;">
                <h4>{alg['name']}</h4>
                <span class="badge {alg['badge']}">{problem_type}</span>
                <p>{alg['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🚀 Algorithmes avancés & boosting")
    advanced_algorithms = []

    if problem_type == "Classification":
        advanced_algorithms = [
            {"name":"Support Vector Machine (SVM)","desc":"Recherche l’hyperplan optimal avec des noyaux","badge":"badge-classification"},
            {"name":"AdaBoost","desc":"Boosting adaptatif mettant l’accent sur les erreurs","badge":"badge-classification"},
            {"name":"Gradient Boosting (GBM)","desc":"Construction séquentielle sur les résidus","badge":"badge-classification"},
            {"name":"XGBoost","desc":"Gradient boosting optimisé, très performant","badge":"badge-classification"},
            {"name":"LightGBM","desc":"Boosting rapide et efficace pour grands jeux de données","badge":"badge-classification"},
        ]
    else:
        advanced_algorithms = [
            {"name":"AdaBoost Regressor","desc":"Boosting adaptatif pour la régression","badge":"badge-regression"},
            {"name":"Gradient Boosting Regressor (GBM)","desc":"Boosting séquentiel minimisant la fonction de perte","badge":"badge-regression"},
            {"name":"XGBoost Regressor","desc":"Gradient boosting extrême pour la régression","badge":"badge-regression"},
            {"name":"LightGBM Regressor","desc":"Gradient boosting léger pour un entraînement rapide","badge":"badge-regression"},
        ]

    cols = st.columns(3)
    for i, alg in enumerate(advanced_algorithms):
        with cols[i % 3]:
            selected = st.session_state.get("selected_algorithm") == alg["name"]
            card_class = "simple-card selected" if selected else "simple-card"
            if st.button("", key=f"alg_{alg['name']}", use_container_width=True):
                st.session_state.selected_algorithm = alg["name"]
                st.rerun()
            st.markdown(f"""
            <div class="{card_class}" style="text-align:center;">
                <h4>{alg['name']}</h4>
                <span class="badge {alg['badge']}">{problem_type}</span>
                <p>{alg['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------
# Unsupervised Learning Cards
# ------------------------
if model_type == "Unsupervised Learning":
    st.subheader("🤖 Algorithme disponible pour le clustering")
    cols = st.columns(3)
    with cols[1]:
        selected = st.session_state.get("selected_algorithm") == "K-Means Clustering"
        card_class = "simple-card selected" if selected else "simple-card"
        if st.button("", key="kmeans_card", use_container_width=True):
            st.session_state.selected_algorithm = "K-Means Clustering"
            st.rerun()
        st.markdown(f"""
        <div class="{card_class}" style="text-align:center;">
            <h4>K-Means Clustering</h4>
            <span class="badge badge-clustering">Clustering</span>
            <p>Partitionne les données en clusters selon la similarité.<br>
            - Rapide et très utilisé<br>
            - Fonctionne bien avec des clusters sphériques<br>
            - Nécessite de définir le nombre de clusters</p>
        </div>
        """, unsafe_allow_html=True)

# Navigation to training page
if st.session_state.get("selected_algorithm") is not None:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.session_state.selected_model_type == "Supervised":
            if st.button("➡️ Passer à l’entraînement", use_container_width=True, type="primary"):
                st.switch_page("pages/5_supervised.py")
        else:
            if st.button("➡️ Passer à l’entraînement", use_container_width=True, type="primary"):
                st.switch_page("pages/6_Unsupervised.py")
else:
    st.info("ℹ️ Veuillez sélectionner un modèle pour continuer.")
