import streamlit as st

# Configuration de la page (si non déjà défini dans app.py)
st.set_page_config(
    page_title="Plateforme de Machine Learning",
    layout="wide"
)

# En-tête
st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <h1>📊 Plateforme de Machine Learning</h1>
        <p style="font-size:18px;">
            Une application web interactive pour l'analyse de données, le prétraitement,
            et l'expérimentation en Machine Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- Objectif du projet ---
st.subheader("🎯 Objectif du projet")
st.markdown(
    """
    Cette plateforme permet aux utilisateurs d'analyser des **datasets qualitatifs et quantitatifs**
    en appliquant un large éventail d'**algorithmes de Machine Learning** 
    à travers un workflow clair et étape par étape.
    """
)

# --- Mise en page : Workflow + Algorithmes ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Workflow utilisateur")
    st.markdown(
        """
        1️⃣ Importer un dataset (**CSV / Excel**)  
        2️⃣ Prétraiter les données *(nettoyage, encodage, normalisation)*  
        3️⃣ Explorer les données via des **visualisations interactives (EDA)**  
        4️⃣ Sélectionner et configurer les **modèles de Machine Learning**  
        5️⃣ Évaluer, comparer et **exporter les résultats**
        """
    )

with col2:
    st.subheader("🤖 Algorithmes disponibles")
    st.markdown(
        """
        **Apprentissage supervisé**
        - Régression linéaire (simple, multiple, polynomiale)
        - Régression logistique
        - Arbres de décision
        - Forêt aléatoire
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Naive Bayes
        - Réseaux de neurones artificiels

        **Apprentissage non supervisé**
        - K-Means Clustering
        """
    )

st.divider()

# --- Démarrage ---
st.subheader("🚀 Pour commencer")
st.markdown(
    """
    Utilisez le **menu de navigation latéral** pour passer d'une étape à l'autre de l'application.
    La plateforme vous guide depuis l'importation du dataset jusqu'à l'évaluation du modèle
    de manière logique et progressive.
    """
)

st.info("👉 Commencez par importer un dataset depuis la page **Téléchargement du dataset**.")
