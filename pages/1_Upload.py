import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(page_title="Chargement des données", page_icon="📁", layout="wide")
st.title("📁 Étape 1 : Chargement des données")

# -----------------------------
# Initialisation du session state
# -----------------------------
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}

# -----------------------------
# Choix de la source de données
# -----------------------------
st.header("Choisir la source des données")

data_source = st.radio(
    "Comment souhaitez-vous fournir le jeu de données ?",
    ["Importer un fichier", "Créer un jeu de données manuellement"],
    horizontal=True
)

# =================================================
# OPTION 1 — IMPORTATION DE FICHIER
# =================================================
if data_source == "Importer un fichier":
    uploaded_file = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"],
        help="Taille maximale du fichier : 200 Mo"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.data = df
            st.success(f"✅ Fichier importé avec succès ! Dimensions : {df.shape}")

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier : {str(e)}")

# =================================================
# OPTION 2 — CRÉATION MANUELLE DU DATASET
# =================================================
elif data_source == "Créer un jeu de données manuellement":
    st.subheader("Créer votre propre jeu de données")

    rows = st.number_input("Nombre de lignes", min_value=1, max_value=1000, value=5)
    cols = st.number_input("Nombre de colonnes", min_value=1, max_value=20, value=3)

    noms_colonnes = []
    types_colonnes = []

    st.markdown("### Définition des colonnes")

    for i in range(cols):
        col1, col2 = st.columns([2, 1])
        with col1:
            nom = st.text_input(f"Nom de la colonne {i+1}", value=f"variable_{i+1}")
        with col2:
            dtype = st.selectbox(
                "Type",
                ["Numérique", "Catégorielle"],
                key=f"type_{i}"
            )
        noms_colonnes.append(nom)
        types_colonnes.append(dtype)

    if st.button("➕ Générer le jeu de données vide"):
        data = {}
        for nom, dtype in zip(noms_colonnes, types_colonnes):
            if dtype == "Numérique":
                data[nom] = [0.0] * rows
            else:
                data[nom] = [""] * rows

        df = pd.DataFrame(data)
        st.session_state.data = df
        st.success("✅ Jeu de données créé avec succès !")

# =================================================
# AFFICHAGE DU DATASET
# =================================================
if st.session_state.data is not None:
    df = st.session_state.data

    st.header("📋 Aperçu du jeu de données")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Aperçu", "Structure", "Statistiques", "Valeurs manquantes"]
    )

    # -----------------------------
    # Onglet 1 : Aperçu
    # -----------------------------
    with tab1:
        st.dataframe(df, use_container_width=True)
        st.caption(f"Affichage de {min(20, len(df))} lignes sur {len(df)}")

    # -----------------------------
    # Onglet 2 : Structure
    # -----------------------------
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Types des variables")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index()
            dtype_df.columns = ["Variable", "Type"]
            st.dataframe(dtype_df, use_container_width=True)

        with col2:
            st.subheader("Informations générales")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

    # -----------------------------
    # Onglet 3 : Statistiques
    # -----------------------------
    with tab3:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all").round(2), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de lignes", len(df))
        with col2:
            st.metric("Nombre de colonnes", df.shape[1])
        with col3:
            st.metric(
                "Variables numériques",
                len(df.select_dtypes(include=np.number).columns)
            )

    # -----------------------------
    # Onglet 4 : Valeurs manquantes
    # -----------------------------
    with tab4:
        st.subheader("Analyse des valeurs manquantes")

        missing_df = pd.DataFrame({
            "Variable": df.columns,
            "Valeurs manquantes": df.isnull().sum(),
            "Pourcentage (%)": (df.isnull().sum() / len(df) * 100).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        if missing_df["Valeurs manquantes"].sum() > 0:
            fig = px.bar(
                missing_df,
                x="Variable",
                y="Pourcentage (%)",
                title="Pourcentage de valeurs manquantes par variable",
                color="Pourcentage (%)",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Aucune valeur manquante détectée")

    # -----------------------------
    # Sauvegarde des infos
    # -----------------------------
    st.session_state.data_info = {
        "shape": df.shape,
        "colonnes": df.columns.tolist(),
        "variables_numeriques": df.select_dtypes(include=np.number).columns.tolist(),
        "variables_categorielles": df.select_dtypes(exclude=np.number).columns.tolist()
    }

    # -----------------------------
    # Navigation
    # -----------------------------
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Supprimer le jeu de données", use_container_width=True):
            st.session_state.data = None
            st.session_state.data_info = {}
            st.rerun()

    with col2:
        if st.button(
            "➡️ Continuer vers le prétraitement",
            type="primary",
            use_container_width=True
        ):
            st.switch_page("pages/2_Preprocessing.py")

else:
    st.info("Importez un fichier ou créez un jeu de données pour commencer.")
