import streamlit as st

if "step" not in st.session_state:
    st.session_state.step = 1

st.title("ML Pipeline App")

st.write("""
Application complète d'apprentissage automatique pour l'analyse de données qualitatives 
et quantitatives avec multiples algorithmes de Machine Learning.
""")

st.subheader("🎯 Fonctionnalités")

col1, col2 = st.columns(2)

with col1:
    st.write("**📥 Gestion des Données**")
    st.write("• Import/Export de datasets")
    st.write("• Nettoyage et préparation")
    st.write("• Gestion des valeurs manquantes")

with col2:
    st.write("**📊 Analyse & Résultats**")
    st.write("• Visualisation graphique")
    st.write("• Entraînement de modèles")
    st.write("• Évaluation des performances")

st.info("👈 Utilisez la barre latérale pour naviguer entre les étapes du pipeline")