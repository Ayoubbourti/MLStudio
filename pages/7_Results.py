import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from io import BytesIO
import base64

# ---------------------------------
# Configuration de la page
# ---------------------------------
st.set_page_config(
    page_title="Tableau de Résultats",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Tableau de Résultats")

# ---------------------------------
# Vérification du dataset
# ---------------------------------
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data
    data_source = "Données Prétraitées"
elif 'data' in st.session_state and st.session_state.data is not None:
    df = st.session_state.data
    data_source = "Données Originales"
else:
    st.warning("⚠️ Aucune donnée trouvée ! Veuillez importer et prétraiter les données d'abord.")
    st.stop()

st.info(f"📊 Source utilisée : **{data_source}** | Dimensions : {df.shape}")

# =================================
# 1️⃣ RÉSUMÉ DU DATASET
# =================================
st.header("📋 Résumé du Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nombre de lignes", df.shape[0])

with col2:
    st.metric("Nombre de colonnes", df.shape[1])

with col3:
    st.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

with col4:
    st.metric("Colonnes numériques", len(df.select_dtypes(include=np.number).columns))

st.divider()

# =================================
# 2️⃣ RÉSULTATS DE L’APPRENTISSAGE SUPERVISÉ
# =================================
st.header("🤖 Résultats de l’Apprentissage Supervisé")

if "trained_model" not in st.session_state:
    st.info("ℹ️ Aucun modèle supervisé entraîné. Veuillez aller à la page d’Apprentissage Supervisé pour entraîner un modèle.")
else:
    model = st.session_state["trained_model"]
    problem_type = st.session_state.get("problem_type", "Inconnu")
    selected_algorithm = st.session_state.get("selected_algorithm", "Inconnu")
    
    # Informations sur le modèle
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Modèle entraîné avec succès")
    with col2:
        st.metric("Algorithme", selected_algorithm)
    with col3:
        st.metric("Type de problème", problem_type)
    
    # Métriques de performance
    if "metrics" in st.session_state:
        st.subheader("📈 Métriques de Performance")
        metrics = st.session_state["metrics"]
        
        cols = st.columns(len(metrics))
        for col, (metric_name, metric_value) in zip(cols, metrics.items()):
            with col:
                st.metric(metric_name.upper(), f"{metric_value:.4f}")
    
    # Visualisations
    st.subheader("📊 Visualisations du Modèle")
    
    if problem_type == "Classification":
        # Matrice de confusion
        if "y_test" in st.session_state and "y_pred" in st.session_state:
            y_test = st.session_state["y_test"]
            y_pred = st.session_state["y_pred"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Matrice de Confusion**")
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Prédit", y="Réel", color="Nombre")
                )
                fig.update_layout(title="Matrice de Confusion")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Distribution des Prédictions**")
                comparison_df = pd.DataFrame({
                    'Réel': y_test.value_counts().sort_index(),
                    'Prédit': pd.Series(y_pred).value_counts().sort_index()
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Réel', x=comparison_df.index, y=comparison_df['Réel']))
                fig.add_trace(go.Bar(name='Prédit', x=comparison_df.index, y=comparison_df['Prédit']))
                fig.update_layout(barmode='group', title="Réel vs Prédit")
                st.plotly_chart(fig, use_container_width=True)
            
            # Rapport de classification
            st.write("**Rapport de Classification**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']))
    
    else:  # Régression
        if "y_test" in st.session_state and "y_pred" in st.session_state:
            y_test = st.session_state["y_test"]
            y_pred = st.session_state["y_pred"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Valeurs Réelles vs Prédites**")
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Valeurs Réelles', 'y': 'Valeurs Prédites'},
                    title="Valeurs Réelles vs Prédites"
                )
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Prédiction Parfaite',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Distribution des Résidus**")
                residuals = y_test - y_pred
                fig = px.histogram(
                    residuals,
                    nbins=30,
                    labels={'value': 'Résidus', 'count': 'Fréquence'},
                    title="Distribution des Résidus"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Résidus vs Valeurs Prédites**")
            fig = px.scatter(
                x=y_pred,
                y=residuals,
                labels={'x': 'Valeurs Prédites', 'y': 'Résidus'},
                title="Résidus vs Valeurs Prédites"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    # Export du modèle
    st.subheader("💾 Export du Modèle Supervisé")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Télécharger le Modèle (.pkl)", use_container_width=True):
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="💾 Télécharger le modèle entraîné",
                data=buffer,
                file_name=f"{selected_algorithm.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )
    
    with col2:
        if st.button("📊 Télécharger les Métriques (.csv)", use_container_width=True):
            if "metrics" in st.session_state:
                metrics_df = pd.DataFrame([st.session_state["metrics"]])
                csv = metrics_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger CSV des Métriques",
                    data=csv,
                    file_name="model_metrics.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("🔄 Réinitialiser le Modèle Supervisé", use_container_width=True):
            keys_to_remove = ["trained_model", "metrics", "y_test", "y_pred", "model_params", "X", "y"]
            for key in keys_to_remove:
                st.session_state.pop(key, None)
            st.success("✅ Modèle supervisé réinitialisé avec succès !")
            st.rerun()

st.divider()

# =================================
# 3️⃣ RÉSULTATS DE L’APPRENTISSAGE NON SUPERVISÉ
# =================================
st.header("🔍 Résultats de l’Apprentissage Non Supervisé")

if "unsup_model" not in st.session_state:
    st.info("ℹ️ Aucun modèle non supervisé entraîné. Veuillez aller à la page d’Apprentissage Non Supervisé pour entraîner un modèle.")
else:
    model = st.session_state["unsup_model"]
    model_name = st.session_state.get("unsup_model_name", "K-Means")
    labels = st.session_state.get("unsup_labels")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ Modèle entraîné avec succès")
    with col2:
        st.metric("Algorithme", model_name)
    with col3:
        st.metric("Nombre de Clusters", len(np.unique(labels)) if labels is not None else 0)
    
    if "unsup_metrics" in st.session_state:
        st.subheader("📊 Métriques de Clustering")
        metrics = st.session_state["unsup_metrics"]
        
        cols = st.columns(len(metrics))
        for col, (metric_name, metric_value) in zip(cols, metrics.items()):
            with col:
                st.metric(metric_name, f"{metric_value:.4f}")
        
        with st.expander("ℹ️ Explication des Métriques"):
            st.write("""
            - **Silhouette Score** : Variations entre -1 et 1. Plus c'est élevé, mieux c'est. Mesure la similarité d’un point à son cluster.
            - **Calinski-Harabasz Score** : Plus c'est élevé, mieux c'est. Ratio de dispersion entre clusters et à l’intérieur des clusters.
            - **Davies-Bouldin Score** : Plus c'est faible, mieux c'est. Moyenne de similarité entre chaque cluster et le cluster le plus similaire.
            """)
    
    st.subheader("📈 Visualisations des Clusters")
    
    numeric_cols = df.select_dtypes(include=np.number)
    
    if labels is not None and numeric_cols.shape[1] >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Visualisation 2D des Clusters**")
            plot_df = numeric_cols.iloc[:, :2].copy()
            plot_df.columns = ["Feature 1", "Feature 2"]
            plot_df["Cluster"] = labels.astype(str)
            
            fig = px.scatter(
                plot_df,
                x="Feature 1",
                y="Feature 2",
                color="Cluster",
                title="Clusters K-Means (2 premières features)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            if hasattr(model, 'cluster_centers_'):
                centers = model.cluster_centers_[:, :2]
                fig.add_trace(go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(size=15, color='black', symbol='x', line=dict(width=2)),
                    name='Centroides'
                ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Distribution des tailles des clusters**")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Nombre d’échantillons'},
                title="Nombre d’échantillons par Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if numeric_cols.shape[1] > 2:
            st.write("**Statistiques des Clusters par Feature**")
            
            selected_features = st.multiselect(
                "Sélectionner les features à analyser",
                numeric_cols.columns.tolist(),
                default=numeric_cols.columns[:3].tolist()
            )
            
            if selected_features:
                stats_df = numeric_cols[selected_features].copy()
                stats_df['Cluster'] = labels
                cluster_stats = stats_df.groupby('Cluster').mean()
                
                fig = px.imshow(
                    cluster_stats.T,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    labels=dict(x="Cluster", y="Feature", color="Valeur Moyenne")
                )
                fig.update_layout(title="Valeur Moyenne des Features par Cluster")
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ La visualisation nécessite au moins 2 features numériques.")
    
    st.subheader("💾 Export du Modèle Non Supervisé")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Télécharger le Modèle (.pkl)", key="unsup_download", use_container_width=True):
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="💾 Télécharger le modèle de clustering",
                data=buffer,
                file_name=f"{model_name.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )
    
    with col2:
        if st.button("📊 Télécharger les Labels (.csv)", use_container_width=True):
            if labels is not None:
                labels_df = pd.DataFrame({'Cluster': labels})
                csv = labels_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Télécharger les labels CSV",
                    data=csv,
                    file_name="cluster_labels.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("🔄 Réinitialiser le Modèle Non Supervisé", key="unsup_reset", use_container_width=True):
            keys_to_remove = ["unsup_model", "unsup_labels", "unsup_model_name", "unsup_metrics"]
            for key in keys_to_remove:
                st.session_state.pop(key, None)
            st.success("✅ Modèle non supervisé réinitialisé avec succès !")
            st.rerun()

st.divider()

# =================================
# 4️⃣ ÉTAT DU PIPELINE
# =================================
st.header("📦 Aperçu de l’État du Pipeline")

has_data = ("data" in st.session_state or "processed_data" in st.session_state)
has_processed = "processed_data" in st.session_state
has_supervised = "trained_model" in st.session_state
has_unsupervised = "unsup_model" in st.session_state

status_data = {
    "Étape": [
        "📁 Dataset chargé",
        "🔧 Prétraitement effectué",
        "🤖 Modèle Supervisé",
        "🔍 Modèle Non Supervisé"
    ],
    "Statut": [
        "✅ Terminé" if has_data else "❌ Incomplet",
        "✅ Terminé" if has_processed else "⚠️ Partiel",
        "✅ Terminé" if has_supervised else "❌ Non entraîné",
        "✅ Terminé" if has_unsupervised else "❌ Non exécuté"
    ]
}

status_df = pd.DataFrame(status_data)
st.dataframe(status_df, use_container_width=True, hide_index=True)

if has_supervised and has_unsupervised:
    st.success("🎉 Les modèles supervisé et non supervisé sont disponibles !")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Supervisé** : {st.session_state.get('selected_algorithm', 'Inconnu')}")
    with col2:
        st.info(f"**Non Supervisé** : {st.session_state.get('unsup_model_name', 'Inconnu')}")
elif has_supervised:
    st.info(f"ℹ️ Seul le **Modèle Supervisé** est entraîné : {st.session_state.get('selected_algorithm', 'Inconnu')}")
elif has_unsupervised:
    st.info(f"ℹ️ Seul le **Modèle Non Supervisé** est exécuté : {st.session_state.get('unsup_model_name', 'Inconnu')}")
