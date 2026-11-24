import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Prétraitement", page_icon="🔧")

st.title("🔧 Étape 2 : Prétraitement des données")

# Vérifier si les données sont chargées
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("⚠️ Aucune donnée trouvée ! Veuillez d'abord télécharger les données.")
    if st.button("Aller au téléchargement des données"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

df = st.session_state.data.copy()

# Afficher les informations actuelles du dataset
st.header("Dataset actuel")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Lignes", len(df))
with col2:
    st.metric("Colonnes", len(df.columns))
with col3:
    st.metric("Valeurs manquantes", df.isnull().sum().sum())

# Sélection de la variable cible
st.header("Sélection de la variable cible")

all_columns = df.columns.tolist()
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

target_column = st.selectbox(
    "Sélectionnez la variable cible (ce que vous voulez prédire)",
    all_columns,
    index=all_columns.index(st.session_state.target_column) if st.session_state.target_column in all_columns else 0
)
st.session_state.target_column = target_column

feature_columns = [col for col in all_columns if col != target_column]
st.session_state.feature_columns = feature_columns

st.info(f"✅ Cible sélectionnée : **{target_column}** | Caractéristiques : {len(feature_columns)} colonnes")

# Options de prétraitement
st.header("Options de prétraitement")

# Créer des onglets pour les différentes étapes
tab1, tab2, tab3, tab4 = st.tabs(["Valeurs manquantes", "Mise à l'échelle", "Encodage", "Sélection des caractéristiques"])

# Initialiser les variables avec des valeurs par défaut
strategy = "Supprimer les lignes"
constant_val = "0"
scaling_method = "Pas de mise à l'échelle"
exclude_cols = []
encoding_method = "Encodage par étiquette"
max_categories = 10
correlation_threshold = 0.9
remove_correlated = False

with tab1:
    st.subheader("Gestion des valeurs manquantes")
    
    missing_cols = df[feature_columns].isnull().sum()
    missing_cols = missing_cols[missing_cols > 0].index.tolist()
    
    if missing_cols:
        st.warning(f"Colonnes avec des valeurs manquantes : {', '.join(missing_cols)}")
        
        strategy = st.selectbox(
            "Stratégie d'imputation",
            ["Supprimer les lignes", "Moyenne (numérique uniquement)", "Médiane (numérique uniquement)", "Mode", "Valeur constante"]
        )
        
        if strategy == "Valeur constante":
            constant_val = st.text_input("Valeur constante à utiliser", "0")
        
        # Gestion spécifique aux colonnes
        st.subheader("Gestion spécifique par colonne")
        for col in missing_cols:
            col_strategy = st.selectbox(
                f"Stratégie pour {col}",
                ["Utiliser la stratégie globale", "Supprimer la colonne", "Moyenne", "Médiane", "Mode", "Personnalisé"],
                key=f"missing_{col}"
            )
            
            if col_strategy == "Personnalisé":
                custom_val = st.text_input(f"Valeur personnalisée pour {col}", key=f"custom_{col}")
    
    else:
        st.success("✅ Aucune valeur manquante trouvée dans les colonnes de caractéristiques !")

with tab2:
    st.subheader("Mise à l'échelle des caractéristiques")
    
    numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write(f"Colonnes numériques détectées : {', '.join(numeric_cols)}")
        
        scaling_method = st.radio(
            "Sélectionnez la méthode de mise à l'échelle",
            ["Pas de mise à l'échelle", "StandardScaler (moyenne=0, écart-type=1)", "MinMaxScaler (intervalle 0-1)", "RobustScaler (robuste aux valeurs aberrantes)"]
        )
        
        if scaling_method != "Pas de mise à l'échelle":
            # Option pour exclure certaines colonnes
            exclude_cols = st.multiselect(
                "Exclure certaines colonnes de la mise à l'échelle",
                numeric_cols,
                help="Sélectionnez les colonnes que vous ne souhaitez pas mettre à l'échelle"
            )
            
            scale_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if scale_cols:
                st.info(f"Colonnes à mettre à l'échelle : {', '.join(scale_cols)}")
    
    else:
        st.info("Aucune colonne numérique trouvée pour la mise à l'échelle")

with tab3:
    st.subheader("Encodage des variables catégorielles")
    
    categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        st.write(f"Colonnes catégorielles détectées : {', '.join(categorical_cols)}")
        
        encoding_method = st.radio(
            "Sélectionnez la méthode d'encodage",
            ["Encodage par étiquette", "Encodage One-Hot", "Encodage cible", "Encodage par fréquence"]
        )
        
        if encoding_method == "Encodage One-Hot":
            max_categories = st.slider(
                "Nombre maximum de catégories par colonne (pour éviter une dimensionnalité élevée)",
                2, 50, 10,
                help="Les colonnes avec plus de valeurs uniques seront encodées différemment"
            )
    
    else:
        st.success("✅ Aucune colonne catégorielle trouvée !")

with tab4:
    st.subheader("Sélection des caractéristiques")
    
    # Sélection basée sur la corrélation
    if len(feature_columns) > 1:
        remove_correlated = st.checkbox("Supprimer les caractéristiques fortement corrélées", value=False)
        
        if remove_correlated:
            correlation_threshold = st.slider(
                "Seuil de corrélation",
                0.7, 1.0, 0.9, 0.05,
                help="Les caractéristiques avec une corrélation supérieure à ce seuil seront supprimées"
            )
    
    # Sélection manuelle des caractéristiques
    st.subheader("Sélection manuelle des caractéristiques")
    
    selected_features = st.multiselect(
        "Sélectionnez les caractéristiques à conserver",
        feature_columns,
        default=feature_columns
    )
    
    st.session_state.selected_features = selected_features

# Appliquer le prétraitement
st.header("Appliquer le prétraitement")

if st.button("🚀 Appliquer le prétraitement", type="primary", use_container_width=True):
    with st.spinner("Traitement des données..."):
        try:
            processed_df = df.copy()
            
            # Gestion des valeurs manquantes
            if strategy == "Supprimer les lignes":
                processed_df = processed_df.dropna()
            elif strategy == "Moyenne (numérique uniquement)":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
                    processed_df[numeric_cols].mean()
                )
            elif strategy == "Médiane (numérique uniquement)":
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
                    processed_df[numeric_cols].median()
                )
            elif strategy == "Mode":
                for col in processed_df.columns:
                    mode_val = processed_df[col].mode()
                    if not mode_val.empty:
                        processed_df[col] = processed_df[col].fillna(mode_val[0])
            elif strategy == "Valeur constante":
                processed_df = processed_df.fillna(constant_val)
            
            # Gestion de la mise à l'échelle
            numeric_cols = processed_df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            if scaling_method != "Pas de mise à l'échelle" and numeric_cols:
                scale_cols = [col for col in numeric_cols if col not in exclude_cols]
                if scale_cols:
                    if scaling_method == "StandardScaler (moyenne=0, std=1)":
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler (range 0-1)":
                        scaler = MinMaxScaler()
                    elif scaling_method == "RobustScaler (robuste aux valeurs aberrantes)":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                    
                    processed_df[scale_cols] = scaler.fit_transform(processed_df[scale_cols])
                    st.session_state.scaler = scaler
            
            # Gestion de l'encodage catégoriel
            categorical_cols = processed_df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                if encoding_method == "Encodage par étiquette":
                    encoder = LabelEncoder()
                    for col in categorical_cols:
                        if processed_df[col].nunique() <= max_categories:
                            processed_df[col] = encoder.fit_transform(processed_df[col].astype(str))
                    st.session_state.encoder = encoder
                elif encoding_method == "Encodage One-Hot":
                    for col in categorical_cols:
                        if processed_df[col].nunique() <= max_categories:
                            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                            processed_df = pd.concat([processed_df.drop(columns=[col]), dummies], axis=1)
            
            # Gestion des caractéristiques fortement corrélées
            if remove_correlated and len(selected_features) > 1:
                numeric_selected = [col for col in selected_features if col in processed_df.select_dtypes(include=[np.number]).columns]
                if len(numeric_selected) > 1:
                    corr_matrix = processed_df[numeric_selected].corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
                    
                    if to_drop:
                        processed_df = processed_df.drop(columns=to_drop)
                        selected_features = [col for col in selected_features if col not in to_drop]
                        st.info(f"Caractéristiques fortement corrélées supprimées : {', '.join(to_drop)}")
            
            # Conserver uniquement les caractéristiques sélectionnées et la cible
            if selected_features and target_column in processed_df.columns:
                columns_to_keep = selected_features + [target_column]
                existing_columns = [col for col in columns_to_keep if col in processed_df.columns]
                processed_df = processed_df[existing_columns]

            
            # Stocker les données prétraitées
            st.session_state.processed_data = processed_df
            
            # Stocker les paramètres de prétraitement
            st.session_state.preprocessing_params = {
                'target_column': target_column,
                'feature_columns': selected_features,
                'missing_strategy': strategy,
                'scaling_method': scaling_method,
                'encoding_method': encoding_method
            }
            
            st.success("✅ Prétraitement terminé avec succès !")
            
            # Comparaison avant/après
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Données originales")
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"Shape : {df.shape}")
            
            with col2:
                st.subheader("Données prétraitées")
                st.dataframe(processed_df.head(), use_container_width=True)
                st.caption(f"Shape : {processed_df.shape}")
            
        except Exception as e:
            st.error(f"Erreur pendant le prétraitement : {str(e)}")
            import traceback
            st.error(f"Détails de l'erreur : {traceback.format_exc()}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⬅️ Retour au téléchargement des données"):
        st.switch_page("pages/1_Upload.py")

with col2:
    if st.button("🔄 Réinitialiser le prétraitement"):
        if 'processed_data' in st.session_state:
            del st.session_state.processed_data
        st.rerun()

with col3:
    if st.button("➡️ Continuer vers l'EDA"):
        if 'processed_data' in st.session_state:
            st.switch_page("pages/3_EDA.py")
        else:
            st.warning("Veuillez d'abord appliquer le prétraitement !")
