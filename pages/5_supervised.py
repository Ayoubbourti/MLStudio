import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# Tentative d'import de XGBoost et LightGBM (optionnels)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# -------------------------------------------------
# Configuration de la page
# -------------------------------------------------
st.set_page_config(page_title="Apprentissage supervisé", page_icon="🤖", layout="wide")
st.title("🤖 Étape 5 : Apprentissage supervisé")

# -------------------------------------------------
# Vérification du session state
# -------------------------------------------------
required_keys = ["selected_algorithm", "problem_type", "target_column"]
missing = [k for k in required_keys if k not in st.session_state]

if missing:
    st.error(f"❌ États requis manquants : {missing}")
    st.stop()

# -------------------------------------------------
# Chargement des données
# -------------------------------------------------
if "processed_data" in st.session_state and st.session_state.processed_data is not None:
    df = st.session_state.processed_data.copy()
    data_source = "Données prétraitées"
elif "data" in st.session_state and st.session_state.data is not None:
    df = st.session_state.data.copy()
    data_source = "Données originales"
else:
    st.error("❌ Aucune donnée disponible. Veuillez d’abord importer et prétraiter les données.")
    st.stop()

target = st.session_state.target_column
problem_type = st.session_state.problem_type
selected_algorithm = st.session_state.selected_algorithm

# -------------------------------------------------
# Construction de X et y (AUCUN prétraitement ici)
# -------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------------------
# Assurer uniquement des variables numériques (REQUIS par sklearn)
# -------------------------------------------------
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    st.warning(
        "⚠️ Les colonnes non numériques ont été automatiquement exclues :\n"
        + ", ".join(non_numeric_cols)
    )
    X = X.select_dtypes(include=[np.number])

# Stockage pour les pages suivantes
st.session_state.X = X
st.session_state.y = y

# -------------------------------------------------
# Informations UI
# -------------------------------------------------
st.markdown(f"### Modèle sélectionné : **{selected_algorithm}**")
st.markdown(f"### Type de problème : **{problem_type}**")
st.info(f"📁 Source des données : **{data_source}** | Dimensions : {df.shape}")
st.info(f"📊 Variables utilisées pour l’entraînement : {X.shape[1]} colonnes numériques")

# -------------------------------------------------
# Séparation Train / Test
# -------------------------------------------------
st.subheader("Séparation Entraînement / Test")

col1, col2 = st.columns(2)
with col1:
    test_size = st.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
with col2:
    random_state = st.number_input("Graine aléatoire", 0, 1000, 42)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y if problem_type == "Classification" else None
)

# -------------------------------------------------
# Configuration du modèle
# -------------------------------------------------
st.subheader("Configuration du modèle")

model = None
model_params = {}

# ============================================
# MODÈLES DE CLASSIFICATION
# ============================================
if selected_algorithm == "Logistic Regression":
    C = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, 0.01)
    model = LogisticRegression(C=C, max_iter=1000)
    model_params = {"C": C}

elif selected_algorithm == "Naive Bayes":
    model = GaussianNB()

elif selected_algorithm == "Decision Tree":
    max_depth = st.slider("Profondeur maximale", 2, 50, 10)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model_params = {"max_depth": max_depth}

elif selected_algorithm == "Random Forest":
    n_estimators = st.slider("Nombre d’arbres", 50, 500, 100, 50)
    max_depth = st.slider("Profondeur maximale", 2, 50, 10)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model_params = {"n_estimators": n_estimators, "max_depth": max_depth}

elif selected_algorithm == "K-Nearest Neighbors (KNN)":
    k = st.slider("Nombre de voisins (k)", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model_params = {"n_neighbors": k}

elif selected_algorithm == "Support Vector Machine (SVM)":
    C = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, 0.01)
    kernel = st.selectbox("Noyau (kernel)", ["linear", "rbf", "poly", "sigmoid"])
    model = SVC(C=C, kernel=kernel, random_state=random_state)
    model_params = {"C": C, "kernel": kernel}

elif selected_algorithm == "AdaBoost":
    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 2.0, 1.0, 0.01)
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate}

elif selected_algorithm == "Gradient Boosting (GBM)":
    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

elif selected_algorithm == "XGBoost":
    if not HAS_XGBOOST:
        st.error("❌ XGBoost n’est pas installé. Installez-le avec : pip install xgboost")
        st.stop()

    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric='logloss'
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

elif selected_algorithm == "LightGBM":
    if not HAS_LIGHTGBM:
        st.error("❌ LightGBM n’est pas installé. Installez-le avec : pip install lightgbm")
        st.stop()

    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbose=-1
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

# ============================================
# MODÈLES DE RÉGRESSION
# ============================================
elif selected_algorithm == "Linear Regression":
    model = LinearRegression()

elif selected_algorithm == "Support Vector Regression (SVR)":
    C = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, 0.01)
    kernel = st.selectbox("Noyau (kernel)", ["linear", "rbf", "poly"])
    model = SVR(C=C, kernel=kernel)
    model_params = {"C": C, "kernel": kernel}

elif selected_algorithm == "Decision Tree Regressor":
    max_depth = st.slider("Profondeur maximale", 2, 50, 10)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model_params = {"max_depth": max_depth}

elif selected_algorithm == "Random Forest Regressor":
    n_estimators = st.slider("Nombre d’arbres", 50, 500, 100, 50)
    max_depth = st.slider("Profondeur maximale", 2, 50, 10)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model_params = {"n_estimators": n_estimators, "max_depth": max_depth}

elif selected_algorithm == "AdaBoost Regressor":
    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 2.0, 1.0, 0.01)
    model = AdaBoostRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
    model_params = {"n_estimators": n_estimators, "learning_rate": learning_rate}

elif selected_algorithm == "Gradient Boosting Regressor (GBM)":
    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

elif selected_algorithm == "XGBoost Regressor":
    if not HAS_XGBOOST:
        st.error("❌ XGBoost n’est pas installé. Installez-le avec : pip install xgboost")
        st.stop()

    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

elif selected_algorithm == "LightGBM Regressor":
    if not HAS_LIGHTGBM:
        st.error("❌ LightGBM n’est pas installé. Installez-le avec : pip install lightgbm")
        st.stop()

    n_estimators = st.slider("Nombre d’estimateurs", 50, 500, 100, 50)
    learning_rate = st.slider("Taux d’apprentissage", 0.01, 0.3, 0.1, 0.01)
    max_depth = st.slider("Profondeur maximale", 2, 10, 3)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbose=-1
    )
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth
    }

# -------------------------------------------------
# Entraînement du modèle
# -------------------------------------------------
st.subheader("Entraîner le modèle")

if st.button("🚀 Entraîner le modèle"):
    with st.spinner("Entraînement du modèle en cours..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    st.success("✅ Modèle entraîné avec succès !")

    # -------------------------------------------------
    # Métriques (logique supervisée pure)
    # -------------------------------------------------
    if problem_type == "Classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        cols = st.columns(4)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k.upper(), f"{v:.4f}")

    else:
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

        cols = st.columns(3)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k.upper(), f"{v:.4f}")

    # -------------------------------------------------
    # Stockage pour les étapes suivantes
    # -------------------------------------------------
    st.session_state.trained_model = model
    st.session_state.metrics = metrics
    st.session_state.model_params = model_params
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred

# -------------------------------------------------
# Navigation vers les résultats
# -------------------------------------------------
if "trained_model" in st.session_state:
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📊 Voir le tableau de résultats", use_container_width=True, type="primary"):
            st.switch_page("pages/7_Results.py")
