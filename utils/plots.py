# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve


# Histograms
def plot_histograms(df):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("plot_histograms expects a pandas DataFrame.")

    df.hist(figsize=(15, 10), bins=20)
    plt.suptitle("Histograms of variables")
    plt.show()


# Correlation matrix
def plot_correlation_matrix(df):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("plot_correlation_matrix expects a pandas DataFrame.")

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.show()


# Scatter plot
def scatter_plot(df, x, y, hue=None):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")

    if x not in df.columns or y not in df.columns:
        raise ValueError("Specified columns do not exist.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f"Scatter plot: {x} vs {y}")
    plt.show()


# Learning curve
def plot_learning_curve(model, X, y):
    if model is None:
        raise ValueError("Model is None.")

    if X is None or y is None:
        raise ValueError("X or y is None.")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring="accuracy"
    )

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")
    plt.xlabel("Dataset size")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Learning curve")
    plt.show()


# SVM decision boundary
def plot_svm_decision_boundary(model, X, y):
    if model is None:
        raise ValueError("SVM model is None.")

    if X is None or y is None:
        raise ValueError("X or y is None.")

    if X.shape[1] < 2:
        raise ValueError("SVM requires at least 2 features for visualization.")

    if not hasattr(model, "predict"):
        raise TypeError("Model does not have predict() method.")

    # Apply PCA if dimension > 2
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X)
    else:
        X_plot = X

    h = 0.02
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    if X.shape[1] > 2:
        grid = pca.inverse_transform(grid)

    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolor="k")
    plt.title("SVM decision boundary")
    plt.show()


# Confusion matrix
def plot_confusion(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.show()


# Feature importance
def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not support feature_importances_.")

    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Feature number mismatch.")

    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), np.array(feature_names)[indices])
    plt.title("Feature importance")
    plt.show()


# Cluster visualization
def plot_clusters(X, labels):
    if X is None or labels is None:
        raise ValueError("X or labels is None.")

    if X.shape[0] != len(labels):
        raise ValueError("X and labels must have same length.")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("Cluster visualization (PCA)")
    plt.show()
