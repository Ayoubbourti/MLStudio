from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
)


# Classification metrics
def classification_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-score": f1_score(y_true, y_pred, average="weighted")
    }


# Regression metrics
def regression_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


# Global model test
def test_model(model, X_test, y_test, task="classification"):
    if model is None:
        raise ValueError("Model is None.")

    if not hasattr(model, "predict"):
        raise TypeError("Model does not have predict() method.")

    y_pred = model.predict(X_test)

    if task == "classification":
        return classification_metrics(y_test, y_pred)

    elif task == "regression":
        return regression_metrics(y_test, y_pred)

    else:
        raise ValueError("task must be 'classification' or 'regression'.")
