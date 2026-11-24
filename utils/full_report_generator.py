# full_report_generator.py

import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
from utils.evaluation import test_model
from utils.plots import (
    plot_histograms, plot_correlation_matrix, scatter_plot,
    plot_learning_curve, plot_svm_decision_boundary, plot_confusion,
    plot_feature_importance, plot_clusters
)
import pandas as pd
import numpy as np

# -------------------------
# Helper to save plot to PDF in memory
# -------------------------
def plot_to_pdf(pdf, plot_func, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    tmp = BytesIO()
    plt.savefig(tmp, format='png', bbox_inches='tight')
    plt.close()
    tmp.seek(0)
    pdf.image(tmp, w=180)
    pdf.ln(5)

# -------------------------
# PDF Report
# -------------------------
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "ML Model Evaluation Report", ln=True, align="C")
        self.ln(10)

    def add_section_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def add_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 8, text)
        self.ln(5)

# -------------------------
# Main function
# -------------------------
def generate_full_report(model, X_train, y_train, X_test, y_test,
                         df=None, task="classification", feature_names=None,
                         cluster_labels=None, scatter_cols=None):
    """
    Generates a full ML report as a PDF in memory (BytesIO)
    """
    pdf = PDFReport()
    pdf.add_page()

    # 1. Metrics
    pdf.add_section_title("Model Metrics")
    metrics = test_model(model, X_test, y_test, task=task)
    metric_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    pdf.add_text(metric_text)

    # 2. Plots
    if df is not None:
        pdf.add_section_title("Histograms")
        plot_to_pdf(pdf, plot_histograms, df)

        pdf.add_section_title("Correlation Matrix")
        plot_to_pdf(pdf, plot_correlation_matrix, df)

        if scatter_cols:
            for x, y in scatter_cols:
                pdf.add_section_title(f"Scatter Plot: {x} vs {y}")
                plot_to_pdf(pdf, scatter_plot, df, x, y)

    pdf.add_section_title("Learning Curve")
    plot_to_pdf(pdf, plot_learning_curve, model, X_train, y_train)

    if hasattr(model, "predict") and "SVC" in str(type(model)):
        try:
            pdf.add_section_title("SVM Decision Boundary")
            plot_to_pdf(pdf, plot_svm_decision_boundary, model, X_test, y_test)
        except Exception as e:
            pdf.add_text(f"SVM boundary plot skipped: {e}")

    if task == "classification":
        y_pred = model.predict(X_test)
        pdf.add_section_title("Confusion Matrix")
        plot_to_pdf(pdf, plot_confusion, y_test, y_pred)

    if feature_names is not None and hasattr(model, "feature_importances_"):
        pdf.add_section_title("Feature Importance")
        plot_to_pdf(pdf, plot_feature_importance, model, feature_names)

    if cluster_labels is not None:
        pdf.add_section_title("Cluster Visualization (PCA)")
        plot_to_pdf(pdf, plot_clusters, X_test, cluster_labels)

    # Return PDF as BytesIO
    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

# -------------------------
# Example usage (for testing)
# -------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X = data.data
    y = data.target
    df = pd.DataFrame(X, columns=data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    scatter_cols = [("sepal length (cm)", "sepal width (cm)"), ("petal length (cm)", "petal width (cm)")]
    pdf_bytes = generate_full_report(
        model, X_train, y_train, X_test, y_test,
        df=df, task="classification", feature_names=data.feature_names,
        scatter_cols=scatter_cols
    )

    # Save PDF locally for testing
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes.read())
