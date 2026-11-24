
# MLStudio
### End-to-End Machine Learning Platform

MLStudio is a full-stack web application designed to simplify the machine learning workflow, from raw dataset exploration to model evaluation and result export.

The platform provides an integrated environment for dataset management, data preprocessing, interactive visualization, machine learning model training, and performance comparison.

Built with **React** and **FastAPI**, MLStudio combines an intuitive user interface with a Python-based machine learning backend.

---

## Key Features

### 1. Dataset Management
- Upload datasets in CSV and Excel formats.
- Preview dataset contents.
- Explore rows, columns, data types, and statistical summaries.

### 2. Data Preprocessing
- Handle missing values.
- Encode categorical variables.
- Normalize and scale numerical features.
- Preview transformed datasets.

### 3. Exploratory Data Analysis
- Generate histograms and box plots.
- Explore relationships using scatter plots.
- Visualize feature correlations through heatmaps.
- Customize data visualizations.

### 4. Machine Learning
Train and experiment with multiple machine learning algorithms:

| Category | Algorithms |
|---|---|
| Regression | Linear Regression |
| Classification | Logistic Regression, SVM, Decision Tree, Random Forest, KNN |
| Clustering | K-Means |
| Deep Learning | Neural Networks (optional extension) |

### 5. Model Evaluation
- Train/test dataset splitting.
- Accuracy, Precision, and Recall.
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
- R² score.
- Confusion matrix.
- Model performance comparison.

### 6. Export and Results
- Download trained models.
- Export predictions.
- Download generated charts.
- Export evaluation metrics.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, Vite, Axios, React Router |
| Backend | Python, FastAPI, Uvicorn |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Styling | Tailwind CSS (optional) |

---

## Project Architecture

```text
MLStudio/
│
├── backend/
│   ├── app/
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── docs/
│   ├── architecture/
│   └── api/
│
└── README.md
```

---

## Getting Started

### Prerequisites

Make sure you have installed:

- Python 3.10+
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Ayoubbourti/MLStudio.git
cd MLStudio
```

### 2. Start the Backend

```bash
cd backend

python -m venv .venv
```

Activate the virtual environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Backend URL: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 3. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: http://localhost:5173/

---

## Machine Learning Workflow

```text
Dataset Upload
      |
      v
Data Exploration
      |
      v
Data Preprocessing
      |
      v
Feature Preparation
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Performance Comparison
      |
      v
Export Results
```

---

## Project Objectives

MLStudio aims to provide a practical and accessible environment for experimenting with machine learning workflows through a unified web interface.

The project brings together data engineering, exploratory data analysis, preprocessing, model development, and evaluation in a single application.

---

## Author

**Ayoub Bourti**

Master's Student in Artificial Intelligence & Data Science

GitHub: [@Ayoubbourti](https://github.com/Ayoubbourti)

LinkedIn: [Ayoub Bourti](https://www.linkedin.com/in/ayoub-bourti-b831b22b4/)

---

*MLStudio — From raw data to machine learning insights.*
