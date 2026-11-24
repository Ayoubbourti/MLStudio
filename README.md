Python-ML-Project
ML Platform – Machine Learning Web Application

A complete web platform for dataset management, preprocessing, visualization, model training, comparison, and exporting results.
Built using React (frontend) and FastAPI (backend).

🚀 Tech Stack
Frontend

React (Vite)

Axios

TailwindCSS (optional)

React Router

Backend

FastAPI

Pandas / NumPy

Scikit-Learn

Matplotlib / Seaborn

Uvicorn

🎯 Main Features
📂 Dataset Management

✔️ Upload & preview datasets (CSV, Excel)
✔️ Display rows, columns, and statistics

🔧 Preprocessing Tools

✔️ Clean missing data
✔️ Encode categorical features
✔️ Normalize / scale numerical data
✔️ Preview the transformed dataset

📊 Data Visualization

✔️ Histograms
✔️ Box plots
✔️ Scatter plots
✔️ Correlation heatmaps
✔️ Customizable charts

🤖 Machine Learning Models

✔️ Linear Regression
✔️ Logistic Regression
✔️ SVM
✔️ Decision Trees
✔️ Random Forest
✔️ KNN
✔️ K-Means Clustering
✔️ Neural Networks (optional extension)

📈 Model Evaluation

✔️ Accuracy, Precision, Recall
✔️ MSE, RMSE, R²
✔️ Confusion matrix
✔️ Train/test split

🧾 Export

✔️ Download trained models
✔️ Download predictions
✔️ Download charts & metrics

📁 Project Structure
backend/   → FastAPI backend (ML services)
frontend/  → React SPA (UI & visualizations)
docs/      → Technical documentation, architecture, API spec

🛠️ How to Run the Project
🔹 Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload


➡️ API will run at:
http://127.0.0.1:8000

➡️ Swagger docs:
http://127.0.0.1:8000/docs

🔹 Frontend (React)
cd frontend
npm install
npm run dev


➡️ Frontend will run at:
http://localhost:5173/
 (default Vite port)
