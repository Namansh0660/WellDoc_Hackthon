
# 🩺 AI-Driven Diabetes Risk Prediction Engine

This repository contains a **Streamlit web application** and a **Jupyter Notebook** designed for predicting **90-day readmission risk** in diabetic patients using machine learning models. The project was built for the **WellDoc Hackathon** and demonstrates AI-driven healthcare insights with interactive dashboards.

---

## 📂 Repository Structure

```
├── WellDoc_Hackathon.ipynb   # Jupyter Notebook (data exploration, feature engineering, model training)
├── app.py                    # Streamlit web app for risk prediction & insights
├── requirements.txt          # Python dependencies
├── risk_prediction_model.pkl # Pre-trained model (if provided)
├── README.md                 # Project documentation
```

---

## 🚀 Features

✅ **Individual Patient Risk Assessment** – Input patient details and get risk probability + recommendations.  
✅ **Model Performance Dashboard** – Compare Random Forest, Gradient Boosting, and XGBoost with AUROC and PR curves.  
✅ **Data Insights & Analytics** – Visualize dataset statistics, correlations, and risk distributions.  
✅ **Batch Predictions** – Upload CSV files of multiple patients to generate batch predictions.  
✅ **Interactive Visualizations** – Plotly-powered charts for ROC curves, feature importance, and risk gauges.  
✅ **Synthetic Dataset Generator** – No external data needed, dataset is generated dynamically.  

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` 🚀

---

## 📊 Usage

### ▶️ **Jupyter Notebook**
- Open `WellDoc_Hackathon.ipynb` in Jupyter/Colab.
- Contains dataset preparation, feature engineering, and model training.

### ▶️ **Streamlit App**
- Navigate between pages:
  - **Patient Risk Assessment** – Get AI-powered predictions for individuals.  
  - **Model Performance** – Compare different ML models.  
  - **Data Insights** – Explore correlations & trends.  
  - **Batch Predictions** – Upload CSV for multiple patients.  

---

## 📈 Expected CSV Format for Batch Upload

When using **Batch Predictions**, upload a CSV like this:

| patient_id | age_group | hospital_days | emergency_visits | a1c_result |
|------------|-----------|---------------|------------------|------------|
| P001       | 6         | 5             | 0                | Norm       |
| P002       | 4         | 3             | 1                | >7         |
| P003       | 7         | 12            | 2                | >8         |

---

## 📦 Dependencies

- Python 3.8+
- Streamlit
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Plotly

(See `requirements.txt` for full list)

---

## 📌 Notes

- The app uses **synthetic patient data** for demonstration purposes.  
- Models are trained on generated data, **not real medical datasets**.  
- ⚠️ **Disclaimer:** This tool is for demonstration only and should not be used for actual clinical decision-making without proper validation.

---

## 🤝 Contributors

👨‍💻 Developed for **WellDoc Hackathon 2025**  
✨ Built with ❤️ using Streamlit & scikit-learn

---

## 📜 License
MIT Licernse

This project is licensed under the MIT License.





🩺 Diabetes Risk Prediction Engine

This repository contains a Streamlit-based interactive dashboard for predicting diabetes readmission risks using machine learning models. The system simulates a hospital risk assessment tool, allowing healthcare professionals and researchers to:

Assess individual patient risk.

Compare model performances.

Explore data insights.

Perform batch predictions for multiple patients.

⚠️ Disclaimer: The dataset used here is synthetic (generated within the app) and is intended for demonstration only. This project is not for actual clinical use without proper validation on real-world medical datasets.

🚀 Features

Patient Risk Assessment

Input patient demographics, hospitalization details, and diabetes management factors.

Predict 90-day readmission risk with probability, category (Low/Medium/High), and clinical recommendations.

Visualize risk probability with a gauge chart.

Model Performance

Compare multiple machine learning models:

Random Forest

Gradient Boosting

XGBoost

Evaluate metrics like AUROC and Average Precision.

Visualize ROC curves and feature importance.

Data Insights

Explore dataset statistics, distributions, and correlations.

Risk factor analysis (A1C, hospital stay, etc.).

Interactive plots with Plotly.

Batch Predictions

Upload a CSV file with patient data.

Generate risk predictions for multiple patients simultaneously.

Download results as CSV for further use.

📂 Repository Structure
├── app.py                   # Streamlit dashboard  
├── WellDoc_Hackathon.ipynb  # Jupyter notebook (model experiments & development)  
├── requirements.txt         # Python dependencies  
└── README.md                # Project documentation (this file)  

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/diabetes-risk-dashboard.git
cd diabetes-risk-dashboard


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Open your browser at http://localhost:8501
.

🖥️ Usage Guide
1. Patient Risk Assessment

Navigate to Patient Risk Assessment.

Enter patient details (age group, gender, race, hospital days, medications, etc.).

Click Calculate Risk.

View:

Risk probability & category

Gauge chart visualization

Clinical recommendations

2. Model Performance

Compare Random Forest, Gradient Boosting, and XGBoost.

Metrics shown: AUROC, Average Precision.

Visualizations:

ROC curve comparison

Feature importance (top 15 features)

3. Data Insights

Dataset statistics: total patients, high-risk rate, avg. hospital stay, etc.

Interactive plots:

A1C vs Readmission rate

Hospital stay distribution by risk

Correlation heatmap

4. Batch Predictions

Upload a CSV file with patient records.

Example format:

patient_id,age_group,hospital_days,emergency_visits,a1c_result
P001,6,5,0,Norm
P002,4,3,1,>7
P003,7,12,2,>8


Generate predictions → download results as CSV.

View summary stats (counts of High/Medium/Low risk patients).

📊 Models Used

Random Forest

Gradient Boosting

XGBoost

Models are trained on synthetic patient data generated within the app.

📝 Notes

Built with: Streamlit, scikit-learn, XGBoost, Plotly, Pandas, NumPy.

The notebook (WellDoc_Hackathon.ipynb) contains model experimentation and preprocessing pipelines.

Current implementation uses synthetic data → should be retrained with real patient datasets for deployment in healthcare.
