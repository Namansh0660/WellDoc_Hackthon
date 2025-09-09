
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

