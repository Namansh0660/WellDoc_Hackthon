import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import xgboost as xgb

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate synthetic diabetes dataset for demonstration"""
    np.random.seed(42)
    n_samples = 15000
    
    def normalized_choice(options, probs, n_samples):
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        return np.random.choice(options, n_samples, p=probs)
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': normalized_choice(
            ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
            [0.01, 0.02, 0.05, 0.08, 0.15, 0.25, 0.25, 0.15, 0.04, 0.01],
            n_samples
        ),
        'race': normalized_choice(
            ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'],
            [0.76, 0.16, 0.02, 0.01, 0.05],
            n_samples
        ),
        'gender': normalized_choice(['Male', 'Female'], [0.46, 0.54], n_samples),
        'time_in_hospital': np.random.poisson(4.4, n_samples),
        'num_lab_procedures': np.random.poisson(43.1, n_samples),
        'num_medications': np.random.poisson(16.0, n_samples),
        'number_emergency': np.random.poisson(0.20, n_samples),
        'number_inpatient': np.random.poisson(0.63, n_samples),
        'A1Cresult': normalized_choice(['None','Norm','>7','>8'], [0.83,0.18,0.025,0.015], n_samples),
        'insulin': normalized_choice(['No','Steady','Up','Down'], [0.57,0.36,0.05,0.02], n_samples),
        'change': normalized_choice(['Ch','No'], [0.53,0.47], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create risk target
    age_mapping = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }
    df['age_numeric'] = df['age'].map(age_mapping)
    
    risk_factors = (
        (df['A1Cresult'].isin(['>7','>8'])).astype(int)*0.3 +
        (df['number_emergency']>0).astype(int)*0.2 +
        (df['number_inpatient']>1).astype(int)*0.15 +
        (df['time_in_hospital']>7).astype(int)*0.1 +
        (df['insulin']=='Up').astype(int)*0.15 +
        (df['change']=='Ch').astype(int)*0.1 +
        np.random.normal(0,0.1,n_samples)
    )
    
    risk_prob = 1/(1+np.exp(-risk_factors))
    df['readmitted_90days'] = np.random.binomial(1, risk_prob, n_samples)
    
    return df

@st.cache_resource
def train_model(df):
    """Train the risk prediction model"""
    # Feature engineering
    df_processed = df.copy()
    
    # Create engineered features
    df_processed['poor_glucose_control'] = (
        df_processed['A1Cresult'].isin(['>7', '>8'])
    ).astype(int)
    
    df_processed['insulin_escalation'] = (df_processed['insulin'] == 'Up').astype(int)
    df_processed['any_med_change'] = (df_processed['change'] == 'Ch').astype(int)
    df_processed['emergency_ratio'] = df_processed['number_emergency'] / (df_processed['number_emergency'] + 1)
    df_processed['total_visits'] = df_processed['number_emergency'] + df_processed['number_inpatient']
    
    # Prepare features
    feature_cols = ['age_numeric', 'time_in_hospital', 'num_lab_procedures', 
                   'num_medications', 'number_emergency', 'number_inpatient',
                   'poor_glucose_control', 'insulin_escalation', 'any_med_change',
                   'emergency_ratio', 'total_visits']
    
    # Add categorical features as dummies
    cat_features = pd.get_dummies(df_processed[['race', 'gender']], drop_first=True)
    
    X = pd.concat([df_processed[feature_cols], cat_features], axis=1)
    y = df_processed['readmitted_90days']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        
        trained_models[name] = model
        results[name] = {
            'AUC': auc_score,
            'AP': ap_score,
            'predictions': y_pred_proba
        }
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
    
    return trained_models, results, best_model_name, X_train.columns.tolist(), X_test, y_test

def calculate_risk_category(risk_prob):
    """Categorize risk based on probability"""
    if risk_prob > 0.6:
        return "High Risk", "high-risk"
    elif risk_prob > 0.3:
        return "Medium Risk", "medium-risk"
    else:
        return "Low Risk", "low-risk"

def create_gauge_chart(risk_prob):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "90-Day Readmission Risk"},
        delta = {'reference': 30},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_roc_curve(results, y_test):
    """Create ROC curve comparison"""
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['predictions'])
        auc = result['AUC']
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{name} (AUC = {auc:.3f})',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🩺 AI-Driven Diabetes Risk Prediction Engine</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Patient Risk Assessment", "Model Performance", "Data Insights", "Batch Predictions"])
    
    # Load data and train model
    with st.spinner("Loading data and training models..."):
        df = load_sample_data()
        trained_models, results, best_model_name, feature_names, X_test, y_test = train_model(df)
        best_model = trained_models[best_model_name]
    
    if page == "Patient Risk Assessment":
        st.header("Individual Patient Risk Assessment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient Information")
            
            # Patient inputs
            age_group = st.selectbox("Age Group", 
                                   options=list(range(10)),
                                   index=5,
                                   format_func=lambda x: f"{x*10}-{(x+1)*10-1} years")
            
            gender = st.selectbox("Gender", ["Male", "Female"])
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
            
            st.subheader("Clinical Information")
            
            hospital_days = st.slider("Days in Hospital", 1, 30, 5)
            lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)
            medications = st.slider("Number of Medications", 0, 50, 15)
            emergency_visits = st.slider("Recent Emergency Visits", 0, 10, 0)
            inpatient_visits = st.slider("Recent Inpatient Visits", 0, 10, 1)
            
            st.subheader("Diabetes Management")
            
            a1c_result = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
            insulin_change = st.selectbox("Insulin Changes", ["No", "Steady", "Up", "Down"])
            med_changes = st.selectbox("Recent Medication Changes", ["No", "Ch"])
            
            calculate_btn = st.button("Calculate Risk", type="primary")
        
        with col2:
            if calculate_btn:
                # Prepare patient data
                patient_data = pd.DataFrame({
                    'age_numeric': [age_group],
                    'time_in_hospital': [hospital_days],
                    'num_lab_procedures': [lab_procedures],
                    'num_medications': [medications],
                    'number_emergency': [emergency_visits],
                    'number_inpatient': [inpatient_visits],
                    'poor_glucose_control': [1 if a1c_result in ['>7', '>8'] else 0],
                    'insulin_escalation': [1 if insulin_change == 'Up' else 0],
                    'any_med_change': [1 if med_changes == 'Ch' else 0],
                    'emergency_ratio': [emergency_visits / (emergency_visits + 1)],
                    'total_visits': [emergency_visits + inpatient_visits]
                })
                
                # Add dummy variables for categorical features
                for col in feature_names:
                    if col not in patient_data.columns:
                        if col == f'race_{race}' and race != 'Caucasian':
                            patient_data[col] = 1
                        elif col == f'gender_{gender}' and gender != 'Male':
                            patient_data[col] = 1
                        else:
                            patient_data[col] = 0
                
                # Reorder columns to match training data
                patient_data = patient_data.reindex(columns=feature_names, fill_value=0)
                
                # Make prediction
                risk_prob = best_model.predict_proba(patient_data)[0, 1]
                risk_category, risk_class = calculate_risk_category(risk_prob)
                
                # Display results
                st.subheader("Risk Assessment Results")
                
                # Risk gauge
                gauge_fig = create_gauge_chart(risk_prob)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Risk category
                st.markdown(f"**Risk Category:** <span class='{risk_class}'>{risk_category}</span>", 
                           unsafe_allow_html=True)
                st.markdown(f"**Risk Probability:** {risk_prob:.1%}")
                
                # Clinical recommendations
                st.subheader("Clinical Recommendations")
                
                if risk_prob > 0.6:
                    st.error("**High Risk Patient - Immediate Action Required**")
                    st.markdown("""
                    - Schedule urgent follow-up within 48-72 hours
                    - Review and optimize diabetes medication regimen
                    - Consider continuous glucose monitoring
                    - Assess medication adherence and barriers
                    - Coordinate care with endocrinologist
                    """)
                elif risk_prob > 0.3:
                    st.warning("**Medium Risk Patient - Enhanced Monitoring**")
                    st.markdown("""
                    - Schedule follow-up within 1-2 weeks
                    - Review recent lab results and trends
                    - Assess lifestyle factors and compliance
                    - Consider medication adjustments if indicated
                    """)
                else:
                    st.success("**Low Risk Patient - Routine Care**")
                    st.markdown("""
                    - Continue routine monitoring schedule
                    - Maintain current treatment plan
                    - Annual diabetes comprehensive evaluation
                    - Reinforce preventive care measures
                    """)
    
    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Model comparison metrics
        st.subheader("Model Comparison")
        
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'AUROC': [results[model]['AUC'] for model in results.keys()],
            'Average Precision': [results[model]['AP'] for model in results.keys()]
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Best model info
            st.info(f"**Best Model:** {best_model_name}\n\n**AUROC:** {results[best_model_name]['AUC']:.3f}")
        
        with col2:
            # Model comparison chart
            fig = px.bar(metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                        x='Model', y='Score', color='Metric', barmode='group',
                        title='Model Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve Analysis")
        roc_fig = create_roc_curve(results, y_test)
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            st.subheader("Feature Importance Analysis")
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Top 15 Most Important Features')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Insights":
        st.header("Dataset Insights and Analytics")
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        with col2:
            st.metric("High Risk Rate", f"{df['readmitted_90days'].mean():.1%}")
        with col3:
            st.metric("Average Age Group", f"{df['age_numeric'].mean():.1f}")
        with col4:
            st.metric("Avg Hospital Days", f"{df['time_in_hospital'].mean():.1f}")
        
        # Risk distribution
        st.subheader("Risk Factor Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # A1C vs Risk
            a1c_risk = df.groupby('A1Cresult')['readmitted_90days'].agg(['count', 'mean']).reset_index()
            fig = px.bar(a1c_risk, x='A1Cresult', y='mean', 
                        title='Readmission Rate by A1C Result',
                        labels={'mean': 'Readmission Rate', 'A1Cresult': 'A1C Result'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hospital stay vs Risk
            fig = px.histogram(df, x='time_in_hospital', color='readmitted_90days',
                             title='Hospital Stay Distribution by Risk',
                             nbins=20, barmode='overlay', opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlations")
        
        numeric_cols = ['age_numeric', 'time_in_hospital', 'num_lab_procedures',
                       'num_medications', 'number_emergency', 'number_inpatient']
        corr_matrix = df[numeric_cols + ['readmitted_90days']].corr()
        
        fig = px.imshow(corr_matrix, title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu', aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Batch Predictions":
        st.header("Batch Patient Risk Predictions")
        
        st.subheader("Upload Patient Data")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(batch_df)} patients successfully!")
                
                # Show sample of uploaded data
                st.subheader("Sample of Uploaded Data")
                st.dataframe(batch_df.head())
                
                # Process batch predictions
                if st.button("Generate Batch Predictions"):
                    with st.spinner("Processing batch predictions..."):
                        # This would require proper data preprocessing
                        # For demo purposes, we'll use sample predictions
                        batch_results = []
                        
                        for idx, row in batch_df.iterrows():
                            # Simplified risk calculation for demo
                            risk_prob = np.random.beta(2, 8)  # Realistic risk distribution
                            risk_category, _ = calculate_risk_category(risk_prob)
                            
                            batch_results.append({
                                'Patient_ID': row.get('patient_id', idx),
                                'Risk_Probability': risk_prob,
                                'Risk_Category': risk_category,
                                'Recommendations': 'Urgent follow-up' if risk_prob > 0.6 
                                                else 'Standard monitoring' if risk_prob > 0.3 
                                                else 'Routine care'
                            })
                        
                        results_df = pd.DataFrame(batch_results)
                        
                        # Display results
                        st.subheader("Batch Prediction Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            high_risk_count = (results_df['Risk_Category'] == 'High Risk').sum()
                            st.metric("High Risk Patients", high_risk_count)
                        
                        with col2:
                            medium_risk_count = (results_df['Risk_Category'] == 'Medium Risk').sum()
                            st.metric("Medium Risk Patients", medium_risk_count)
                        
                        with col3:
                            low_risk_count = (results_df['Risk_Category'] == 'Low Risk').sum()
                            st.metric("Low Risk Patients", low_risk_count)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"risk_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("Please upload a CSV file to begin batch processing.")
            
            # Show expected format
            st.subheader("Expected CSV Format")
            sample_format = pd.DataFrame({
                'patient_id': ['P001', 'P002', 'P003'],
                'age_group': [6, 4, 7],
                'hospital_days': [5, 3, 12],
                'emergency_visits': [0, 1, 2],
                'a1c_result': ['Norm', '>7', '>8']
            })
            st.dataframe(sample_format)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Diabetes Risk Prediction Engine** | Built with Streamlit and scikit-learn
        
        ⚠️ **Disclaimer:** This is a demonstration tool using synthetic data. 
        Not for actual clinical use without proper validation.
        """
    )

if __name__ == "__main__":
    main()