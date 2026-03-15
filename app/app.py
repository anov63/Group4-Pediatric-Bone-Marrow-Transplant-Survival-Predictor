import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import base64
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bone Marrow Transplant Survival Predictor",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'Nunito', sans-serif !important;
    }}

    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    
    .block-container {{
        background-color: rgba(255, 255, 255, 0.65); 
        backdrop-filter: blur(12px); 
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 2rem;
        max-width: 1100px !important;  
        margin-left: auto !important;
        margin-right: auto !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.08); 
        border: 1px solid rgba(255, 255, 255, 0.5);
    }}
    
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.95);
    }}
    
    div.stButton > button:first-child {{
        background-color: #df586d; 
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 10px rgba(223, 88, 109, 0.3);
        border: none;
        transition: all 0.3s ease-in-out;
    }}
    
    div.stButton > button:hover {{
        background-color: #c44558;
        box-shadow: 0px 6px 15px rgba(223, 88, 109, 0.5);
        transform: translateY(-2px);
    }}
    
    [data-testid="stMetric"] {{
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #df586d;
    }}
    
    .streamlit-expanderHeader {{
        font-weight: 600;
        font-size: 16px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_png_as_page_bg("Red Minimalist Healthcare Presentation.png")
except FileNotFoundError:
    st.warning("Background image 'Red Minimalist Healthcare Presentation.png' not found. Please ensure it is in the same directory as this script.")

st.title("🩸 Pediatric Bone Marrow Transplant Survival Predictor")
st.markdown("Predict the survival status of pediatric patients after a bone marrow transplant using Machine Learning.")

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / 'models'

models_info = {
    'XGBoost': models_dir / 'xgboost_model.pkl',
    'SVM': models_dir / 'modele_svm_bmt.pkl',
    'Random Forest': models_dir / 'rf_model.pkl',
    'LightGBM': models_dir / 'lgbm_model.pkl'
}

if 'show_other_models' not in st.session_state:
    st.session_state.show_other_models = False

st.sidebar.header("⚙️ Model Settings")

if st.sidebar.button("Test other models"):
    st.session_state.show_other_models = not st.session_state.show_other_models

if st.session_state.show_other_models:
    selected_model_name = st.sidebar.selectbox(
        "Choose a trained model", 
        list(models_info.keys()),
        index=list(models_info.keys()).index('XGBoost')
    )
else:
    selected_model_name = 'XGBoost'
    st.sidebar.info("Using default model: **XGBoost**")

model_path = models_info[selected_model_name]

if not model_path.exists():
    st.sidebar.error(f"Model file not found: {model_path.name}")
    st.stop()

try:
    model = load_model(model_path)
    st.sidebar.success(f"Successfully loaded {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.header("Patient & Donor Data Input")
st.markdown("Expand the sections below to enter the clinical features:")

with st.expander("📊 Continuous Medical Metrics", expanded=True):
    col_cont1, col_cont2, col_cont3 = st.columns(3)
    with col_cont1:
        recipient_age = st.slider("Recipient Age (years)", 0.0, 25.0, 10.0, 0.1)
        body_mass = st.slider("Recipient Body Mass (kg)", 0.0, 150.0, 30.0, 1.0)
    with col_cont2:
        cd34_dose = st.slider("CD34 Dose (log_CD34kgx10d6)", -0.5, 5.0, 1.5, 0.1)
        cd3d_dose = st.slider("CD3d Dose (CD3dkgx10d8)", 0.0, 200.0, 50.0, 1.0)
    with col_cont3:
        cd3dcd34_ratio = st.slider("CD3d/CD34 Ratio", 0.0, 200.0, 50.0, 1.0)

with st.expander("🏥 Demographic & Disease Information"):
    col_dem1, col_dem2 = st.columns(2)
    with col_dem1:
        recipient_gender = st.selectbox("Recipient Gender", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        disease = st.selectbox("Disease Type", options=[0, 1, 2, 3, 4], format_func=lambda x: ["ALL", "AML", "Chronic", "Lymphoma", "Nonmalignant"][x])
        disease_group = st.selectbox("Disease Group (0, 1, 2)", [0, 1, 2])
    with col_dem2:
        risk_group = st.selectbox("Risk Group", options=[0, 1], format_func=lambda x: "Low" if x==0 else "High")
        tx_post_relapse = st.selectbox("Tx Post Relapse", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        stem_cell_source = st.selectbox("Stem Cell Source", options=[0, 1], format_func=lambda x: "Bone Marrow" if x==0 else "Peripheral Blood")

with st.expander("🧬 Donor & Compatibility Matching"):
    col_don1, col_don2, col_don3 = st.columns(3)
    with col_don1:
        donor_age_35 = st.selectbox("Donor Age > 35", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        gender_match = st.selectbox("Gender Match", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        hla_match = st.selectbox("HLA Match", options=[0, 1], format_func=lambda x: "Mismatch" if x==0 else "Match")
        hla_mismatch = st.selectbox("HLA Mismatch Degree", [0, 1, 2])
    with col_don2:
        abo_match = st.selectbox("ABO Match", options=[0, 1], format_func=lambda x: "Mismatch" if x==0 else "Match")
        donor_abo = st.selectbox("Donor ABO", options=[0, 1, 2, 3], format_func=lambda x: ["A", "B", "AB", "O"][x])
        recipient_abo = st.selectbox("Recipient ABO", options=[0, 1, 2, 3], format_func=lambda x: ["A", "B", "AB", "O"][x])
        recipient_rh = st.selectbox("Recipient Rh", options=[0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
    with col_don3:
        donor_cmv = st.selectbox("Donor CMV Status", options=[0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
        recipient_cmv = st.selectbox("Recipient CMV Status", options=[0, 1], format_func=lambda x: "Negative" if x==0 else "Positive")
        ii_iv = st.selectbox("aGVHD II-IV", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

input_data = {
    'Recipientage': recipient_age,
    'log_CD34kgx10d6': cd34_dose,
    'Rbodymass': body_mass,
    'RecipientRh': recipient_rh,
    'Riskgroup': risk_group,
    'Stemcellsource': stem_cell_source,
    'Txpostrelapse': tx_post_relapse,
    'Donorage35': donor_age_35,
    'RecipientCMV': recipient_cmv,
    'CD3dkgx10d8': cd3d_dose,
    'CD3dCD34': cd3dcd34_ratio,
    'Diseasegroup': disease_group,
    'ABOmatch': abo_match,
    'IIIV': ii_iv,
    'DonorCMV': donor_cmv,
    'HLAmatch': hla_match,
    'RecipientABO': recipient_abo,
    'Recipientgender': recipient_gender,
    'Disease': disease,
    'Gendermatch': gender_match,
    'HLAmismatch': hla_mismatch,
    'DonorABO': donor_abo
}

df_input = pd.DataFrame([input_data])

exact_columns = [
    'Recipientage', 'log_CD34kgx10d6', 'Rbodymass', 'RecipientRh', 'Riskgroup', 
    'Stemcellsource', 'Txpostrelapse', 'Donorage35', 'RecipientCMV', 'CD3dkgx10d8', 
    'CD3dCD34', 'Diseasegroup', 'ABOmatch', 'IIIV', 'DonorCMV', 'HLAmatch', 
    'RecipientABO', 'Recipientgender', 'Disease', 'Gendermatch', 'HLAmismatch', 'DonorABO'
]
df_input = df_input[exact_columns]

if selected_model_name == 'LightGBM':
    categorical_cols = [
        'Recipientgender', 'Stemcellsource', 'Donorage35', 'IIIV', 'Gendermatch',
        'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'DonorCMV', 
        'RecipientCMV', 'Disease', 'Riskgroup', 'Txpostrelapse', 'Diseasegroup', 
        'HLAmatch', 'HLAmismatch'
    ]
    for c in categorical_cols:
        df_input[c] = df_input[c].astype('category')

st.markdown("---")

if st.button("🔍 Predict Survival Status", type="primary", use_container_width=True):
    with st.spinner(f"Running {selected_model_name} inference..."):
        try:
            prediction = model.predict(df_input)[0]
            
            if hasattr(model, 'predict_proba'):
                raw_prediction = model.predict_proba(df_input)
                probability_value = raw_prediction[0][1]
                clean_probability = float(probability_value)
            else:
                clean_probability = float(prediction)
                
            st.markdown("### 📋 Prediction Results")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error("⚠️ **High Risk Profile Detected (Deceased Predicted)**")
                    st.markdown("The model indicates a lower probability of survival based on the provided clinical metrics.")
                else:
                    st.success("✅ **Favorable Profile Detected (Survival Predicted)**")
                    st.markdown("The model indicates a higher probability of survival based on the provided clinical metrics.")
                    
            with col_res2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = clean_probability * 100,
                    number = {'suffix': "%", 'font': {'size': 40, 'color': '#212121', 'family': 'Nunito'}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Calculated Risk", 'font': {'size': 18, 'family': 'Nunito'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#df586d"},
                        'bgcolor': "white",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(76, 175, 80, 0.2)"},
                            {'range': [40, 70], 'color': "rgba(255, 235, 59, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(223, 88, 109, 0.2)"}
                        ],
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Nunito"},
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=250
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if selected_model_name == 'XGBoost':
                st.markdown("---")
                st.markdown("### 🧠 Model Reasoning (SHAP)")
                st.markdown("The chart below explains how each clinical feature contributed to this specific patient's risk score. Red pushes the risk higher, blue pushes the risk lower.")
                
                with st.expander("📊 View Feature Impacts", expanded=False):
                    with st.spinner("Calculating SHAP values..."):
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer(df_input)
                            
                            shap.plots.bar(shap_values[0], show=False)
                            
                            fig_shap = plt.gcf()
                            ax_shap = plt.gca()
                            
                            fig_shap.patch.set_alpha(0.0)
                            ax_shap.patch.set_alpha(0.0)
                            
                            st.pyplot(fig_shap, transparent=True)
                            
                            plt.clf() 
                            
                        except Exception as e:
                            st.error(f"Could not generate SHAP explanation: {e}")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
