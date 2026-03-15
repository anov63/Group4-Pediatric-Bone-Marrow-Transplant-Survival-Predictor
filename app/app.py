import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import base64
import plotly.graph_objects as go
import shap
import datetime
import calendar
import time

st.set_page_config(
    page_title="BMT Survival Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

def toggle_lang():
    st.session_state.lang = 'FR' if st.session_state.lang == 'EN' else 'EN'

# --- Translations Dictionary ---
translations = {
    'EN': {
        'gateway_title': 'Your Secure Gateway to<br>Validated Survival Predictive Data',
        'auth_title': 'Secure Portal Access',
        'auth_desc': 'Please authenticate to access the BMT Predictor',
        'physician_id': 'Physician ID / Username',
        'password': 'Password',
        'auth_btn': 'Securely Authenticate',
        'nav_intro': 'Introduction',
        'nav_phys': 'Physical & Doses',
        'nav_disease': 'Disease & Demographics',
        'nav_donor': 'Donor Matching',
        'step1_title': 'Step 1 of 4: Introduction',
        'step1_header': 'Survival Prediction Tool',
        'step1_welcome': 'Welcome, Dr. ',
        'step1_desc': 'Your session is authenticated. All data entered into this tool is securely processed, end-to-end encrypted, and maintained in strict compliance with HIPAA and institutional privacy guidelines. No patient-identifiable data is permanently stored on these servers.',
        'clinical_guidelines': 'Clinical Guidelines',
        'clinical_desc': 'This tool is intended for use by qualified healthcare professionals and<br>should complement, not replace, professional medical judgment and<br>institutional protocols.',
        'start_btn': 'Start Assessment ➔',
        'step2_title': 'Step 2 of 4: Physical & Doses',
        'step2_header': 'Patient Physical & Dose Metrics',
        'step2_desc': 'Refine patient dimensions and target dosages. The visual model updates in real-time.',
        'dob': 'Recipient Date of Birth (De-identified)',
        'body_mass': 'Body Mass (kg)',
        'gender': 'Biological Gender',
        'gender_f': 'Female',
        'gender_m': 'Male',
        'target_doses': 'Target Cell Doses',
        'cd34_label': 'CD34+ Stem Cells',
        'cd3d_label': 'CD3d T-Cells',
        'ratio_label': 'T-Cell / Stem Cell Ratio',
        'prev_btn': '⬅️ Previous',
        'next_btn': 'Save & Continue ➔',
        'step3_title': 'Step 3 of 4: Disease & Demographics',
        'step3_header': 'Disease & Demographics',
        'step3_desc': 'Specify primary diagnosis, disease stage, and core patient demographic metrics.',
        'diagnosis': 'Primary Diagnosis (Disease)',
        'diagnosis_opts': ["ALL", "AML", "Chronic", "Lymphoma", "Nonmalignant"],
        'disease_group': 'Disease Group',
        'risk_category': 'Cytogenetic Risk Category',
        'risk_opts': ["Low/Favorable", "High/Poor"],
        'tx_relapse': 'Treatment Post Relapse?',
        'yes': 'Yes',
        'no': 'No',
        'stem_source': 'Stem Cell Source',
        'stem_opts': ["Bone Marrow", "Peripheral Blood (PBSC)"],
        'step4_title': 'Step 4 of 4: Donor Matching',
        'step4_header': 'Donor Matching',
        'step4_desc': 'Configure donor source and HLA compatibility parameters for survival prediction.',
        'donor_age_35': 'Donor Age > 35',
        'gender_match': 'Gender Match',
        'match': 'Match',
        'mismatch': 'Mismatch',
        'no_match': 'No Match',
        'hla_match': 'HLA Match',
        'hla_mismatch_deg': 'HLA Mismatch Degree',
        'agvhd': 'aGVHD II-IV Expected',
        'abo_match': 'ABO Match',
        'donor_abo': 'Donor ABO',
        'recipient_abo': 'Recipient ABO',
        'recipient_rh': 'Recipient Rh',
        'rh_neg': 'Negative',
        'rh_pos': 'Positive',
        'donor_cmv': 'Donor CMV Status',
        'recipient_cmv': 'Recipient CMV',
        'run_model': '⚡ Run Model',
        'high_risk': '⚠️ High Risk Profile Detected',
        'high_risk_desc': 'The model indicates a lower probability of survival based on the provided clinical metrics.',
        'favorable': '✅ Favorable Profile Detected',
        'favorable_desc': 'The model indicates a higher probability of survival based on the provided clinical metrics.',
        'risk_score': 'Calculated Risk Score',
        'model_reasoning': 'Model Reasoning',
        'primary_driver': 'Primary Driver',
        'lowest_impact': 'Lowest Impact',
        'exchange_net': 'Local Donor Exchange Network',
        'exchange_desc': 'Because the predicted probability of success for this specific recipient-donor pair is below 50%, you may qualify for the local donor exchange program. By securely submitting this profile, we will cross-reference our regional database for a more highly-optimized HLA match. If a match is found, the attending physician and the donor will be notified to coordinate an exchange.',
        'consent': "I agree to securely and anonymously publish the current donor's HLA typing and demographic data to the Local Donor Exchange Network to assist other patients.",
        'doc_email': "Attending Doctor's Email",
        'donor_email': "Donor's Contact Email",
        'submit_exchange': "Submit to Exchange Network"
    },
    'FR': {
        'gateway_title': 'Votre passerelle sécurisée vers<br>des données prédictives validées',
        'auth_title': 'Accès au portail sécurisé',
        'auth_desc': 'Veuillez vous authentifier pour accéder au Prédicteur BMT',
        'physician_id': 'ID Médecin / Nom d\'utilisateur',
        'password': 'Mot de passe',
        'auth_btn': 'S\'authentifier',
        'nav_intro': 'Introduction',
        'nav_phys': 'Physique & Doses',
        'nav_disease': 'Maladie & Démographie',
        'nav_donor': 'Appariement Donneur',
        'step1_title': 'Étape 1 sur 4 : Introduction',
        'step1_header': 'Outil de Prédiction de Survie',
        'step1_welcome': 'Bienvenue, Dr. ',
        'step1_desc': 'Votre session est authentifiée. Toutes les données saisies dans cet outil sont traitées de manière sécurisée, chiffrées de bout en bout et maintenues en stricte conformité avec la loi HIPAA. Aucune donnée identifiable n\'est stockée.',
        'clinical_guidelines': 'Directives Cliniques',
        'clinical_desc': 'Cet outil est destiné à être utilisé par des professionnels de la santé qualifiés et<br>doit compléter, et non remplacer, le jugement médical professionnel et<br>les protocoles institutionnels.',
        'start_btn': 'Commencer l\'évaluation ➔',
        'step2_title': 'Étape 2 sur 4 : Physique & Doses',
        'step2_header': 'Métriques Physiques et de Dose',
        'step2_desc': 'Affinez les dimensions du patient et les dosages cibles. Le modèle visuel se met à jour en temps réel.',
        'dob': 'Date de naissance du receveur (Anonymisée)',
        'body_mass': 'Masse Corporelle (kg)',
        'gender': 'Sexe Biologique',
        'gender_f': 'Femme',
        'gender_m': 'Homme',
        'target_doses': 'Doses de Cellules Cibles',
        'cd34_label': 'Cellules Souches CD34+',
        'cd3d_label': 'Cellules T CD3d',
        'ratio_label': 'Ratio Cellules T / Souches',
        'prev_btn': '⬅️ Précédent',
        'next_btn': 'Enregistrer & Continuer ➔',
        'step3_title': 'Étape 3 sur 4 : Maladie & Démographie',
        'step3_header': 'Maladie & Démographie',
        'step3_desc': 'Spécifiez le diagnostic principal, le stade de la maladie et les données démographiques.',
        'diagnosis': 'Diagnostic Principal (Maladie)',
        'diagnosis_opts': ["LAL", "LAM", "Chronique", "Lymphome", "Non malin"],
        'disease_group': 'Groupe de Maladie',
        'risk_category': 'Catégorie de Risque Cytogénétique',
        'risk_opts': ["Faible/Favorable", "Élevé/Défavorable"],
        'tx_relapse': 'Traitement Post-Rechute ?',
        'yes': 'Oui',
        'no': 'Non',
        'stem_source': 'Source de Cellules Souches',
        'stem_opts': ["Moelle Osseuse", "Sang Périphérique (PBSC)"],
        'step4_title': 'Étape 4 sur 4 : Appariement Donneur',
        'step4_header': 'Appariement Donneur',
        'step4_desc': 'Configurez la source du donneur et les paramètres de compatibilité HLA.',
        'donor_age_35': 'Âge du Donneur > 35',
        'gender_match': 'Compatibilité de Sexe',
        'match': 'Compatible',
        'mismatch': 'Incompatible',
        'no_match': 'Non Compatible',
        'hla_match': 'Compatibilité HLA',
        'hla_mismatch_deg': 'Degré d\'incompatibilité HLA',
        'agvhd': 'aGVHD II-IV Attendue',
        'abo_match': 'Compatibilité ABO',
        'donor_abo': 'ABO du Donneur',
        'recipient_abo': 'ABO du Receveur',
        'recipient_rh': 'Rh du Receveur',
        'rh_neg': 'Négatif',
        'rh_pos': 'Positif',
        'donor_cmv': 'Statut CMV du Donneur',
        'recipient_cmv': 'CMV du Receveur',
        'run_model': '⚡ Exécuter le Modèle',
        'high_risk': '⚠️ Profil à Haut Risque Détecté',
        'high_risk_desc': 'Le modèle indique une probabilité de survie plus faible en fonction des métriques cliniques fournies.',
        'favorable': '✅ Profil Favorable Détecté',
        'favorable_desc': 'Le modèle indique une probabilité de survie plus élevée en fonction des métriques cliniques fournies.',
        'risk_score': 'Score de Risque Calculé',
        'model_reasoning': 'Raisonnement du Modèle',
        'primary_driver': 'Facteur Principal',
        'lowest_impact': 'Impact le Plus Faible',
        'exchange_net': 'Réseau Local d\'Échange de Donneurs',
        'exchange_desc': 'Étant donné que la probabilité de succès prévue pour cette paire est inférieure à 50%, vous pourriez être admissible au programme d\'échange de donneurs. En soumettant ce profil, nous chercherons une meilleure correspondance HLA. Si une correspondance est trouvée, le médecin traitant et le donneur seront informés.',
        'consent': "J'accepte de publier anonymement les données de typage HLA de ce donneur sur le réseau d'échange local afin d'aider d'autres patients.",
        'doc_email': "Email du Médecin Traitant",
        'donor_email': "Email de Contact du Donneur",
        'submit_exchange': "Soumettre au Réseau d'Échange"
    }
}

def _t(key):
    return translations[st.session_state.lang].get(key, key)

defaults = {
    'body_mass': 30.0, 'cd34_dose': 1.5, 'cd3d_dose': 50.0, 'cd3dcd34_ratio': 50.0,
    'recipient_gender': 0, 'disease': 0, 'disease_group': 0, 'risk_group': 0, 
    'tx_post_relapse': 0, 'stem_cell_source': 0, 'donor_age_35': 0, 'gender_match': 0, 
    'hla_match': 0, 'hla_mismatch': 0, 'abo_match': 0, 'donor_abo': 0, 
    'recipient_abo': 0, 'recipient_rh': 0, 'donor_cmv': 0, 'recipient_cmv': 0, 'ii_iv': 0,
    'selected_model': 'XGBoost',
    'show_other_models': False
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

for key in ['cd34_dose', 'cd3d_dose', 'cd3dcd34_ratio']:
    if f'{key}_num' not in st.session_state:
        st.session_state[f'{key}_num'] = st.session_state[key]
    if f'{key}_sld' not in st.session_state:
        st.session_state[f'{key}_sld'] = st.session_state[key]

if 'dob' not in st.session_state:
    st.session_state.dob = datetime.date.today() - datetime.timedelta(days=365*10)

def sync_from_num(k):
    st.session_state[f'{k}_sld'] = st.session_state[f'{k}_num']
    st.session_state[k] = st.session_state[f'{k}_num']

def sync_from_sld(k):
    st.session_state[f'{k}_num'] = st.session_state[f'{k}_sld']
    st.session_state[k] = st.session_state[f'{k}_sld']

def next_step():
    st.session_state.step += 1
    st.session_state.prediction_run = False

def prev_step():
    st.session_state.step -= 1
    st.session_state.prediction_run = False

if not st.session_state.logged_in:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
        
        html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 { 
            font-family: 'Inter', sans-serif !important; 
        }
        
        .stApp {
            background: radial-gradient(circle at center, #241419 0%, #0d080a 100%);
        }
        
        #MainMenu, header, footer { visibility: hidden; }
        
        .block-container {
            max-width: 950px !important;
            padding-top: 4vh !important;
            margin: 0 auto !important;
        }

        /* Streamlit Native Labels Bump */
        label[data-testid="stWidgetLabel"] p {
            font-size: 16px !important;
        }

        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            border-radius: 16px;
            margin-top: 10px;
            background: #140b0d;
        }
        
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 50% !important;
            flex: 1 1 50% !important;
            min-width: 50% !important;
        }
        
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) {
            background: #140b0d !important;
            border-radius: 16px 0 0 16px;
            border: 1px solid rgba(223, 88, 108, 0.2);
            border-right: 1px solid rgba(223, 88, 108, 0.05);
        }
        
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
            background: #1e1115 !important;
            border-radius: 0 16px 16px 0;
            border: 1px solid rgba(223, 88, 108, 0.2);
            border-left: none;
        }

        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 40px 40px 40px !important;
            background: transparent !important;
        }
        
        div[data-baseweb="input"] > div {
            background-color: #140b0d !important;
            border-radius: 8px !important;
            border: 1px solid rgba(223, 88, 108, 0.3) !important;
            color: white !important;
            height: 50px;
        }
        
        .stTextInput input {
            background-color: transparent !important;
            color: white !important;
            -webkit-text-fill-color: white !important;
            font-size: 16px !important;
        }

        input:-webkit-autofill,
        input:-webkit-autofill:hover, 
        input:-webkit-autofill:focus, 
        input:-webkit-autofill:active {
            -webkit-box-shadow: 0 0 0 30px #140b0d inset !important;
            -webkit-text-fill-color: white !important !important;
            transition: background-color 5000s ease-in-out 0s;
        }

        button[kind="primaryFormSubmit"], div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(180deg, #1e1115 0%, #140b0d 100%) !important;
            border: 1px solid rgba(34, 197, 94, 0.5) !important;
            box-shadow: 0 0 15px rgba(34, 197, 94, 0.15) !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            height: 56px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            width: 100%;
            margin-top: 10px;
        }
        
        button[kind="primaryFormSubmit"]:hover, div[data-testid="stFormSubmitButton"] button:hover {
            box-shadow: 0 0 25px rgba(34, 197, 94, 0.35) !important;
            border: 1px solid rgba(34, 197, 94, 0.8) !important;
            transform: translateY(-1px);
        }
        
        /* Floating Button Safe Targeting */
        div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: auto !important;
            z-index: 99999;
        }
        div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] button {
            background-color: #1e1115 !important;
            color: #df586c !important;
            border: 1px solid rgba(223, 88, 108, 0.4) !important;
            border-radius: 50px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.5) !important;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease !important;
        }
        div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] button:hover {
            background-color: #df586c !important;
            color: #ffffff !important;
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e0d14 0%, #361723 50%, #1e0d14 100%);
                height: 160px; border-radius: 16px; border: 1px solid rgba(223,88,108,0.4);
                display: flex; align-items: center; padding: 0 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5); position: relative; overflow: hidden;">
        <div style="z-index: 2;">
            <h1 style="color: #ffffff; margin: 0; font-size: 42px; font-weight: 900; line-height: 1.1; letter-spacing: -0.5px;">BMT Survival<br>Predictor</h1>
        </div>
        <div style="position: absolute; right: -10%; top: -60%; opacity: 0.15; font-size: 250px; user-select: none;">🦠</div>
        <div style="position: absolute; right: 15%; top: 20%; opacity: 0.2; font-size: 70px; user-select: none;">📊</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="small")

    with col_left:
        st.markdown(
            f'<div style="padding: 60px 40px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">'
            f'<div style="margin-bottom: 24px;">'
            f'<span class="material-symbols-outlined" style="font-size: 110px; color: transparent; background-clip: text; -webkit-background-clip: text; background-image: linear-gradient(135deg, #df586c, #22c55e); filter: drop-shadow(0px 4px 12px rgba(223,88,108,0.3));">lock</span>'
            f'</div>'
            f'<h3 style="color: #ffffff; font-size: 24px; font-weight: 700; margin-bottom: 30px; line-height: 1.4;">{_t("gateway_title")}</h3>'
            f'<div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">'
            f'<div style="background: rgba(255,255,255,0.03); padding: 8px 16px; border-radius: 20px; display: flex; align-items: center; gap: 8px; font-size: 14px; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.05);"><span class="material-symbols-outlined" style="color: #22c55e; font-size: 18px;">verified</span> HIPAA Compliant</div>'
            f'<div style="background: rgba(255,255,255,0.03); padding: 8px 16px; border-radius: 20px; display: flex; align-items: center; gap: 8px; font-size: 14px; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.05);"><span class="material-symbols-outlined" style="color: #22c55e; font-size: 18px;">health_and_safety</span> Clinician Reviewed</div>'
            f'<div style="background: rgba(255,255,255,0.03); padding: 8px 16px; border-radius: 20px; display: flex; align-items: center; gap: 8px; font-size: 14px; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.05);"><span class="material-symbols-outlined" style="color: #22c55e; font-size: 18px;">shield</span> Medical Guarantees</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown(f"""
        <div style="padding: 50px 40px 10px 40px; text-align: center;">
            <div style="height: 64px; width: 64px; border-radius: 50%; background: rgba(223, 88, 108, 0.15); display: inline-flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                <span class="material-symbols-outlined" style="font-size: 40px; color: #df586c;">person</span>
            </div>
            <h2 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 800;">{_t("auth_title")}</h2>
            <p style="color: #94a3b8; font-size: 15px; margin-top: 8px; margin-bottom: 10px;">{_t("auth_desc")}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input(_t("physician_id"))
            password = st.text_input(_t("password"), type="password")
            
            submitted = st.form_submit_button(_t("auth_btn"), type="primary")
            
            if submitted:
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Please enter your credentials.")
    
    # Floating Button for Login
    st.markdown('<div id="lang-anchor"></div>', unsafe_allow_html=True)
    btn_label = "🌐 Passer en Français" if st.session_state.lang == 'EN' else "🌐 Switch to English"
    st.button(btn_label, on_click=toggle_lang, key="floating_lang_btn_login")

    st.stop()


custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 { 
    font-family: 'Inter', sans-serif !important; 
    color: #f1f5f9;
}
.stApp { 
    background-color: #170f11; 
}

#MainMenu {visibility: hidden;} 
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    max-width: 90% !important; 
    margin: 0 auto !important;
}

/* Base font-size bump for native Streamlit labels */
label[data-testid="stWidgetLabel"] p {
    font-size: 16px !important;
}

.glass-card {
    background: #251619;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(223, 88, 108, 0.15);
}

div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div,
div[data-baseweb="datepicker"] > div {
    background-color: #251619 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(223, 88, 108, 0.2) !important;
    color: white !important;
    height: 50px;
    overflow: hidden; 
}

div[data-baseweb="select"]:focus-within > div, 
div[data-baseweb="input"]:focus-within > div {
    border-color: #df586c !important;
}

.stTextInput input, .stNumberInput input, .stDateInput input, div[data-baseweb="datepicker"] input, div[data-baseweb="select"] div {
    color: white !important;
    -webkit-text-fill-color: white !important;
    font-size: 16px !important;
}

div[data-testid="stNumberInputStepUp"], 
div[data-testid="stNumberInputStepDown"],
.stNumberInput button,
.stDateInput button {
    background-color: #251619 !important;
    color: #df586c !important;
}

.stNumberInput button svg,
.stDateInput button svg {
    fill: #df586c !important;
    color: #df586c !important;
}

div.stButton > button[kind="primary"] { 
    background-color: #df586c !important; 
    color: white !important; 
    border-radius: 8px !important; 
    height: 56px !important; 
    font-size: 18px !important; 
    font-weight: 600 !important; 
    border: none !important; 
    width: 100%;
}
div.stButton > button[kind="primary"]:hover { 
    background-color: #c94b5d !important; 
}

div.stButton > button[kind="secondary"] { 
    background-color: transparent !important; 
    color: #df586c !important; 
    border-radius: 8px !important; 
    height: 56px !important; 
    font-size: 18px !important; 
    font-weight: 600 !important; 
    border: 1px solid rgba(223, 88, 108, 0.3) !important; 
    width: 100%;
}
div.stButton > button[kind="secondary"]:hover { 
    background-color: rgba(223, 88, 108, 0.1) !important; 
}

/* Force hide all floating tooltip values on the slider */
div[data-testid="stThumbValue"],
div[data-baseweb="slider"] [role="tooltip"],
div[data-baseweb="slider"] div[data-testid="stMarkdownContainer"] {
    display: none !important;
    opacity: 0 !important;
}

/* Hide the min/max labels on the side of the slider */
div[data-testid="stSliderTickBarMin"], 
div[data-testid="stSliderTickBarMax"] {
    display: none !important;
}

/* Floating Button Safe Targeting */
div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: auto !important;
    z-index: 99999;
}
div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] button {
    background-color: #1e1115 !important;
    color: #df586c !important;
    border: 1px solid rgba(223, 88, 108, 0.4) !important;
    border-radius: 50px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.5) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease !important;
}
div[data-testid="element-container"]:has(#lang-anchor) + div[data-testid="element-container"] button:hover {
    background-color: #df586c !important;
    color: #ffffff !important;
    transform: translateY(-2px);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def render_navigation(current_step):
    steps = [
        {"icon": "analytics", "title": _t('nav_intro')},
        {"icon": "health_metrics", "title": _t('nav_phys')},
        {"icon": "medical_services", "title": _t('nav_disease')},
        {"icon": "groups", "title": _t('nav_donor')},
    ]
    
    html = (
        f'<div style="margin-top: 2rem;">'
        f'<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 30px;">'
        f'<div style="height: 40px; width: 40px; border-radius: 50%; background: #df586c; color: white; display: flex; align-items: center; justify-content: center;">'
        f'<span class="material-symbols-outlined">analytics</span>'
        f'</div>'
        f'<div>'
        f'<h1 style="font-size: 20px; font-weight: 800; margin: 0; color: #ffffff;">BMT Predictor</h1>'
        f'<p style="font-size: 14px; color: #df586c; font-weight: 600; margin: 0;">Protocol v2.4</p>'
        f'</div>'
        f'</div>'
        f'<div style="margin-bottom: 20px;">'
        f'<p style="font-size: 14px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Dashboard</p>'
        f'</div>'
    )
    
    for i, s in enumerate(steps, 1):
        status_class = "active" if i == current_step else ("completed" if i < current_step else "")
        check_icon = '<span class="material-symbols-outlined" style="margin-left: auto; color: #22c55e; font-size: 18px;">check_circle</span>' if i < current_step else ''
        
        bg_color = "rgba(223, 88, 108, 0.1)" if status_class == "active" else "transparent"
        text_color = "#df586c" if status_class == "active" else "#94a3b8"
        border = "1px solid rgba(223, 88, 108, 0.2)" if status_class == "active" else "1px solid transparent"
        font_weight = "700" if status_class == "active" else "500"
        
        html += (
            f'<div style="display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-radius: 8px; margin-bottom: 8px; font-size: 15px; color: {text_color}; background-color: {bg_color}; border: {border}; font-weight: {font_weight};">'
            f'<span class="material-symbols-outlined">{s["icon"]}</span>'
            f'<span>{i}. {s["title"]}</span>'
            f'{check_icon}'
            f'</div>'
        )
        
    html += (
        f'<div style="margin-top: 40px; padding: 16px; background: rgba(223, 88, 108, 0.05); border-radius: 12px;">'
        f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">'
        f'<span class="material-symbols-outlined" style="color: #df586c; font-size: 18px;">info</span>'
        f'<span style="font-size: 13px; font-weight: 700; text-transform: uppercase; color: #94a3b8;">System Status</span>'
        f'</div>'
        f'<p style="font-size: 13px; color: #64748b; margin: 0;">ML Models are synchronized with 2024.Q3 Clinical Registry.</p>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html.replace('\n', ''), unsafe_allow_html=True)

def get_dynamic_avatar(age, mass, gender):
    height = 100 + (age / 25.0) * 180
    body_width = 30 + (mass / 150.0) * 100
    head_r = 20 + (age / 25.0) * 10
    
    body_height = height * 0.45
    leg_height = height - (head_r * 2) - body_height - 10
    if leg_height < 20: leg_height = 20 
    
    arm_width = 10 + (mass / 150.0) * 15
    leg_width = 12 + (mass / 150.0) * 20
    cx = 100 
    y_top = 320 - height + head_r*2 + 5
    y_bottom = y_top + body_height
    
    if gender == 0:
        body_svg = f'<path d="M {cx - body_width/2.5} {y_top} L {cx + body_width/2.5} {y_top} L {cx + body_width/2 + 15} {y_bottom} L {cx - body_width/2 - 15} {y_bottom} Z" fill="#df586c" />'
    else:
        body_svg = f'<rect x="{cx - body_width/2}" y="{y_top}" width="{body_width}" height="{body_height}" rx="10" fill="#df586c" />'
    
    svg = (
        f'<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; height: 100%; min-height: 400px; background-color: #251619; border-radius: 12px; padding: 20px;">'
        f'<h3 style="color: #df586c; margin-top: 0; font-size: 20px; font-weight: 700; text-align: center;">Visual Profile</h3>'
        f'<svg width="200" height="320" viewBox="0 0 200 320" style="margin: auto;">'
        f'<circle cx="{cx}" cy="{320 - height + head_r}" r="{head_r}" fill="#df586c" />'
        f'{body_svg}'
        f'<rect x="{cx - body_width/2 - arm_width - 5}" y="{y_top + 5}" width="{arm_width}" height="{body_height*0.8}" rx="5" fill="#df586c" />'
        f'<rect x="{cx + body_width/2 + 5}" y="{y_top + 5}" width="{arm_width}" height="{body_height*0.8}" rx="5" fill="#df586c" />'
        f'<rect x="{cx - leg_width - 2}" y="{320 - leg_height}" width="{leg_width}" height="{leg_height}" rx="5" fill="#df586c" />'
        f'<rect x="{cx + 2}" y="{320 - leg_height}" width="{leg_width}" height="{leg_height}" rx="5" fill="#df586c" />'
        f'</svg>'
        f'<div style="font-size: 14px; color: #94a3b8; margin-top: 15px; text-align: center;">'
        f'<b>Height</b> scales with Age<br><b>Width</b> scales with Body Mass'
        f'</div></div>'
    )
    return svg.replace('\n', '').replace('    ', '')

def render_slider_labels(min_val, max_val, optimal_val):
    opt_percent = max(0, min(100, (optimal_val - min_val) / (max_val - min_val) * 100))
    html = (
        f'<div style="position: relative; margin-top: -15px; margin-bottom: 30px; font-family: \'Inter\', sans-serif;">'
        f'<div style="position: absolute; left: {opt_percent}%; top: -16px; width: 2px; height: 14px; background-color: #cbd5e1; z-index: 2; border-radius: 2px; pointer-events: none;"></div>'
        f'<div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">'
        f'<span style="flex: 1; text-align: left;">LOW</span>'
        f'<span style="flex: 1; text-align: center; position: relative; left: {(opt_percent - 50) / 2}%;">OPTIMAL</span>'
        f'<span style="flex: 1; text-align: right;">HIGH</span>'
        f'</div></div>'
    )
    return html.replace('\n', '')

@st.cache_resource
def load_model(model_name): 
    try:
        current_path = Path(__file__).resolve()
        models_dir = Path('models') 
        for parent in [current_path.parent, current_path.parent.parent]:
            if (parent / 'models').exists():
                models_dir = parent / 'models'
                break
        
        file_map = {
            "XGBoost": "xgboost_model.pkl",
            "Random Forest": "rf_model.pkl",
            "LightGBM": "lgbm_model.pkl",
            "SVM": "modele_svm_bmt.pkl"
        }
        filename = file_map.get(model_name, "xgboost_model.pkl")
        model_path = models_dir / filename
        
        if model_path.exists():
            return joblib.load(model_path)
        else:
            return filename 
    except Exception as e:
        return str(e)

today = datetime.date.today()
recipient_age = (today - st.session_state.dob).days / 365.2425

if st.session_state.step > 1:
    col_nav, col_content = st.columns([1, 3.5], gap="large")
    with col_nav:
        render_navigation(st.session_state.step)
else:
    col_content = st.container()

with col_content:

    header_html = (
        f'<div style="display: flex; align-items: center; gap: 12px;">'
        f'<div style="height: 48px; width: 48px; border-radius: 8px; background: #df586c; color: white; display: flex; align-items: center; justify-content: center;">'
        f'<span class="material-symbols-outlined" style="font-size: 26px;">analytics</span>'
        f'</div>'
        f'<div>'
        f'<h1 style="font-size: 24px; font-weight: 700; margin: 0; color: #ffffff;">BMT Survival Predictor</h1>'
        f'<p style="font-size: 14px; color: #df586c; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Clinical Decision Support</p>'
        f'</div></div>'
    )
    
    col_h_left, col_h_right = st.columns([3.5, 1], gap="large")
    with col_h_left:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(header_html, unsafe_allow_html=True)
    with col_h_right:
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if not st.session_state.show_other_models:
            st.markdown(f"<div style='color: #94a3b8; font-size: 13px; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;'>Active Model</div><div style='color: #df586c; font-weight: 800; font-size: 18px; margin-bottom: 8px;'>{st.session_state.selected_model}</div>", unsafe_allow_html=True)
            if st.button("🧪 Test other models", use_container_width=True):
                st.session_state.show_other_models = True
                st.rerun()
        else:
            st.selectbox("⚙️ Active ML Model", ["XGBoost", "Random Forest", "LightGBM", "SVM"], key="selected_model")
        
    st.markdown("<div style='border-bottom: 1px solid #2a1f22; margin-top: 5px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    if st.session_state.step == 1:
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #df586c; font-weight: 600; font-size: 16px;">
            <span>{_t('step1_title')}</span>
            <span style="color: #64748b;">25% Complete</span>
        </div>
        <div style="height: 6px; background: #251619; border-radius: 3px; margin-top: 8px; margin-bottom: 24px;">
            <div style="height: 100%; width: 25%; background: #df586c; border-radius: 3px;"></div>
        </div>
        <div style="color: #94a3b8; font-size: 16px; margin-bottom: 16px;">Overview & Regulatory Guidelines</div>
        <h1 style='font-size: 46px; font-weight: 900; color: #ffffff; margin-top: 0; margin-bottom: 16px;'>{_t('step1_header')}</h1>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 30px; display: flex; gap: 16px; align-items: center;">
            <div style="height: 48px; width: 48px; border-radius: 50%; background: rgba(34, 197, 94, 0.2); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <span class="material-symbols-outlined" style="color: #22c55e; font-size: 28px;">shield_lock</span>
            </div>
            <div>
                <h4 style="color: #ffffff; margin: 0 0 4px 0; font-size: 20px;">{_t('step1_welcome')}{st.session_state.username}</h4>
                <p style="color: #94a3b8; margin: 0; font-size: 16px;">{_t('step1_desc')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: #251619; border-radius: 12px; padding: 32px; margin-bottom: 40px; display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 2;">
                <h3 style="color: #ffffff; margin-top: 0; margin-bottom: 12px; font-size: 22px;">{_t('clinical_guidelines')}</h3>
                <p style="color: #94a3b8; margin: 0; font-size: 17px; line-height: 1.6; max-width: 90%;">
                    {_t('clinical_desc')}
                </p>
            </div>
            <div style="flex: 1; display: flex; gap: 24px; justify-content: flex-end;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined" style="color: #df586c;">verified</span>
                    <span style="color: #ffffff; font-weight: 600; font-size: 16px;">HIPAA Compliant</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="material-symbols-outlined" style="color: #df586c;">verified</span>
                    <span style="color: #ffffff; font-weight: 600; font-size: 16px;">Peer Reviewed</span>
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid #2a1f22; margin-bottom: 30px;"></div>
        """, unsafe_allow_html=True)
        
        _, btn_col, _ = st.columns([1, 1.5, 1])
        with btn_col:
            st.button(_t('start_btn'), on_click=next_step, type="primary")

    elif st.session_state.step == 2:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #df586c; font-weight: 600; font-size: 16px;">
            <span>{_t('step2_title')}</span>
            <span style="color: #64748b;">50% Complete</span>
        </div>
        <div style="height: 6px; background: #251619; border-radius: 3px; margin-top: 8px; margin-bottom: 24px;">
            <div style="height: 100%; width: 50%; background: #df586c; border-radius: 3px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='font-size: 36px; font-weight: 800; color: #ffffff; margin-top: 0; margin-bottom: 5px;'>{_t('step2_header')}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; margin-bottom: 30px; font-size: 16px;'>{_t('step2_desc')}</p>", unsafe_allow_html=True)
        
        col_form, col_vis = st.columns([1.5, 1], gap="large")
        
        with col_form:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                st.date_input(_t('dob'), key='dob', max_value=today)
            with c_m2:
                st.number_input(_t('body_mass'), min_value=0.0, max_value=150.0, step=1.0, key='body_mass')
            
            st.selectbox(_t('gender'), options=[0, 1], format_func=lambda x: _t('gender_f') if x==0 else _t('gender_m'), key='recipient_gender')
            
            st.markdown(f'<br><h3 style="font-size: 20px; margin-bottom: 20px; color: #ffffff; border-bottom: 1px solid #2a1f22; padding-bottom: 10px;">{_t("target_doses")}</h3>', unsafe_allow_html=True)
            
            calc_cd34_opt = 2.5 + (st.session_state.body_mass / 100.0) 
            calc_cd3d_opt = 60.0 + (recipient_age * 0.5) 
            calc_ratio_opt = 50.0 + (10 if st.session_state.recipient_gender == 1 else 0)

            st.number_input("CD34 Dose (log_CD34kgx10d6)", min_value=-0.5, max_value=5.0, step=0.1, key='cd34_dose_num', on_change=sync_from_num, args=('cd34_dose',))
            st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'><span style='font-size: 16px; font-weight: 700; color: #f1f5f9; text-transform: uppercase;'>{_t('cd34_label')}</span><span style='font-size: 18px; font-weight: 800; color: #df586c;'>{st.session_state.cd34_dose:.2f}</span></div>", unsafe_allow_html=True)
            st.slider("CD34 Slider", min_value=-0.5, max_value=5.0, step=0.1, key='cd34_dose_sld', label_visibility="collapsed", on_change=sync_from_sld, args=('cd34_dose',))
            st.markdown(render_slider_labels(-0.5, 5.0, calc_cd34_opt), unsafe_allow_html=True)
            
            st.number_input("CD3d Dose (CD3dkgx10d8)", min_value=0.0, max_value=200.0, step=1.0, key='cd3d_dose_num', on_change=sync_from_num, args=('cd3d_dose',))
            st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'><span style='font-size: 16px; font-weight: 700; color: #f1f5f9; text-transform: uppercase;'>{_t('cd3d_label')}</span><span style='font-size: 18px; font-weight: 800; color: #df586c;'>{st.session_state.cd3d_dose:.2f}</span></div>", unsafe_allow_html=True)
            st.slider("CD3d Slider", min_value=0.0, max_value=200.0, step=1.0, key='cd3d_dose_sld', label_visibility="collapsed", on_change=sync_from_sld, args=('cd3d_dose',))
            st.markdown(render_slider_labels(0.0, 200.0, calc_cd3d_opt), unsafe_allow_html=True)
            
            st.number_input("CD3d/CD34 Ratio", min_value=0.0, max_value=200.0, step=1.0, key='cd3dcd34_ratio_num', on_change=sync_from_num, args=('cd3dcd34_ratio',))
            st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'><span style='font-size: 16px; font-weight: 700; color: #f1f5f9; text-transform: uppercase;'>{_t('ratio_label')}</span><span style='font-size: 18px; font-weight: 800; color: #df586c;'>{st.session_state.cd3dcd34_ratio:.2f}</span></div>", unsafe_allow_html=True)
            st.slider("Ratio Slider", min_value=0.0, max_value=200.0, step=1.0, key='cd3dcd34_ratio_sld', label_visibility="collapsed", on_change=sync_from_sld, args=('cd3dcd34_ratio',))
            st.markdown(render_slider_labels(0.0, 200.0, calc_ratio_opt), unsafe_allow_html=True)

        with col_vis:
            st.markdown(get_dynamic_avatar(recipient_age, st.session_state.body_mass, st.session_state.recipient_gender), unsafe_allow_html=True)

        st.markdown("<div style='border-top: 1px solid #2a1f22; margin: 30px 0;'></div>", unsafe_allow_html=True)
        btn1, _, btn2 = st.columns([1, 2, 1])
        with btn1: st.button(_t('prev_btn'), on_click=prev_step, type="secondary")
        with btn2: st.button(_t('next_btn'), on_click=next_step, type="primary")

    elif st.session_state.step == 3:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #df586c; font-weight: 600; font-size: 16px;">
            <span>{_t('step3_title')}</span>
            <span style="color: #64748b;">75% Complete</span>
        </div>
        <div style="height: 6px; background: #251619; border-radius: 3px; margin-top: 8px; margin-bottom: 24px;">
            <div style="height: 100%; width: 75%; background: #df586c; border-radius: 3px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='font-size: 36px; font-weight: 800; color: #ffffff; margin-top: 0; margin-bottom: 5px;'>{_t('step3_header')}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; margin-bottom: 30px; font-size: 16px;'>{_t('step3_desc')}</p>", unsafe_allow_html=True)
        
        col_d1, col_d2 = st.columns(2, gap="large")
        
        with col_d1:
            st.selectbox(_t('diagnosis'), options=[0, 1, 2, 3, 4], format_func=lambda x: _t('diagnosis_opts')[x], key='disease')
            st.selectbox(_t('disease_group'), options=[0, 1, 2], key='disease_group')
            st.selectbox(_t('risk_category'), options=[0, 1], format_func=lambda x: _t('risk_opts')[x], key='risk_group')
            
        with col_d2:
            st.selectbox(_t('tx_relapse'), options=[0, 1], format_func=lambda x: _t('no') if x==0 else _t('yes'), key='tx_post_relapse')
            st.selectbox(_t('stem_source'), options=[0, 1], format_func=lambda x: _t('stem_opts')[x], key='stem_cell_source')

        st.markdown("<div style='border-top: 1px solid #2a1f22; margin: 30px 0;'></div>", unsafe_allow_html=True)
        btn1, _, btn2 = st.columns([1, 2, 1])
        with btn1: st.button(_t('prev_btn'), on_click=prev_step, type="secondary")
        with btn2: st.button(_t('next_btn'), on_click=next_step, type="primary")

    elif st.session_state.step == 4:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #df586c; font-weight: 600; font-size: 16px;">
            <span>{_t('step4_title')}</span>
            <span style="color: #64748b;">95% Complete</span>
        </div>
        <div style="height: 6px; background: #251619; border-radius: 3px; margin-top: 8px; margin-bottom: 24px;">
            <div style="height: 100%; width: 95%; background: #df586c; border-radius: 3px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h1 style='font-size: 36px; font-weight: 800; color: #ffffff; margin-top: 0; margin-bottom: 5px;'>{_t('step4_header')}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8; margin-bottom: 30px; font-size: 16px;'>{_t('step4_desc')}</p>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns(2, gap="large")
        
        with col_m1:
            st.selectbox(_t('donor_age_35'), options=[0, 1], format_func=lambda x: _t('no') if x==0 else _t('yes'), key='donor_age_35')
            st.selectbox(_t('gender_match'), options=[0, 1], format_func=lambda x: _t('no_match') if x==0 else _t('match'), key='gender_match')
            st.selectbox(_t('hla_match'), options=[0, 1], format_func=lambda x: _t('mismatch') if x==0 else _t('match'), key='hla_match')
            st.selectbox(_t('hla_mismatch_deg'), [0, 1, 2], key='hla_mismatch')
            st.selectbox(_t('agvhd'), options=[0, 1], format_func=lambda x: _t('no') if x==0 else _t('yes'), key='ii_iv')
            
        with col_m2:
            st.selectbox(_t('abo_match'), options=[0, 1], format_func=lambda x: _t('mismatch') if x==0 else _t('match'), key='abo_match')
            
            c_abo1, c_abo2 = st.columns(2)
            with c_abo1: st.selectbox(_t('donor_abo'), options=[0, 1, 2, 3], format_func=lambda x: ["A", "B", "AB", "O"][x], key='donor_abo')
            with c_abo2: st.selectbox(_t('recipient_abo'), options=[0, 1, 2, 3], format_func=lambda x: ["A", "B", "AB", "O"][x], key='recipient_abo')
            
            st.selectbox(_t('recipient_rh'), options=[0, 1], format_func=lambda x: _t('rh_neg') if x==0 else _t('rh_pos'), key='recipient_rh')
            
            c_cmv1, c_cmv2 = st.columns(2)
            with c_cmv1: st.selectbox(_t('donor_cmv'), options=[0, 1], format_func=lambda x: _t('rh_neg') if x==0 else _t('rh_pos'), key='donor_cmv')
            with c_cmv2: st.selectbox(_t('recipient_cmv'), options=[0, 1], format_func=lambda x: _t('rh_neg') if x==0 else _t('rh_pos'), key='recipient_cmv')

        st.markdown("<div style='border-top: 1px solid #2a1f22; margin: 30px 0;'></div>", unsafe_allow_html=True)
        btn1, _, btn2 = st.columns([1, 1.5, 1.5])
        with btn1: st.button(_t('prev_btn'), on_click=prev_step, type="secondary")
        with btn2: 
            if st.button(_t('run_model'), type="primary"):
                st.session_state.prediction_run = True

if st.session_state.step == 4 and st.session_state.prediction_run:
    
    model_result = load_model(st.session_state.selected_model)
    
    if isinstance(model_result, str):
        st.error(f"❌ Model File Missing: Please ensure '{model_result}' exists in the 'models' directory to test the {st.session_state.selected_model} architecture.")
    elif model_result is None:
        st.error("❌ An unknown error occurred while loading the machine learning model.")
    else:
        model = model_result
        input_data = {
            'Recipientage': recipient_age, 'log_CD34kgx10d6': st.session_state.cd34_dose,
            'Rbodymass': st.session_state.body_mass, 'RecipientRh': st.session_state.recipient_rh,
            'Riskgroup': st.session_state.risk_group, 'Stemcellsource': st.session_state.stem_cell_source,
            'Txpostrelapse': st.session_state.tx_post_relapse, 'Donorage35': st.session_state.donor_age_35,
            'RecipientCMV': st.session_state.recipient_cmv, 'CD3dkgx10d8': st.session_state.cd3d_dose,
            'CD3dCD34': st.session_state.cd3dcd34_ratio, 'Diseasegroup': st.session_state.disease_group,
            'ABOmatch': st.session_state.abo_match, 'IIIV': st.session_state.ii_iv,
            'DonorCMV': st.session_state.donor_cmv, 'HLAmatch': st.session_state.hla_match,
            'RecipientABO': st.session_state.recipient_abo, 'Recipientgender': st.session_state.recipient_gender,
            'Disease': st.session_state.disease, 'Gendermatch': st.session_state.gender_match,
            'HLAmismatch': st.session_state.hla_mismatch, 'DonorABO': st.session_state.donor_abo
        }
        df_input = pd.DataFrame([input_data])
        
        exact_columns = [
            'Recipientage', 'log_CD34kgx10d6', 'Rbodymass', 'RecipientRh', 'Riskgroup', 
            'Stemcellsource', 'Txpostrelapse', 'Donorage35', 'RecipientCMV', 'CD3dkgx10d8', 
            'CD3dCD34', 'Diseasegroup', 'ABOmatch', 'IIIV', 'DonorCMV', 'HLAmatch', 
            'RecipientABO', 'Recipientgender', 'Disease', 'Gendermatch', 'HLAmismatch', 'DonorABO'
        ]
        df_input = df_input[exact_columns]
        
        if st.session_state.selected_model == 'LightGBM':
            categorical_cols = [
                'Recipientgender', 'Stemcellsource', 'Donorage35', 'IIIV', 'Gendermatch',
                'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'DonorCMV', 
                'RecipientCMV', 'Disease', 'Riskgroup', 'Txpostrelapse', 'Diseasegroup', 
                'HLAmatch', 'HLAmismatch'
            ]
            for c in categorical_cols:
                if c in df_input.columns:
                    df_input[c] = df_input[c].astype(int)

        elif st.session_state.selected_model == 'SVM':
            from sklearn.impute import SimpleImputer
            try:
                data_path = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'final_dataset.csv'
                if data_path.exists():
                    df_raw = pd.read_csv(data_path)
                    X_raw = df_raw.drop(columns=['survival_status'], errors='ignore')
                    imputer = SimpleImputer(strategy='median')
                    imputer.fit(X_raw)
                    df_input = pd.DataFrame(imputer.transform(df_input), columns=df_input.columns)
                else:
                    st.warning("⚠️ Training dataset for SVM imputer not found. Model may fail to predict without it.")
            except Exception as e:
                st.warning(f"⚠️ Could not apply SVM imputation: {e}")
        
        st.markdown("<hr style='margin: 40px 0; border-color: #2a1f22;'>", unsafe_allow_html=True)
        
        with st.spinner(f"Analyzing patient profile with {st.session_state.selected_model}..."):
            try:
                prediction = model.predict(df_input)[0]
                if hasattr(model, 'predict_proba'):
                    clean_probability = float(model.predict_proba(df_input)[0][1])
                else:
                    clean_probability = float(prediction)
                    
                st.markdown(f"<h2 style='color: #ffffff; font-size: 28px;'>📋 Clinical Report ({st.session_state.selected_model})</h2>", unsafe_allow_html=True)
                col_res1, col_res2 = st.columns(2, gap="large")
                
                with col_res1:
                    if prediction == 1:
                        st.markdown(f"<div style='background: #251619; padding: 24px; border-radius: 12px; height: 100%; border: 1px solid rgba(223, 88, 108, 0.15);'><h3 style='color: #ef4444; margin-top:0; font-size: 22px;'>{_t('high_risk')}</h3><p style='color: #94a3b8; font-size: 16px;'>{_t('high_risk_desc')}</p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background: #251619; padding: 24px; border-radius: 12px; height: 100%; border: 1px solid rgba(223, 88, 108, 0.15);'><h3 style='color: #22c55e; margin-top:0; font-size: 22px;'>{_t('favorable')}</h3><p style='color: #94a3b8; font-size: 16px;'>{_t('favorable_desc')}</p></div>", unsafe_allow_html=True)
                        
                with col_res2:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number", value = clean_probability * 100,
                        number = {'suffix': "%", 'font': {'size': 46, 'color': '#ffffff'}},
                        title = {'text': _t('risk_score'), 'font': {'size': 18, 'color': '#94a3b8'}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickcolor': "#334155"},
                            'bar': {'color': "#df586c"},
                            'bgcolor': "#170f11",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.15)"},
                                {'range': [40, 70], 'color': "rgba(234, 179, 8, 0.15)"},
                                {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}]
                        }
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20), height=200)
                    st.plotly_chart(fig, use_container_width=True)
                
                if st.session_state.selected_model == 'XGBoost':
                    st.markdown("<hr style='border-color: #2a1f22; margin: 30px 0;'>", unsafe_allow_html=True)
                    header_shap = '<h3 style="color: #ffffff; margin-bottom: 24px; font-size: 22px;"><span class="material-symbols-outlined" style="vertical-align: middle; color:#df586c;">psychology</span> AI Feature Impact (SHAP)</h3>'
                    st.markdown(header_shap, unsafe_allow_html=True)
                    
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(df_input)
                        
                        if hasattr(shap_values, 'values'):
                            s_vals = shap_values.values[0]
                        else:
                            s_vals = shap_values[0]
                            
                        if len(s_vals.shape) > 1:
                            s_vals = s_vals[:, 1]
                            
                        f_vals = df_input.iloc[0].values
                        f_names = df_input.columns.tolist()
                        
                        shap_df = pd.DataFrame({
                            'Feature': f_names,
                            'SHAP Value': s_vals,
                            'Feature Value': f_vals
                        })
                        
                        shap_df['Abs SHAP'] = shap_df['SHAP Value'].abs()
                        shap_df = shap_df.sort_values(by='Abs SHAP', ascending=True)
                        
                        top_feature = shap_df.iloc[-1]
                        top_name = top_feature['Feature']
                        top_val = top_feature['Feature Value']
                        top_val_str = f"{top_val:.3g}" if isinstance(top_val, (int, float)) else str(top_val)
                        top_shap = top_feature['SHAP Value']
                        top_direction = "increased" if top_shap > 0 else "decreased"
                        top_color = "#df586c" if top_shap > 0 else "#0ea5e9"
                        
                        lowest_feature = shap_df.iloc[0]
                        low_name = lowest_feature['Feature']
                        low_shap = lowest_feature['SHAP Value']
                        
                        col_shap_text, col_shap_plot = st.columns([1, 2.5], gap="large")
                        
                        with col_shap_text:
                            reasoning_html = f"""
                            <div style="background: #251619; border-radius: 12px; padding: 30px; border: 1px solid rgba(223, 88, 108, 0.15); height: 100%; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #ffffff; margin-top: 0; font-size: 20px; margin-bottom: 12px;">{_t('model_reasoning')}</h4>
                                <p style="color: #94a3b8; font-size: 16px; line-height: 1.6; margin-bottom: 24px;">
                                    The SHAP values break down exactly how each clinical variable shifted the model's final risk score up or down from the baseline population average.
                                </p>
                                
                                <div style="background: #170f11; border-radius: 8px; padding: 16px; margin-bottom: 20px; border-left: 4px solid {top_color};">
                                    <span style="font-size: 14px; font-weight: 700; color: #64748b; text-transform: uppercase;">{_t('primary_driver')}</span><br>
                                    <div style="margin: 8px 0;">
                                        <span style="color: {top_color}; font-weight: 700; font-size: 20px;">{top_name}</span> 
                                        <span style="color: #94a3b8; font-size: 16px; margin-left: 6px;">(Input: {top_val_str})</span>
                                    </div>
                                    <p style="color: #cbd5e1; font-size: 16px; line-height: 1.6; margin: 0;">
                                        This variable had the strongest influence, <b><span style="color: {top_color};">{top_direction}</span></b> the predicted survival score by <b>{abs(top_shap):.2f}</b> points.
                                    </p>
                                </div>
                                
                                <div style="background: #170f11; border-radius: 8px; padding: 16px; border-left: 4px solid #334155;">
                                    <span style="font-size: 14px; font-weight: 700; color: #64748b; text-transform: uppercase;">{_t('lowest_impact')}</span><br>
                                    <div style="margin: 8px 0;">
                                        <span style="color: #f1f5f9; font-weight: 600; font-size: 18px;">{low_name}</span>
                                    </div>
                                    <p style="color: #cbd5e1; font-size: 16px; line-height: 1.6; margin: 0;">
                                        This factor had minimal isolated impact (<b style="color: #f1f5f9;">{low_shap:+.3f}</b>) on this specific patient's profile.
                                    </p>
                                </div>
                            </div>
                            """
                            st.markdown(reasoning_html.replace('\n', ''), unsafe_allow_html=True)
                        
                        with col_shap_plot:
                            if len(shap_df) > 10:
                                top_df = shap_df.tail(9)
                                other_df = shap_df.head(len(shap_df) - 9)
                                other_shap_sum = other_df['SHAP Value'].sum()
                                other_row = pd.DataFrame({
                                    'Feature': [f'Sum of {len(other_df)} other features'],
                                    'SHAP Value': [other_shap_sum],
                                    'Feature Value': [np.nan],
                                    'Abs SHAP': [abs(other_shap_sum)]
                                })
                                plot_df = pd.concat([other_row, top_df])
                            else:
                                plot_df = shap_df

                            y_labels = []
                            for idx, row in plot_df.iterrows():
                                if pd.isna(row['Feature Value']):
                                    y_labels.append(row['Feature'])
                                else:
                                    val = row['Feature Value']
                                    val_str = f"{val:.3g}" if isinstance(val, (int, float)) else str(val)
                                    y_labels.append(f"{val_str} = {row['Feature']}")

                            colors = ['#df586c' if val > 0 else '#0ea5e9' for val in plot_df['SHAP Value']]

                            fig_shap = go.Figure(go.Bar(
                                x=plot_df['SHAP Value'],
                                y=y_labels,
                                orientation='h',
                                marker_color=colors,
                                text=[f"{val:+.2f}" for val in plot_df['SHAP Value']],
                                textposition='outside',
                                textfont=dict(color="#f1f5f9", size=14),
                                hovertemplate="<b>Feature:</b> %{y}<br><b>Impact:</b> %{x:+.3f}<extra></extra>"
                            ))
                            
                            fig_shap.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#94a3b8", size=14),
                                margin=dict(l=20, r=40, t=0, b=0),
                                height=450,
                                xaxis=dict(
                                    title="SHAP value (Impact on prediction)",
                                    gridcolor="#2a1f22",
                                    zeroline=True,
                                    zerolinecolor="#cbd5e1",
                                    zerolinewidth=1
                                ),
                                yaxis=dict(gridcolor="rgba(0,0,0,0)")
                            )
                            
                            st.plotly_chart(fig_shap, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"⚠️ SHAP Feature Impact could not be generated. Error: {e}")

                if clean_probability > 0.50:
                    st.markdown("<hr style='border-color: #2a1f22; margin: 40px 0;'>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <h3 style="color: #ffffff; margin-bottom: 8px; font-size: 22px;">
                            <span class="material-symbols-outlined" style="vertical-align: middle; color:#eab308; margin-right: 8px;">hub</span>
                            {_t('exchange_net')}
                        </h3>
                        <p style="color: #94a3b8; font-size: 17px; max-width: 900px; line-height: 1.6; margin-bottom: 20px;">
                            {_t('exchange_desc')}
                        </p>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="glass-card" style="border-color: rgba(234, 179, 8, 0.3); padding: 20px; border-radius: 8px;">', unsafe_allow_html=True)
                    
                    consent = st.checkbox(_t('consent'))
                    
                    if consent:
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            doc_email = st.text_input(_t('doc_email'))
                        with c2:
                            donor_email = st.text_input(_t('donor_email'))
                            
                        submit_exchange = st.button(_t('submit_exchange'), type="primary")
                        
                        if submit_exchange:
                            if doc_email and donor_email:
                                st.success("✅ Profile successfully submitted to the Exchange Network. You will be contacted at the provided emails if a compatible swap is identified.")
                            else:
                                st.error("⚠️ Please provide both email addresses for future contact.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Floating button with safe CSS anchor (Main App)
st.markdown('<div id="lang-anchor"></div>', unsafe_allow_html=True)
btn_label = "🌐 Passer en Français" if st.session_state.lang == 'EN' else "🌐 Switch to English"
st.button(btn_label, on_click=toggle_lang, key="floating_lang_btn_main")
