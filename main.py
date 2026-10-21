# import streamlit as st
# import numpy as np
# import joblib

# st.set_page_config(
#     page_title="Travel Package Prediction",
#     layout="centered"
# )

# st.title("Travel Package Prediction")

# loaded_model=joblib.load("model.joblib")
# loaded_preprocessor=joblib.load("preprocessor.joblib")


import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Travel Package Purchase Prediction - Streamlit Frontend
# Based on the features and preprocessing in project.ipynb
# ---------------------------------------------------------

st.set_page_config(
    page_title="Travel Package Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("model.joblib")
PREPROCESSOR_PATH = Path("preprocessor.joblib")
DATA_PATH = Path("Travel.csv")

# Custom styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f8ff 0%, #eef5ff 55%, #f8fbff 100%);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #102a56, #176b87);
        color: white;
        margin-bottom: 1.3rem;
    }
    .hero h1 { color: white; margin-bottom: .35rem; }
    .hero p { color: #e6f4ff; font-size: 1.05rem; margin-bottom: 0; }
    .metric-card {
        background: white;
        border: 1px solid #dbe7f5;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 3px 14px rgba(24, 55, 90, .06);
    }
    .section-note { color: #52657d; font-size: .92rem; }
    div.stButton > button[kind="primary"] {
        width: 100%; border-radius: 10px; font-weight: 700; min-height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ✈️ Travel Insights")
    st.caption("Travel Package Purchase Prediction")
    st.markdown("---")
    st.markdown("### About this app")
    st.write(
        "Enter a customer's profile and travel preferences to estimate "
        "whether they are likely to purchase the offered travel package."
    )
    st.markdown("---")
    st.caption("Model: trained classifier from your project notebook")
    st.caption("Preprocessing: saved ColumnTransformer")

st.markdown(
    """
    <div class="hero">
      <h1>✈️ Travel Package Predictor</h1>
      <p>Understand customer intent and estimate the likelihood of a travel package purchase.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load saved objects. In the notebook, model.joblib is saved as a dictionary
# containing a model under the key "Xg boost".
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.joblib was not found. Export the trained model from your notebook "
            "and place model.joblib in the same folder as main.py."
        )
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "preprocessor.joblib was not found. Export the fitted preprocessor from "
            "your notebook and place preprocessor.joblib in the same folder as main.py."
        )

    saved_model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Support either a single estimator or the dictionary saved by the notebook.