import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter

# ---------- Page Config ----------
st.set_page_config(
    page_title="Biomass AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f7f9fb;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    padding: 2rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #2c7be5, #00b894);
    color: white;
}

.card {
    padding: 1.5rem;
    border-radius: 12px;
    background: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

.upload-box {
    border: 2px dashed #cbd5e1;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown("""
<div class="hero">
    <h1>🌾 Hyperspectral Biomass Prediction</h1>
    <p>Turn hyperspectral data into actionable biomass insights. This application allows users to upload hyperspectral datasets covering the spectral range **374 nm to 2500 nm (1 nm resolution)**.

A **Random Forest regression model** is used to predict biomass based on spectral reflectance features extracted from the hyperspectral data.
</p>
    <p><b>374–2500 nm | 1 nm resolution | Random Forest Model</b></p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------- FEATURES ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📊 Upload Data</h3>
        <p>Upload hyperspectral CSV datasets and start analysis instantly.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>⚙️ Processing</h3>
        <p>Processing and Prediction pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>📈 Instant Results</h3>
        <p>Get biomass predictions and download results in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("biomass_model.pkl")

model = load_model()

# ---------- UPLOAD SECTION ----------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Hyperspectral CSV", type=["csv"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PROCESS ----------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Select spectral columns
    start_col = st.number_input("Start Column", 0, len(df.columns)-1, 0)
    end_col = st.number_input("End Column", 1, len(df.columns), len(df.columns))

    X = df.iloc[:, start_col:end_col]

    # Preprocessing
    X_smooth = savgol_filter(X, 9, 3, deriv=1, mode='nearest')
    X_smooth = pd.DataFrame(X_smooth)

    # Prediction
    predictions = model.predict(X_smooth)
    df['Predicted_Biomass'] = predictions

    # ---------- RESULTS ----------
    st.subheader("🌱 Biomass Predictions")
    st.dataframe(df[['Predicted_Biomass']], use_container_width=True)

    # Simple chart
    st.line_chart(df['Predicted_Biomass'])

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv, "predictions.csv")
