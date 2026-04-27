import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter

st.set_page_config(page_title="Biomass Prediction App", layout="wide")

st.title("🌾 Hyperspectral Biomass Prediction")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("biomass_model.pkl")

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("Upload hyperspectral CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Preview Data")
    st.dataframe(df.head())

    # Assume spectral columns start from index 0 or user selects
    st.write("### Select spectral range")
    start_col = st.number_input("Start column index", 0, len(df.columns)-1, 0)
    end_col = st.number_input("End column index", 1, len(df.columns), len(df.columns))

    X = df.iloc[:, start_col:end_col]

    # Preprocessing
    X_smooth = savgol_filter(X, 9, 3, deriv=1, mode='nearest')
    X_smooth = pd.DataFrame(X_smooth)

    # Predict
    predictions = model.predict(X_smooth)

    df['Predicted_Biomass'] = predictions

    st.write("### Predictions")
    st.dataframe(df[['Predicted_Biomass']])

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
