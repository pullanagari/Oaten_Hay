import io
import joblib
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from scipy.signal import savgol_filter
from streamlit_javascript import st_javascript

# ---------- Page Config ----------
st.set_page_config(
    page_title="Biomass AI", layout="wide", initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown(
    """
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
    margin-bottom: 1rem;
}

.upload-box {
    border: 2px dashed #cbd5e1;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    background-color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: DYNAMIC QR CODE GENERATOR ----------
with st.sidebar:
  st.header("QR Code Link")

  # 1. Dynamically fetch the current parent window URL using JavaScript
  detected_url = st_javascript(
      "await fetch('').then(r => window.parent.location.href)"
  )

  # Fallback: if JS execution is loading or returns 0/None, give a placeholder
  if not detected_url or detected_url == 0:
    app_url = "https://share.streamlit.io"
  else:
    app_url = str(detected_url)

  # 2. Allow user to confirm or overwrite the address
  user_input = st.text_input("QR Code URL Target:", value=app_url)

  if user_input:
    # Generate QR code targeting the app url
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(user_input)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.image(byte_im, caption="Scan to visit this app", use_container_width=True)

    st.download_button(
        label="Download QR Code",
        data=byte_im,
        file_name="app_qrcode.png",
        mime="image/png",
    )

# ---------- MAIN HERO SECTION ----------
st.markdown(
    """
<div class="hero">
    <h1>🌾 Hyperspectral Estimation of Biomass and Crude Protein in Oaten Hay</h1>
    <p>Transform hyperspectral reflectance data of Oaten hay canopy into quantitative estimates of above-ground biomass (t ha⁻¹) and crude protein (CP, %). 
    This application enables users to upload hyperspectral data spanning in the spectral range 374–2500 nm at 1 nm resolution. 
    A machine learning framework based on Random Forest regression is employed to model the relationship between spectral features and target variables, providing robust predictions of both biomass and crude protein.
</p>
    <p><b>374–2500 nm | 1 nm resolution </b></p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ---------- FEATURES ----------
col1, col2, col3 = st.columns(3)

with col1:
  st.markdown(
      """
    <div class="card">
        <h3>📊 Upload Data</h3>
        <p>Upload hyperspectral CSV datasets and start analysis instantly.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

with col2:
  st.markdown(
      """
    <div class="card">
        <h3>⚙️ Processing</h3>
        <p>Processing and Prediction pipeline.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

with col3:
  st.markdown(
      """
    <div class="card">
        <h3>📈 Instant Results</h3>
        <p>Get biomass predictions (tonnes/ha) and download results in seconds.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.write("")


# ---------- MODEL LOADING ----------
@st.cache_resource
def load_models():
  biomass_model = joblib.load("biomass_model.pkl")
  cp_model = joblib.load("CP_model.pkl")
  return biomass_model, cp_model


biomass_model, cp_model = load_models()

# ---------- UPLOAD SECTION ----------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload Hyperspectral CSV", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PROCESS Pipeline ----------
if uploaded_file:
  df = pd.read_csv(uploaded_file)

  st.subheader("🔍 Data Preview")
  st.dataframe(df.head(), use_container_width=True)

  # Select spectral columns
  start_col = st.number_input("Start Column Index", 0, len(df.columns) - 1, 0)
  end_col = st.number_input(
      "End Column Index", 1, len(df.columns), len(df.columns)
  )

  X = df.iloc[:, start_col:end_col]

  # Preprocessing (Savitzky-Golay Filter)
  X_smooth = savgol_filter(X, 9, 3, deriv=1, mode="nearest")
  X_smooth = pd.DataFrame(X_smooth)

  # Prediction Engine
  biomass_pred = biomass_model.predict(X_smooth)
  cp_pred = cp_model.predict(X_smooth)
  df["Predicted_Biomass"] = biomass_pred
  df["Predicted_CP"] = cp_pred

  # ---------- RESULTS DISPLAY ----------
  st.subheader("🌱 Predictions")

  # --- Summary metrics ---
  m_col1, m_col2 = st.columns(2)
  with m_col1:
    st.metric("Mean Biomass (t/ha)", round(df["Predicted_Biomass"].mean(), 2))
  with m_col2:
    st.metric("Mean Crude Protein (%)", round(df["Predicted_CP"].mean(), 2))

  # --- Charts ---
  c_col1, c_col2 = st.columns(2)
  with c_col1:
    st.write("### Biomass (t/ha)")
    st.line_chart(df["Predicted_Biomass"])
  with c_col2:
    st.write("### Crude Protein (%)")
    st.line_chart(df["Predicted_CP"])

  # --- Data Table View ---
  st.write("### Prediction Table")
  st.dataframe(
      df[["Predicted_Biomass", "Predicted_CP"]], use_container_width=True
  )

  # --- File Downloader ---
  csv = df.to_csv(index=False).encode("utf-8")
  st.download_button(
      "⬇️ Download Results", csv, "predictions.csv", mime="text/csv"
  )
