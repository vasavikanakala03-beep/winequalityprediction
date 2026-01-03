import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🍷 Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
RF_model = pickle.load(open("finalized_RFmodel.save", "rb"))
scaler = pickle.load(open("scaler_model.sav", "rb"))

# ---------------- FANCY WINE BACKGROUND + CSS ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1514361892635-eae31da8f3fd");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark overlay for readability */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.55);
    z-index: -1;
}

/* Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 25px;
    color: white;
}

/* Headings */
h1, h2, h3 {
    text-align: center;
    color: #f5f5f5;
    text-shadow: 2px 2px 10px black;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #8B0000, #B22222);
    color: white;
    font-size: 18px;
    border-radius: 14px;
    padding: 10px;
    width: 100%;
    border: none;
}

/* Slider text */
.stSlider label {
    font-weight: bold;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🍷 Wine Quality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predict the Quality of Red Wine using Machine Learning</h3>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🍇 Enter Wine Chemical Properties")

    fixed_acidity = st.slider("Fixed Acidity", 4.0, 15.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar (log)", 0.1, 2.0, 0.64)
    chlorides = st.slider("Chlorides (log)", 0.01, 1.0, 0.9)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (log)", 0.1, 2.0, 0.56)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 10.0, 300.0, 98.0)
    density = st.slider("Density", 0.98, 1.01, 1.0)
    ph = st.slider("pH", 2.5, 4.5, 3.0)
    sulphates = st.slider("Sulphates (log)", 0.1, 2.0, 0.68)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.5)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🍷 Predict Wine Quality"):
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, ph, sulphates, alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid',
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ])

    input_scaled = scaler.transform(input_data)
    prediction = int(np.round(RF_model.predict(input_scaled)[0]))

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    if prediction <= 4:
        st.error(f"🍋 Poor Quality Wine (Score: {prediction})")
    elif prediction == 5:
        st.warning(f"🍷 Average Quality Wine (Score: {prediction})")
    elif prediction == 6:
        st.success(f"🍷 Good Quality Wine (Score: {prediction})")
    else:
        st.success(f"🏆 Premium Wine Quality! (Score: {prediction})")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:white;'>Crafted with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
