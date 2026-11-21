
# source .venv/bin/activate
# python -m streamlit run app.py --server.port 8502 --server.address 127.0.0.1

# ============================================================
# 1. Import Libraries
# ============================================================

import streamlit as st
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================================
# 2. Configure App
# ============================================================

st.set_page_config(
    page_title="Football Free-Kick Predictor",
    page_icon="⚽",
    layout="centered"
)



# ============================================================
# CUSTOM APP STYLING
# ============================================================

st.markdown("""
<style>

/* ========================================
   1. ALL TEXT SIZES
   ======================================== */

/* 1.1 Main App Title */
h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
}

/* 1.2 Section Headings */
h2, h3 {
    font-size: 30px !important;
    font-weight: 700 !important;
}

/* 1.3 Main Description */
div[data-testid="stMain"] .stMarkdown p {
    font-size: 18px;
}

/* 1.4 Slider Parameter Labels */
div[data-testid="stSlider"]
label[data-testid="stWidgetLabel"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* 1.5 Slider Numerical Values */
div[data-testid="stSlider"]
[data-rac][style*="position: absolute"] span {
    font-size: 18px !important;
}


/* ========================================
   2. SLIDER STYLING
   ======================================== */

/* 2.1 Slider Track Thickness */
div[data-testid="stSlider"]
div[data-orientation="horizontal"] > div:first-child {
    height: 10px !important;
    border-radius: 6px !important;
}

/* 2.2 Slider Colour */
div[data-testid="stSlider"] {
    accent-color: #FACC15;
}

/* 2.3 Slider Handle Size */
div[data-testid="stSlider"]
div[data-rac][style*="position: absolute"] {
    width: 20px !important;
    height: 20px !important;
}

/* 2.4 Slider Handle Colour */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #FACC15 !important;
    border-color: #FACC15 !important;
}

/* Note:
   Filled slider colour is controlled primarily
   by primaryColor in config.toml.
   No background overrides on inner slider divs.
*/


/* ========================================
   3. PREDICT BUTTON STYLING
   ======================================== */

/* 3.1 Button Size and Shape */
div[data-testid="stButton"] button {
    min-height: 65px !important;
    padding: 14px 30px !important;
    border-radius: 14px !important;
}

/* 3.2 Button Background Colour */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #FACC15 !important;
    border: none !important;
}

/* 3.3 Button Text */
div[data-testid="stButton"] button p {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* 3.4 Button Hover Colour */
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #EAB308 !important;
}


/* ========================================
   4. PREDICTION RESULTS STYLING
   ======================================== */

/* 4.1 Result Heading */
div[data-testid="stMain"] h3 {
    font-size: 30px !important;
}

/* 4.2 Metric Labels */
div[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}

/* 4.3 Metric Values */
div[data-testid="stMetricValue"] {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #FACC15 !important;
}

/* 4.4 Goal / Miss Message */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    padding: 16px !important;
}

/* 4.5 Goal / Miss Text */
div[data-testid="stAlert"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


st.title("⚽ Football Free-Kick Predictor")

st.write(
    "Predict whether a free kick will score and estimate "
    "its goal-crossing time using neural networks."
)


# ============================================================
# 3. User Inputs
# ============================================================

st.subheader("Free-Kick Parameters")

speed = st.slider(
    "Initial Speed (m/s)",
    min_value=25.0,
    max_value=32.0,
    value=28.0,
    step=0.1
)

elevation = st.slider(
    "Elevation Angle (°)",
    min_value=14.0,
    max_value=19.0,
    value=15.8,
    step=0.1
)

spin = st.slider(
    "Spin Rate (RPM)",
    min_value=0,
    max_value=350,
    value=150,
    step=5
)



# ============================================================
# 4. Load Trained Neural Networks
# ============================================================

folder = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    outcome_model = load_model(folder / "outcome_model.keras", compile=False)
    time_model = load_model(folder / "time_model.keras", compile=False)

    return outcome_model, time_model

outcome_model, time_model = load_models()


# ============================================================
# 5. Generate Predictions
# ============================================================

if st.button("Predict Free Kick", type="primary"):

    # Prepare input features
    new_shot = np.array(
        [[speed, elevation, spin]],
        dtype=np.float32
    )

    # Predict scoring outcome
    probability = float(
        outcome_model.predict(new_shot, verbose=0)[0][0]
    )

    # Display results
    st.subheader("Prediction Results")
    st.metric("Model Scoring Output", f"{probability * 100:.1f}%")

    if probability >= 0.5:
        st.success("⚽ GOAL!")

        # Predict travel time for successful shots
        predicted_time = float(
            time_model.predict(new_shot, verbose=0)[0][0]
        )

        st.metric("Predicted Goal-Crossing Time", f"{predicted_time:.3f} s")

    else:
        st.error("❌ MISS!")
        st.info("Travel time is not predicted for missed shots.")

