from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow importing our existing XAI module
sys.path.insert(0, str(PROJECT_ROOT))

from xai import explain_decision
from src.flight_csv import analyze_flight_csv


# Configure the browser page
st.set_page_config(
    page_title="AeroMentor Mini",
    page_icon="✈️",
    layout="wide",
)


MODEL_FILE = PROJECT_ROOT / "models" / "flight_classifier.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()


# Dashboard header
st.title("✈️ AeroMentor Mini")
st.subheader("Intelligent Flight Performance Analysis")

st.info(
    "Experimental dashboard using synthetic flight data. "
    "Not intended for real-world aviation decisions."
)

st.divider()

# Sidebar controls
st.sidebar.header("Flight Measurements")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Manual Analysis", "CSV Flight Analysis"],
)



if analysis_mode == "Manual Analysis":

    altitude_mae = st.sidebar.number_input(
        "Altitude MAE (ft)",
        min_value=0.0,
        value=35.0,
        step=5.0,
    )

    airspeed_mae = st.sidebar.number_input(
        "Airspeed MAE (knots)",
        min_value=0.0,
        value=2.5,
        step=0.5,
    )

    heading_mae = st.sidebar.number_input(
        "Heading MAE (degrees)",
        min_value=0.0,
        max_value=180.0,
        value=15.0,
        step=1.0,
    )

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload flight CSV",
        type=["csv"],
    )

    st.sidebar.subheader("Target Flight Conditions")

    target_altitude = st.sidebar.number_input(
        "Target altitude (ft)",
        value=3000.0,
    )

    target_airspeed = st.sidebar.number_input(
        "Target airspeed (kt)",
        value=120.0,
    )

    target_heading = st.sidebar.number_input(
        "Target heading (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=90.0,
    )

    if uploaded_file is None:
        st.info("Upload a flight CSV to begin analysis.")
        st.stop()

    try:
        flight_data = pd.read_csv(uploaded_file)

        targets = {
            "altitude_ft": target_altitude,
            "airspeed_kt": target_airspeed,
            "heading_deg": target_heading,
        }

        results = analyze_flight_csv(
            flight_data,
            targets,
        )

        altitude_mae = results["altitude_mae_ft"]
        airspeed_mae = results["airspeed_mae_kt"]
        heading_mae = results["heading_mae_deg"]

    except (ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Unable to analyze CSV: {error}")
        st.stop()

if analysis_mode == "CSV Flight Analysis":

    st.header("Recorded Flight Measurements")

    st.write(
        "These graphs show how the recorded flight "
        "measurements changed over time."
    )

    charts = [
        ("altitude_ft", "Altitude (ft)"),
        ("airspeed_kt", "Airspeed (kt)"),
        ("heading_deg", "Heading (degrees)"),
    ]

    for column, label in charts:

        st.subheader(label)

        chart_data = flight_data[
            ["time_s", column]
        ].set_index("time_s")

        st.line_chart(chart_data)

# Display performance metrics
st.header("Flight Performance Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Altitude MAE",
        f"{altitude_mae:.1f} ft",
    )

with col2:
    st.metric(
        "Airspeed MAE",
        f"{airspeed_mae:.1f} kt",
    )

with col3:
    st.metric(
        "Heading MAE",
        f"{heading_mae:.1f}°",
    )

st.divider()

# Display each error measurement separately
st.header("Performance Measurements")

st.write(
    "The values below are mean absolute errors relative "
    "to target flight measurements."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Altitude error (ft)**")
    st.bar_chart(
        {"Altitude": [altitude_mae]},
        horizontal=True,
    )

with col2:
    st.write("**Airspeed error (kt)**")
    st.bar_chart(
        {"Airspeed": [airspeed_mae]},
        horizontal=True,
    )

with col3:
    st.write("**Heading error (°)**")
    st.bar_chart(
        {"Heading": [heading_mae]},
        horizontal=True,
    )

st.divider()


# Machine Learning prediction
st.header("AI Flight Classification")

FEATURES = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]

flight = pd.DataFrame([{
    "altitude_mae_ft": altitude_mae,
    "airspeed_mae_kt": airspeed_mae,
    "heading_mae_deg": heading_mae,
}], columns=FEATURES)

prediction = model.predict(flight)[0]

if prediction == "stable":
    st.success("Flight classification: STABLE")
else:
    st.error("Flight classification: UNSTABLE")

# Explainable AI
st.subheader("Why did the model make this prediction?")

decisions = explain_decision(model, flight)

FEATURE_DESCRIPTIONS = {
    "altitude_mae_ft": ("Altitude", "ft"),
    "airspeed_mae_kt": ("Airspeed", "knots"),
    "heading_mae_deg": ("Heading", "degrees"),
}

for decision in decisions:
    name, unit = FEATURE_DESCRIPTIONS[decision["feature"]]

    operator_text = (
        "at or below"
        if decision["operator"] == "<="
        else "above"
    )

    st.write(
        f"- **{name}:** {decision['value']:.2f} {unit} "
        f"is {operator_text} the model's learned "
        f"split threshold of "
        f"{decision['threshold']:.2f} {unit}."
    )

st.caption(
    "These classifications and explanations use a "
    "model trained on synthetic examples. They are "
    "not validated aviation safety assessments."
)


st.caption(
    "AeroMentor Mini | Synthetic-data research prototype"
)
