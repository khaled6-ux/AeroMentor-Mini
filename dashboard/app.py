
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import AeroMentor modules
from src.xai import explain_decision
from src.flight_csv import analyze_flight_csv
from src.analyze_recording import analyze_recording_data


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# HELPER: BUILD FLIGHT PHASE TIMELINE
# --------------------------------------------------

def build_phase_segments(data):

    times = data["time_s"].tolist()
    phases = data["flight_phase"].tolist()

    if len(times) > 1:
        sample_interval = float(
            pd.Series(times).diff().dropna().median()
        )
    else:
        sample_interval = 1.0

    segments = []
    start = 0

    for index in range(1, len(phases) + 1):

        if (
            index == len(phases)
            or phases[index] != phases[start]
        ):

            end_time = (
                times[index]
                if index < len(times)
                else times[-1] + sample_interval
            )

            segments.append({
                "phase": phases[start],
                "start_s": times[start],
                "end_s": end_time,
                "duration_s": end_time - times[start],
            })

            start = index

    return pd.DataFrame(segments)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("✈️ AeroMentor Mini")

st.subheader(
    "Intelligent Flight Performance Analysis"
)

st.info(
    "Experimental aviation research prototype using "
    "synthetic and FlightGear simulated data. "
    "Not intended for real-world aviation decisions."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("AeroMentor")

st.sidebar.header("Analysis Mode")

analysis_mode = st.sidebar.radio(
    "Choose analysis mode",
    [
        "Manual Analysis",
        "CSV Flight Analysis",
        "Flight Recording Analysis",
    ],
)


# ==================================================
# MODE 1: FLIGHT RECORDING ANALYSIS
# ==================================================

if analysis_mode == "Flight Recording Analysis":

    st.header("Flight Recording Analysis")

    st.write(
        "Upload a FlightGear recording to analyze "
        "aircraft movement and flight characteristics."
    )

    st.sidebar.subheader("Reference Flight Conditions")

    target_altitude = st.sidebar.number_input(
        "Target altitude (ft)",
        min_value=0.0,
        value=3000.0,
        step=100.0,
        key="recording_target_altitude",
    )

    target_airspeed = st.sidebar.number_input(
        "Target airspeed (kt)",
        min_value=0.0,
        value=120.0,
        step=5.0,
        key="recording_target_airspeed",
    )

    target_heading = st.sidebar.number_input(
        "Target heading (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=90.0,
        step=5.0,
        key="recording_target_heading",
    )

    st.sidebar.caption(
        "Enter the intended reference conditions for this exercise. "
        "For meaningful results, use a recording of one exercise "
        "(such as level flight); a full flight with takeoff, turns "
        "and climbing cannot be fairly judged against one fixed target."
    )

    uploaded_file = st.file_uploader(
        "Upload FlightGear recording",
        type=["csv"],
        key="recording_upload",
    )

    if uploaded_file is None:

        st.info(
            "Upload a FlightGear CSV recording to begin."
        )

        st.stop()

    try:

        raw_data = pd.read_csv(uploaded_file)

        analyzed_data = analyze_recording_data(
            raw_data
        )

        recording_targets = {
            "altitude_ft": target_altitude,
            "airspeed_kt": target_airspeed,
            "heading_deg": target_heading,
        }

        recording_errors = analyze_flight_csv(
            raw_data,
            recording_targets,
        )

    except (
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:

        st.error(
            f"Unable to analyze recording: {error}"
        )

        st.stop()

    # ----------------------------------------------
    # FLIGHT OVERVIEW
    # ----------------------------------------------

    st.divider()

    st.subheader("Flight Overview")

    duration = (
        analyzed_data["time_s"].iloc[-1]
        - analyzed_data["time_s"].iloc[0]
    )

    altitude_change = (
        analyzed_data["altitude_ft"].iloc[-1]
        - analyzed_data["altitude_ft"].iloc[0]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Flight Duration",
            f"{duration:.1f} sec",
        )

    with col2:
        st.metric(
            "Maximum Altitude",
            f"{analyzed_data['altitude_ft'].max():.1f} ft",
        )

    with col3:
        st.metric(
            "Maximum Airspeed",
            f"{analyzed_data['airspeed_kt'].max():.1f} kt",
        )

    with col4:
        st.metric(
            "Altitude Change",
            f"{altitude_change:+.1f} ft",
        )

    # ----------------------------------------------
    # FLIGHT MEASUREMENTS
    # ----------------------------------------------

    st.divider()

    st.header("Flight Measurements")

    st.write(
        "Explore how aircraft measurements changed "
        "throughout the recorded flight."
    )

    charts = [
        (
            "altitude_ft",
            "Altitude",
            "Altitude (ft)",
        ),
        (
            "airspeed_kt",
            "Airspeed",
            "Airspeed (knots)",
        ),
        (
            "heading_deg",
            "Heading",
            "Heading (degrees)",
        ),
        (
            "vertical_speed_fpm",
            "Vertical Speed",
            "Vertical Speed (ft/min)",
        ),
        (
            "acceleration_kt_s",
            "Acceleration",
            "Acceleration (knots/sec)",
        ),
    ]

    for index in range(0, len(charts), 2):

        columns = st.columns(2)

        for column, chart in zip(
            columns,
            charts[index:index + 2],
        ):

            feature, title, label = chart

            with column:

                st.subheader(title)

                chart_data = analyzed_data[
                    ["time_s", feature]
                ].set_index("time_s")

                st.line_chart(
                    chart_data,
                    y_label=label,
                    x_label="Time (seconds)",
                    height=280,
                )

    # ----------------------------------------------
    # FLIGHT PHASE TIMELINE
    # ----------------------------------------------

    st.divider()

    st.header("Flight Phase Timeline")

    st.write(
        "See when different flight phases were "
        "detected during the recording."
    )

    phase_segments = build_phase_segments(
        analyzed_data
    )

    timeline_spec = {
        "mark": {
            "type": "bar",
            "cornerRadius": 3,
        },
        "encoding": {
            "x": {
                "field": "start_s",
                "type": "quantitative",
                "title": "Time (seconds)",
            },
            "x2": {
                "field": "end_s",
            },
            "y": {
                "field": "phase",
                "type": "nominal",
                "title": "Flight Phase",
            },
            "color": {
                "field": "phase",
                "type": "nominal",
                "title": "Phase",
            },
            "tooltip": [
                {
                    "field": "phase",
                    "type": "nominal",
                },
                {
                    "field": "start_s",
                    "type": "quantitative",
                    "title": "Start (s)",
                },
                {
                    "field": "end_s",
                    "type": "quantitative",
                    "title": "End (s)",
                },
                {
                    "field": "duration_s",
                    "type": "quantitative",
                    "title": "Duration (s)",
                },
            ],
        },
        "height": 220,
    }

    st.vega_lite_chart(
        phase_segments,
        timeline_spec,
        use_container_width=True,
    )

    # ----------------------------------------------
    # PHASE STATISTICS
    # ----------------------------------------------

    st.subheader("Flight Phase Statistics")

    phase_counts = (
        analyzed_data["flight_phase"]
        .value_counts()
        .rename_axis("Phase")
        .reset_index(name="Samples")
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.bar_chart(
            phase_counts.set_index("Phase"),
            y="Samples",
            horizontal=True,
        )

    with col2:

        st.dataframe(
            phase_counts,
            hide_index=True,
            use_container_width=True,
        )

    # ----------------------------------------------
    # ML CLASSIFICATION + EXPLAINABLE AI
    # ----------------------------------------------

    st.divider()

    st.header("Reference Deviation and AI Classification")

    st.write(
        "The following errors compare the recorded measurements "
        "with the reference conditions entered in the sidebar. "
        "The existing ML model then classifies these three error values."
    )

    st.write(
        f"**Reference targets:** "
        f"Altitude {target_altitude:.1f} ft | "
        f"Airspeed {target_airspeed:.1f} kt | "
        f"Heading {target_heading:.1f}°"
    )

    altitude_mae = recording_errors["altitude_mae_ft"]
    airspeed_mae = recording_errors["airspeed_mae_kt"]
    heading_mae = recording_errors["heading_mae_deg"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Altitude MAE", f"{altitude_mae:.2f} ft")

    with metric_col2:
        st.metric("Airspeed MAE", f"{airspeed_mae:.2f} kt")

    with metric_col3:
        st.metric("Heading MAE", f"{heading_mae:.2f}°")

    features = [
        "altitude_mae_ft",
        "airspeed_mae_kt",
        "heading_mae_deg",
    ]

    flight_features = pd.DataFrame(
        [{
            "altitude_mae_ft": altitude_mae,
            "airspeed_mae_kt": airspeed_mae,
            "heading_mae_deg": heading_mae,
        }],
        columns=features,
    )

    prediction = model.predict(flight_features)[0]

    st.subheader("Experimental ML Classification")

    if prediction == "stable":
        st.success("Flight classification: STABLE")
    else:
        st.error("Flight classification: UNSTABLE")

    st.subheader("Why did the model make this prediction?")

    decisions = explain_decision(model, flight_features)

    feature_descriptions = {
        "altitude_mae_ft": ("Altitude", "ft"),
        "airspeed_mae_kt": ("Airspeed", "knots"),
        "heading_mae_deg": ("Heading", "degrees"),
    }

    for decision in decisions:
        name, unit = feature_descriptions[decision["feature"]]

        operator_text = (
            "at or below"
            if decision["operator"] == "<="
            else "above"
        )

        st.write(
            f"- **{name}:** {decision['value']:.2f} {unit} "
            f"is {operator_text} the model's learned split "
            f"threshold of {decision['threshold']:.2f} {unit}."
        )

    st.caption(
        "This classification uses a model trained on synthetic examples. "
        "It is an experimental research result, not a validated aviation "
        "safety assessment or certified pilot-training recommendation."
    )

    # ----------------------------------------------
    # AUTOMATIC FLIGHT SUMMARY
    # ----------------------------------------------

    st.divider()

    st.header("Automatic Flight Summary")

    st.write(
        f"""
        - Total recorded samples: **{len(analyzed_data)}**
        - Flight duration: **{duration:.2f} seconds**
        - Maximum altitude: **{analyzed_data['altitude_ft'].max():.2f} ft**
        - Minimum altitude: **{analyzed_data['altitude_ft'].min():.2f} ft**
        - Maximum airspeed: **{analyzed_data['airspeed_kt'].max():.2f} knots**
        - Maximum vertical speed: **{analyzed_data['vertical_speed_fpm'].max():.2f} ft/min**
        - Minimum vertical speed: **{analyzed_data['vertical_speed_fpm'].min():.2f} ft/min**
        """
    )

    # ----------------------------------------------
    # DOWNLOAD ANALYSIS
    # ----------------------------------------------

    st.divider()

    csv_output = analyzed_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Flight Analysis CSV",
        data=csv_output,
        file_name="aeromentor_flight_analysis.csv",
        mime="text/csv",
    )

    st.caption(
        "Flight phases are identified using provisional "
        "measurement thresholds. These results are "
        "experimental and are not certified pilot guidance."
    )

    st.success("Flight recording analysis completed.")

    st.stop()


# ==================================================
# MODE 2: MANUAL ANALYSIS
# ==================================================

if analysis_mode == "Manual Analysis":

    st.sidebar.subheader("Flight Measurements")

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


# ==================================================
# MODE 3: CSV FLIGHT ANALYSIS
# ==================================================

else:

    st.sidebar.subheader("Flight Measurements")

    uploaded_file = st.sidebar.file_uploader(
        "Upload flight CSV",
        type=["csv"],
        key="mae_upload",
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

        st.info(
            "Upload a flight CSV to begin analysis."
        )

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

    except (
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:

        st.error(
            f"Unable to analyze CSV: {error}"
        )

        st.stop()


# ==================================================
# SHARED MANUAL / CSV ANALYSIS
# ==================================================

if analysis_mode == "CSV Flight Analysis":

    st.header("Recorded Flight Measurements")

    charts = [
        ("altitude_ft", "Altitude (ft)"),
        ("airspeed_kt", "Airspeed (knots)"),
        ("heading_deg", "Heading (degrees)"),
    ]

    for feature, label in charts:

        st.subheader(label)

        chart_data = flight_data[
            ["time_s", feature]
        ].set_index("time_s")

        st.line_chart(chart_data)


# --------------------------------------------------
# PERFORMANCE OVERVIEW
# --------------------------------------------------

st.divider()

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


# --------------------------------------------------
# PERFORMANCE MEASUREMENTS
# --------------------------------------------------

st.divider()

st.header("Performance Measurements")

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

    st.write("**Heading error (degrees)**")

    st.bar_chart(
        {"Heading": [heading_mae]},
        horizontal=True,
    )


# --------------------------------------------------
# MACHINE LEARNING CLASSIFICATION
# --------------------------------------------------

st.divider()

st.header("AI Flight Classification")

FEATURES = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]

flight = pd.DataFrame(
    [{
        "altitude_mae_ft": altitude_mae,
        "airspeed_mae_kt": airspeed_mae,
        "heading_mae_deg": heading_mae,
    }],
    columns=FEATURES,
)

prediction = model.predict(flight)[0]

if prediction == "stable":

    st.success(
        "Flight classification: STABLE"
    )

else:

    st.error(
        "Flight classification: UNSTABLE"
    )


# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------

st.subheader(
    "Why did the model make this prediction?"
)

decisions = explain_decision(
    model,
    flight,
)

FEATURE_DESCRIPTIONS = {
    "altitude_mae_ft": ("Altitude", "ft"),
    "airspeed_mae_kt": ("Airspeed", "knots"),
    "heading_mae_deg": ("Heading", "degrees"),
}

for decision in decisions:

    name, unit = FEATURE_DESCRIPTIONS[
        decision["feature"]
    ]

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
    "These classifications and explanations use a model "
    "trained on synthetic examples. They are not "
    "validated aviation safety assessments."
)

st.caption(
    "AeroMentor Mini | Synthetic-data research prototype"
)
