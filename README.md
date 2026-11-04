# ✈️ AeroMentor Mini

### Intelligent Flight Performance Analysis Using Machine Learning and Explainable AI

AeroMentor Mini is an educational research prototype that analyzes simulated aircraft flight data, evaluates flight performance, and uses Machine Learning (ML) and Explainable AI (XAI) to provide interpretable feedback.

The system integrates FlightGear telemetry, Python-based flight analysis, a trained Decision Tree classifier, and an interactive Streamlit dashboard.

---

## 1. Project Objectives

The main objectives are:

- Collect aircraft telemetry from FlightGear.
- Analyze altitude, airspeed, and heading measurements.
- Calculate flight performance errors using Mean Absolute Error (MAE).
- Identify basic flight phases.
- Classify flight performance using Machine Learning.
- Explain model predictions using Explainable AI.
- Present results through an interactive dashboard.

## 2. System Architecture

The project follows this workflow:

1. **Flight Data Collection:** FlightGear provides simulated aircraft telemetry.
2. **Flight Data Analysis:** Python calculates flight measurements, vertical speed, acceleration, and flight phases.
3. **Machine Learning:** A Decision Tree classifier predicts stable or unstable flight performance.
4. **Explainable AI:** The model's decision path is translated into understandable explanations.
5. **Visualization:** Streamlit presents flight measurements, graphs, classifications, and explanations.

## 3. Main Features

### Flight Performance Analysis
- Altitude error calculation.
- Airspeed error calculation.
- Heading error calculation.
- Mean Absolute Error (MAE).

### Machine Learning
- Decision Tree classification.
- Synthetic flight training dataset.
- Training and testing split.
- Classification report and confusion matrix.
- Saved trained model.

### Explainable AI
- Decision path extraction.
- Learned decision threshold explanations.
- Human-readable flight feedback.

### FlightGear Integration
- Live telemetry connection.
- Flight recording to CSV.
- Altitude, airspeed, and heading collection.
- Recorded flight analysis.

### Flight Phase Detection
- Stationary.
- Accelerating.
- Decelerating.
- Moving.
- Turning.
- Climbing.
- Descending.

### Interactive Dashboard
- Manual flight analysis.
- CSV flight analysis.
- Flight recording analysis.
- Interactive graphs and statistics.
- Flight phase timeline.
- ML classification and XAI explanations.
- Downloadable analysis results.

## 4. Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- FlightGear
- Git and GitHub

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/khaled6-ux/AeroMentor-Mini.git
cd AeroMentor-Mini
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Project

### Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

### Train the Machine Learning model

```bash
python src/train_model.py
```

### Generate synthetic flight data

```bash
python src/generate_training_data.py
```

### Analyze a recorded flight

```bash
python src/analyze_recording.py
```

To analyze a new recording:

```bash
python src/analyze_recording.py --input data/flightgear_recording.csv --output data/flightgear_analysis.csv
```

### Record FlightGear telemetry

Start FlightGear with its telemetry property interface configured for localhost port 5500, then execute:

```bash
python src/flightgear_client.py
```

Press `CTRL+C` to stop recording.

## 7. Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

The current project includes 19 automated tests covering flight analysis, CSV validation, ML classification, recording analysis, and XAI explanations.

## 8. Project Limitations

- The ML training dataset consists of synthetic examples.
- FlightGear provides simulated rather than real aircraft data.
- Flight phase detection uses provisional measurement thresholds.
- Classification results have not been validated against real aviation training data.
- The system is an educational research prototype, not a certified flight training or aviation safety system.

## 9. Future Improvements

Potential research extensions include:

- Training with larger and more realistic flight datasets.
- Testing with additional flight scenarios.
- Improving classification generalization.
- Exploring additional Explainable AI techniques.
- Integrating more aircraft performance measurements.

## 10. Author and Repository

**Project:** AeroMentor Mini

**GitHub:** https://github.com/khaled6-ux/AeroMentor-Mini

**Purpose:** Undergraduate research and educational prototype exploring intelligent flight performance analysis using Machine Learning and Explainable AI.

---

*Built as an experimental research prototype. Not intended for real-world aviation decisions.*