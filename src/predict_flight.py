
from pathlib import Path

import joblib
import pandas as pd

# Locate the saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "flight_classifier.joblib"

# Load our trusted, locally trained model
model = joblib.load(MODEL_FILE)

# Create a new flight example
new_flight = pd.DataFrame([{
    "altitude_mae_ft": 160,
    "airspeed_mae_kt": 2.5,
    "heading_mae_deg": 4
}])

# Predict classification
prediction = model.predict(new_flight)

print("AEROMENTOR MINI - FLIGHT PREDICTION")
print("----------------------------------")
print(f"Predicted classification: {prediction[0]}")
