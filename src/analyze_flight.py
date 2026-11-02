
from pathlib import Path
import pandas as pd

# Find the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "synthetic_flight.csv"

# Load our flight data
flight_data = pd.read_csv(DATA_FILE)

# Target flight measurements
targets = {
    "Altitude": 5000,
    "Airspeed": 150,
    "Heading": 90
}

# Experimental tolerances (not aviation standards)
tolerances = {
    "Altitude": 100,
    "Airspeed": 5,
    "Heading": 10
}

print("\nAEROMENTOR MINI - FLIGHT ANALYSIS")
print("--------------------------------")

for measurement, target in targets.items():

    # Calculate errors
    if measurement == "Heading":
        errors = (
            (flight_data[measurement] - target + 180) % 360
        ) - 180
    else:
        errors = flight_data[measurement] - target

    # Mean Absolute Error
    mae = errors.abs().mean()

    # Compare with tolerance
    if mae <= tolerances[measurement]:
        feedback = "Within target tolerance"
    else:
        feedback = "Needs improvement"

    print(f"\n{measurement}")
    print(f"  Mean Absolute Error: {mae:.2f}")
    print(f"  Feedback: {feedback}")

print("\nAnalysis completed successfully!")
