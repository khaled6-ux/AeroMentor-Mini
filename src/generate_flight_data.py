
from pathlib import Path

import numpy as np
import pandas as pd

# Consistent random results
np.random.seed(42)

# Flight settings
time = np.arange(0, 121, 1)

target_altitude = 5000
target_airspeed = 150
target_heading = 90

# Synthetic flight measurements
altitude = (
    target_altitude
    + 200 * np.sin(time / 8)
    + np.random.normal(0, 15, len(time))
)

airspeed = (
    target_airspeed
    + 5 * np.sin(time / 12)
    + np.random.normal(0, 1.5, len(time))
)

heading = (
    target_heading
    + 20 * np.sin(time / 18)
    + np.random.normal(0, 2, len(time))
)

pitch = (
    2 + 1.5 * np.sin(time / 10)
    + np.random.normal(0, 0.3, len(time))
)

roll = (
    4 * np.sin(time / 8)
    + np.random.normal(0, 0.5, len(time))
)

# Create dataset
flight_data = pd.DataFrame({
    "Time": time,
    "Altitude": altitude,
    "Airspeed": airspeed,
    "Heading": heading,
    "Pitch": pitch,
    "Roll": roll
})

# Save CSV in the project's data directory
project_root = Path(__file__).resolve().parent.parent
output_path = project_root / "data" / "synthetic_flight.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)
flight_data.to_csv(output_path, index=False)

print("AeroMentor Mini - Flight Data Generator")
print("--------------------------------------")
print(flight_data.head())
print(f"\nSaved dataset to: {output_path}")
print(f"Total measurements: {len(flight_data)}")
