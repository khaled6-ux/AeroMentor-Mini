
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
RANDOM_SEED = 42
N_FLIGHTS = 500
SAMPLES_PER_FLIGHT = 60

TARGET_ALTITUDE = 5000
TARGET_AIRSPEED = 150
TARGET_HEADING = 90

TOLERANCES = {
    "altitude_mae_ft": 100,
    "airspeed_mae_kt": 5,
    "heading_mae_deg": 10,
}

rng = np.random.default_rng(RANDOM_SEED)


def calculate_mae(values, target):
    return float(np.mean(np.abs(values - target)))


def calculate_heading_mae(values, target):
    errors = (values - target + 180) % 360 - 180
    return float(np.mean(np.abs(errors)))


def generate_flight(flight_id, scenario):

    time = np.arange(SAMPLES_PER_FLIGHT)

    # Start with relatively controlled flight parameters
    altitude_bias = rng.uniform(-20, 20)
    altitude_amplitude = rng.uniform(5, 35)
    altitude_noise = rng.uniform(4, 10)

    airspeed_bias = rng.uniform(-0.8, 0.8)
    airspeed_amplitude = rng.uniform(0.2, 1.2)
    airspeed_noise = rng.uniform(0.3, 0.8)

    heading_bias = rng.uniform(-2, 2)
    heading_amplitude = rng.uniform(0.5, 3)
    heading_noise = rng.uniform(0.3, 0.8)

    # Introduce one deliberately large deviation
    # for unstable training examples.
    if scenario == "unstable":

        fault = rng.choice([
            "altitude",
            "airspeed",
            "heading"
        ])

        if fault == "altitude":
            altitude_bias = (
                rng.choice([-1, 1]) * rng.uniform(120, 170)
            )

        elif fault == "airspeed":
            airspeed_bias = (
                rng.choice([-1, 1]) * rng.uniform(7, 10)
            )

        else:
            heading_bias = (
                rng.choice([-1, 1]) * rng.uniform(13, 20)
            )

    # Generate artificial telemetry
    altitude = (
        TARGET_ALTITUDE
        + altitude_bias
        + altitude_amplitude * np.sin(time / 8)
        + rng.normal(0, altitude_noise, SAMPLES_PER_FLIGHT)
    )

    airspeed = (
        TARGET_AIRSPEED
        + airspeed_bias
        + airspeed_amplitude * np.sin(time / 8)
        + rng.normal(0, airspeed_noise, SAMPLES_PER_FLIGHT)
    )

    heading = (
        TARGET_HEADING
        + heading_bias
        + heading_amplitude * np.sin(time / 8)
        + rng.normal(0, heading_noise, SAMPLES_PER_FLIGHT)
    ) % 360

    # Extract flight performance features
    altitude_mae = calculate_mae(
        altitude, TARGET_ALTITUDE
    )

    airspeed_mae = calculate_mae(
        airspeed, TARGET_AIRSPEED
    )

    heading_mae = calculate_heading_mae(
        heading, TARGET_HEADING
    )

    # Assign the label using our experimental criteria
    stable = (
        altitude_mae <= TOLERANCES["altitude_mae_ft"]
        and airspeed_mae <= TOLERANCES["airspeed_mae_kt"]
        and heading_mae <= TOLERANCES["heading_mae_deg"]
    )

    return {
        "flight_id": flight_id,
        "altitude_mae_ft": altitude_mae,
        "airspeed_mae_kt": airspeed_mae,
        "heading_mae_deg": heading_mae,
        "label": "stable" if stable else "unstable",
    }


# Generate 250 controlled and 250 challenging flights
records = []

for i in range(N_FLIGHTS):

    scenario = (
        "stable" if i < N_FLIGHTS // 2 else "unstable"
    )

    records.append(generate_flight(i + 1, scenario))


# Create and shuffle the dataset
dataset = pd.DataFrame(records)

dataset = dataset.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# Save dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_path = PROJECT_ROOT / "data" / "flight_training_data.csv"

dataset.to_csv(output_path, index=False)


# Display results
print("AEROMENTOR - ML TRAINING DATASET")
print("--------------------------------")

print("\nFirst 10 flights:")
print(dataset.head(10).to_string(index=False))

print("\nClass distribution:")
print(dataset["label"].value_counts())

print(f"\nTotal flights: {len(dataset)}")
print(f"Dataset saved to: {output_path}")
