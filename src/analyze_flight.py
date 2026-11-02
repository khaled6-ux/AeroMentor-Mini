
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "synthetic_flight.csv"

TARGETS = {
    "Altitude": 5000,
    "Airspeed": 150,
    "Heading": 90,
}

TOLERANCES = {
    "Altitude": 100,
    "Airspeed": 5,
    "Heading": 10,
}


def calculate_mae(values, target, circular=False):
    values = pd.Series(values, dtype=float)

    if values.empty or values.isna().any():
        raise ValueError("Measurements must be non-empty and valid.")

    errors = values - target

    if circular:
        errors = (errors + 180) % 360 - 180

    return float(errors.abs().mean())


def main():
    flight_data = pd.read_csv(DATA_FILE)

    print("\nAEROMENTOR MINI - FLIGHT ANALYSIS")
    print("--------------------------------")

    for measurement, target in TARGETS.items():
        mae = calculate_mae(
            flight_data[measurement],
            target,
            circular=(measurement == "Heading"),
        )

        feedback = (
            "Within target tolerance"
            if mae <= TOLERANCES[measurement]
            else "Needs improvement"
        )

        print(f"\n{measurement}")
        print(f"  Mean Absolute Error: {mae:.2f}")
        print(f"  Feedback: {feedback}")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
