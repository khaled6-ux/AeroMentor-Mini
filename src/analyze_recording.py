
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "flightgear_test_flight_01.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "flightgear_analysis_01.csv"
)

REQUIRED_COLUMNS = [
    "time_s",
    "altitude_ft",
    "airspeed_kt",
    "heading_deg",
]


def validate_recording(df):
    if df.empty:
        raise ValueError("The flight recording is empty.")

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(
            "The recording contains missing or invalid values."
        )

    if (df["time_s"].diff().dropna() <= 0).any():
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    if (df["airspeed_kt"] < 0).any():
        raise ValueError("Airspeed cannot be negative.")

    if not df["heading_deg"].between(0, 360).all():
        raise ValueError("Heading must be between 0 and 360.")

    return df


def classify_phase(speed, vertical_speed, turn_rate, acceleration):

    if speed < 1:
        return "stationary"

    if speed >= 20 and vertical_speed >= 150:
        return "climbing"

    if speed >= 20 and vertical_speed <= -150:
        return "descending"

    if speed >= 20 and abs(turn_rate) >= 1.5:
        return "turning"

    if speed >= 3 and acceleration >= 0.5:
        return "accelerating"

    if speed >= 3 and acceleration <= -0.5:
        return "decelerating"

    return "moving"


def analyze_recording_data(data):
    """
    Analyze flight recording data directly from a DataFrame.
    """

    df = validate_recording(data)

    dt = df["time_s"].diff()

    # Vertical speed: feet per minute
    df["vertical_speed_fpm"] = (
        df["altitude_ft"].diff() / dt * 60
    ).fillna(0)

    # Acceleration: knots per second
    df["acceleration_kt_s"] = (
        df["airspeed_kt"].diff() / dt
    ).fillna(0)

    # Correct heading changes across 0/360 degrees
    heading_change = (
        (df["heading_deg"].diff() + 180) % 360
    ) - 180

    df["turn_rate_deg_s"] = (
        heading_change / dt
    ).fillna(0)

    phases = []

    for speed, vertical, turn, acceleration in zip(
        df["airspeed_kt"],
        df["vertical_speed_fpm"],
        df["turn_rate_deg_s"],
        df["acceleration_kt_s"],
    ):
        phases.append(
            classify_phase(
                speed,
                vertical,
                turn,
                acceleration,
            )
        )

    df["flight_phase"] = phases

    return df


def analyze_recording(csv_path):
    """
    Preserve the original CSV file analysis interface.
    """

    data = pd.read_csv(csv_path)

    return analyze_recording_data(data)


def print_summary(df):

    duration = (
        df["time_s"].iloc[-1] - df["time_s"].iloc[0]
    )

    print("\nAEROMENTOR - FLIGHT RECORDING ANALYSIS")
    print("--------------------------------------")

    print(f"Total samples: {len(df)}")
    print(f"Duration: {duration:.2f} seconds")

    print("\nFLIGHT MEASUREMENTS")
    print("------------------")

    print(
        f"Maximum altitude: "
        f"{df['altitude_ft'].max():.2f} ft"
    )

    print(
        f"Minimum altitude: "
        f"{df['altitude_ft'].min():.2f} ft"
    )

    print(
        f"Altitude change: "
        f"{df['altitude_ft'].iloc[-1] - df['altitude_ft'].iloc[0]:.2f} ft"
    )

    print(
        f"Maximum airspeed: "
        f"{df['airspeed_kt'].max():.2f} knots"
    )

    print(
        f"Maximum vertical speed: "
        f"{df['vertical_speed_fpm'].max():.2f} ft/min"
    )

    print("\nFLIGHT PHASE SUMMARY")
    print("--------------------")

    for phase, count in df["flight_phase"].value_counts().items():
        print(f"{phase.capitalize()}: {count} samples")


def main():

    parser = argparse.ArgumentParser(
        description="Analyze recorded FlightGear telemetry."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input flight CSV file",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output analysis CSV file",
    )

    args = parser.parse_args()

    try:
        result = analyze_recording(args.input)

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        result.to_csv(output_path, index=False)

        print_summary(result)

        print("\nAnalysis completed successfully!")
        print(f"Saved analysis: {output_path}")

    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"\nAnalysis failed: {error}")


if __name__ == "__main__":
    main()
