
import pandas as pd

from src.analyze_flight import calculate_mae


REQUIRED_COLUMNS = [
    "time_s",
    "altitude_ft",
    "airspeed_kt",
    "heading_deg",
]


def analyze_flight_csv(data, targets):
    """
    Analyze flight measurements from a DataFrame.

    Parameters:
        data: Flight measurements as a pandas DataFrame.
        targets: Dictionary of target flight values.

    Returns:
        Dictionary containing MAE features.
    """

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    if data.empty:
        raise ValueError("The flight CSV contains no rows.")

    # Convert measurements to numeric values
    data = data.copy()

    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Flight measurements contain missing values."
        )

    if not data[REQUIRED_COLUMNS].apply(
        lambda col: col.map(
            lambda value: pd.notna(value)
            and abs(value) != float("inf")
        )
    ).all().all():
        raise ValueError(
            "Flight measurements must be finite numbers."
        )

    results = {
        "altitude_mae_ft": calculate_mae(
            data["altitude_ft"].to_numpy(),
            targets["altitude_ft"],
        ),
        "airspeed_mae_kt": calculate_mae(
            data["airspeed_kt"].to_numpy(),
            targets["airspeed_kt"],
        ),
        "heading_mae_deg": calculate_mae(
            data["heading_deg"].to_numpy(),
            targets["heading_deg"],
            circular=True,
        ),
    }

    return results
