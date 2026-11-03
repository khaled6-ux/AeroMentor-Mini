
from pathlib import Path

import joblib
import pandas as pd

from xai import explain_decision


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "flight_classifier.joblib"

FEATURES = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]

DESCRIPTIONS = {
    "altitude_mae_ft": ("altitude", "feet"),
    "airspeed_mae_kt": ("airspeed", "knots"),
    "heading_mae_deg": ("heading", "degrees"),
}


def main():
    # Load our locally trained model
    model = joblib.load(MODEL_FILE)

    # Experimental flight example
    flight = pd.DataFrame([{
        "altitude_mae_ft": 35,
        "airspeed_mae_kt": 2.5,
        "heading_mae_deg": 15,
    }], columns=FEATURES)

    prediction = model.predict(flight)[0]
    decisions = explain_decision(model, flight)

    print("AEROMENTOR MINI - EXPLAINABLE AI")
    print("--------------------------------")
    print(f"Prediction: {prediction}")

    print("\nDECISION PATH:")

    for item in decisions:
        print(
            f"- {item['feature']}: {item['value']:.2f} "
            f"{item['operator']} {item['threshold']:.2f}"
        )

    print("\nFLIGHT FEEDBACK")
    print("----------------")

    if prediction == "stable":
        print(
            "The model classified this flight as stable "
            "based on its learned decision rules."
        )
    else:
        print(
            "The model classified this flight as unstable."
        )

        for item in decisions:
            if item["operator"] == ">":
                description, unit = DESCRIPTIONS[item["feature"]]

                print(
                    f"The {description} deviation of "
                    f"{item['value']:.2f} {unit} exceeded "
                    f"a learned split threshold of "
                    f"{item['threshold']:.2f} {unit}."
                )

    print(
        "\nNote: Experimental feedback from synthetic data; "
        "not certified pilot guidance."
    )


if __name__ == "__main__":
    main()
