
import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.xai import explain_decision


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "flight_classifier.joblib"

FEATURES = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]


class TestExplainableAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_FILE)

    def make_flight(self, altitude, airspeed, heading):
        return pd.DataFrame([{
            "altitude_mae_ft": altitude,
            "airspeed_mae_kt": airspeed,
            "heading_mae_deg": heading,
        }], columns=FEATURES)

    def test_stable_flight_explanation(self):
        flight = self.make_flight(35, 2.5, 4)

        decisions = explain_decision(self.model, flight)

        self.assertEqual(
            self.model.predict(flight)[0], "stable"
        )
        self.assertGreater(len(decisions), 0)
        self.assertTrue(
            all(item["operator"] == "<=" for item in decisions)
        )

    def test_unstable_flight_explanation(self):
        flight = self.make_flight(35, 2.5, 15)

        decisions = explain_decision(self.model, flight)

        self.assertEqual(
            self.model.predict(flight)[0], "unstable"
        )
        self.assertTrue(
            any(item["operator"] == ">" for item in decisions)
        )


if __name__ == "__main__":
    unittest.main()
