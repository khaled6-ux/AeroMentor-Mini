
import unittest
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "flight_classifier.joblib"

FEATURES = [
    "altitude_mae_ft",
    "airspeed_mae_kt",
    "heading_mae_deg",
]


class TestMLModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_FILE)

    def test_stable_flight(self):
        flight = pd.DataFrame(
            [[35, 2.5, 4]],
            columns=FEATURES,
        )

        prediction = self.model.predict(flight)

        self.assertEqual(prediction[0], "stable")

    def test_unstable_flight(self):
        flight = pd.DataFrame(
            [[160, 2.5, 4]],
            columns=FEATURES,
        )

        prediction = self.model.predict(flight)

        self.assertEqual(prediction[0], "unstable")


if __name__ == "__main__":
    unittest.main()
