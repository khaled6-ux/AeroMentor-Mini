
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.analyze_recording import (
    analyze_recording,
    validate_recording,
    classify_phase,
)


class TestFlightRecordingAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            "time_s": [0, 1, 2, 3],
            "altitude_ft": [100, 110, 120, 110],
            "airspeed_kt": [0, 10, 20, 20],
            "heading_deg": [359, 0, 1, 2],
        })

    def analyze_data(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test_flight.csv"
            data.to_csv(path, index=False)
            return analyze_recording(path)

    def test_vertical_speed_calculation(self):
        result = self.analyze_data(self.data)

        self.assertAlmostEqual(
            result["vertical_speed_fpm"].iloc[1],
            600.0
        )

        self.assertAlmostEqual(
            result["vertical_speed_fpm"].iloc[3],
            -600.0
        )

    def test_acceleration_calculation(self):
        result = self.analyze_data(self.data)

        self.assertAlmostEqual(
            result["acceleration_kt_s"].iloc[1],
            10.0
        )

    def test_heading_wraparound(self):
        result = self.analyze_data(self.data)

        # Heading changes from 359 to 0 degrees.
        # The actual change is +1 degree, not -359.

        self.assertAlmostEqual(
            result["turn_rate_deg_s"].iloc[1],
            1.0
        )

    def test_stationary_classification(self):
        result = classify_phase(0, 0, 0, 0)

        self.assertEqual(result, "stationary")

    def test_climbing_classification(self):
        result = classify_phase(50, 200, 0, 0)

        self.assertEqual(result, "climbing")

    def test_invalid_timestamps(self):
        invalid_data = self.data.copy()
        invalid_data.loc[2, "time_s"] = 1

        with self.assertRaises(ValueError):
            validate_recording(invalid_data)

    def test_missing_column(self):
        invalid_data = self.data.drop(
            columns=["heading_deg"]
        )

        with self.assertRaises(ValueError):
            validate_recording(invalid_data)


if __name__ == "__main__":
    unittest.main()
