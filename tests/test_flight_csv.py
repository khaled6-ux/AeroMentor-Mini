
import unittest

import pandas as pd

from src.flight_csv import analyze_flight_csv


class TestFlightCSV(unittest.TestCase):

    def setUp(self):
        self.targets = {
            "altitude_ft": 3000,
            "airspeed_kt": 120,
            "heading_deg": 90,
        }

        self.data = pd.DataFrame({
            "time_s": [0, 1, 2, 3, 4],
            "altitude_ft": [3000, 3015, 2990, 3030, 3010],
            "airspeed_kt": [120, 122, 119, 125, 121],
            "heading_deg": [90, 92, 89, 95, 91],
        })

    def test_correct_mae(self):
        result = analyze_flight_csv(
            self.data,
            self.targets,
        )

        self.assertAlmostEqual(
            result["altitude_mae_ft"], 13.0
        )
        self.assertAlmostEqual(
            result["airspeed_mae_kt"], 1.8
        )
        self.assertAlmostEqual(
            result["heading_mae_deg"], 1.8
        )

    def test_missing_column(self):
        invalid_data = self.data.drop(
            columns=["heading_deg"]
        )

        with self.assertRaises(ValueError):
            analyze_flight_csv(
                invalid_data,
                self.targets,
            )

    def test_empty_csv(self):
        empty_data = self.data.iloc[0:0]

        with self.assertRaises(ValueError):
            analyze_flight_csv(
                empty_data,
                self.targets,
            )

    def test_missing_value(self):
        invalid_data = self.data.copy()
        invalid_data.loc[0, "altitude_ft"] = None

        with self.assertRaises(ValueError):
            analyze_flight_csv(
                invalid_data,
                self.targets,
            )

    def test_heading_wraparound(self):
        data = pd.DataFrame({
            "time_s": [0, 1],
            "altitude_ft": [3000, 3000],
            "airspeed_kt": [120, 120],
            "heading_deg": [359, 1],
        })

        targets = {
            "altitude_ft": 3000,
            "airspeed_kt": 120,
            "heading_deg": 0,
        }

        result = analyze_flight_csv(data, targets)

        self.assertAlmostEqual(
            result["heading_mae_deg"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
