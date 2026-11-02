
import unittest

from src.analyze_flight import calculate_mae


class TestFlightAnalysis(unittest.TestCase):

    def test_perfect_altitude(self):
        result = calculate_mae([5000, 5000, 5000], 5000)
        self.assertEqual(result, 0)

    def test_altitude_error(self):
        result = calculate_mae([5100, 4900, 5000], 5000)
        self.assertAlmostEqual(result, 66.67, places=2)

    def test_heading_wraparound(self):
        result = calculate_mae([359, 1], 0, circular=True)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
