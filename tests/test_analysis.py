
import unittest


def calculate_mae(values, target):
    errors = [abs(value - target) for value in values]
    return sum(errors) / len(errors)


class TestFlightAnalysis(unittest.TestCase):

    def test_perfect_altitude(self):
        altitudes = [5000, 5000, 5000]
        result = calculate_mae(altitudes, 5000)
        self.assertEqual(result, 0)

    def test_altitude_error(self):
        altitudes = [5100, 4900, 5000]
        result = calculate_mae(altitudes, 5000)
        self.assertAlmostEqual(result, 66.67, places=2)


if __name__ == "__main__":
    unittest.main()
