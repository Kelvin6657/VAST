"""Statistical conventions tested with synthetic values."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'VAST'))
from progressive import summarize_values

class StatisticsTests(unittest.TestCase):
    def test_mean_and_sample_standard_deviation(self):
        self.assertEqual(summarize_values([1., 2., 3.]),
                         {'mean': 2., 'sample_std': 1., 'n': 3})

    def test_incomplete_or_nonfinite_values_are_rejected(self):
        for values in ([1.], [1., 2., float('nan')], [1., 2., float('inf')]):
            with self.assertRaises(ValueError):
                summarize_values(values)

if __name__ == '__main__':
    unittest.main()
