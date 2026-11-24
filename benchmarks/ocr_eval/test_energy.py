"""Tests for energy.py's parsing and arithmetic (no sudo needed)."""

import unittest

from energy import energy_above_idle, parse_samples

SAMPLE = """Machine model: Mac
*** Sampled system activity (Sun Nov 23 10:00:00 2026 -0800) (100.50ms elapsed) ***

**** Processor usage ****
CPU Power: 1800 mW
GPU Power: 150 mW
ANE Power: 50 mW
Combined Power (CPU + GPU + ANE): 2000 mW

*** Sampled system activity (Sun Nov 23 10:00:00 2026 -0800) (99.50ms elapsed) ***

CPU Power: 5800 mW
GPU Power: 150 mW
ANE Power: 50 mW
Combined Power (CPU + GPU + ANE): 6000 mW

*** Sampled system activity (Sun Nov 23 10:00:00 2026 -0800) (100.00ms elapsed) ***

CPU Power: 1800 mW
"""


class Energy(unittest.TestCase):
    def test_parse_complete_samples_only(self):
        # the third sample was cut off (no Combined line): ignored
        self.assertEqual(parse_samples(SAMPLE), [(0.1005, 2.0), (0.0995, 6.0)])

    def test_energy_uses_each_samples_length(self):
        above, secs, total = energy_above_idle([(0.1005, 2.0), (0.0995, 6.0)], idle_w=2.0)
        self.assertAlmostEqual(secs, 0.2)
        self.assertAlmostEqual(total, 0.1005 * 2 + 0.0995 * 6)
        self.assertAlmostEqual(above, 0.0995 * 4)          # only the busy sample is above idle

    def test_idle_power_gives_zero(self):
        above, _, _ = energy_above_idle([(0.1, 3.0)] * 10, idle_w=3.0)
        self.assertAlmostEqual(above, 0.0)


if __name__ == "__main__":
    unittest.main()
