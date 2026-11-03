import unittest

from src.assessment.trajectory_reliability import (
    assess_trajectory_reliability,
)


def observations(points, times=None, ids=None):
    if times is None:
        times = [i * 100 for i in range(len(points))]
    if ids is None:
        ids = [1] * len(points)

    return [
        {"center": p, "timestamp_ms": t, "object_id": oid}
        for p, t, oid in zip(points, times, ids)
    ]


class TestTrajectoryReliabilityV2(unittest.TestCase):

    def test_horizontal_forward(self):
        points = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
        r = assess_trajectory_reliability(
            observations(points), [(0, 0), (40, 0)]
        )
        self.assertTrue(r["eligible"])
        self.assertEqual(r["path_relative_reversals"], 0)
        self.assertAlmostEqual(r["time_weighted_average_speed_px_s"], 100)
        self.assertAlmostEqual(r["temporal_coverage"], 1)

    def test_vertical_reversal(self):
        points = [(0, 0), (0, 20), (0, 10), (0, 30), (0, 40)]
        r = assess_trajectory_reliability(
            observations(points), [(0, 0), (0, 40)]
        )
        self.assertEqual(r["path_relative_reversals"], 2)

    def test_irregular_sampling_weighted_speed(self):
        points = [(0, 0), (10, 0), (20, 0)]
        r = assess_trajectory_reliability(
            observations(points, [0, 100, 1000]),
            [(0, 0), (20, 0)],
            gap_threshold_ms=1000,
            minimum_samples=3,
        )
        self.assertAlmostEqual(
            r["time_weighted_average_speed_px_s"], 20
        )

    def test_gap_reduces_coverage(self):
        points = [(0, 0), (10, 0), (20, 0)]
        r = assess_trajectory_reliability(
            observations(points, [0, 100, 1100]),
            [(0, 0), (20, 0)],
            minimum_samples=3,
        )
        self.assertFalse(r["eligible"])
        self.assertEqual(r["sampling_gap_count"], 1)
        self.assertLess(r["temporal_coverage"], 0.1)

    def test_tracking_id_change(self):
        points = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
        r = assess_trajectory_reliability(
            observations(points, ids=[1, 1, 2, 2, 2]),
            [(0, 0), (40, 0)],
        )
        self.assertFalse(r["eligible"])
        self.assertEqual(r["tracking_id_changes"], 1)

    def test_non_increasing_timestamps(self):
        points = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
        r = assess_trajectory_reliability(
            observations(points, [0, 100, 100, 300, 400]),
            [(0, 0), (40, 0)],
        )
        self.assertFalse(r["eligible"])
        self.assertEqual(r["invalid_timestamp_count"], 1)

    def test_reversal_does_not_cross_tracking_gap(self):
        points = [(0, 0), (20, 0), (10, 0), (30, 0), (40, 0)]
        r = assess_trajectory_reliability(
            observations(points, [0, 100, 1000, 1100, 1200]),
            [(0, 0), (40, 0)],
        )
        self.assertEqual(r["path_relative_reversals"], 0)


if __name__ == "__main__":
    unittest.main()
