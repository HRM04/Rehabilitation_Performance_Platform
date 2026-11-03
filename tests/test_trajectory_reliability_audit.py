import unittest
from src.assessment.trajectory_assessment import TrajectoryAssessment


class TestTrajectoryReliability(unittest.TestCase):

    def make_assessment(self, points, desired=None):
        a = TrajectoryAssessment(
            start_point=(0, 0),
            target_center=(100, 0),
            target_radius=10,
            waypoints=desired or [],
        )
        a.start_attempt(object_id=1)
        for i, point in enumerate(points):
            a.update(point, object_id=1, timestamp_ms=i * 100)
        return a

    def test_perfect_straight_path(self):
        a = self.make_assessment([(0, 0), (50, 0), (100, 0)])
        m = a.calculate_metrics_for_window(0, 200, object_id=1)
        self.assertAlmostEqual(m["path_length_px"], 100)
        self.assertAlmostEqual(m["movement_efficiency"], 1)
        self.assertAlmostEqual(m["mean_trajectory_deviation_px"], 0)
        self.assertAlmostEqual(m["final_target_error_px"], 0)
        self.assertTrue(m["target_reached"])

    def test_detour_reduces_directness(self):
        a = self.make_assessment([(0, 0), (50, 50), (100, 0)])
        m = a.calculate_metrics_for_window(0, 200, object_id=1)
        self.assertAlmostEqual(m["path_length_px"], 141.421356, places=4)
        self.assertAlmostEqual(m["movement_efficiency"], 2**-0.5)
        self.assertGreater(m["mean_trajectory_deviation_px"], 0)

    def test_curved_shortcut_is_not_path_adherence(self):
        a = self.make_assessment(
            [(0, 0), (50, 0), (100, 0)],
            desired=[(50, 50)],
        )
        m = a.calculate_metrics_for_window(0, 200, object_id=1)
        self.assertAlmostEqual(m["movement_efficiency"], 1)
        self.assertGreater(m["mean_trajectory_deviation_px"], 0)

    def test_single_observation_limitation(self):
        a = self.make_assessment([(100, 0)])
        m = a.calculate_metrics_for_window(0, 0, object_id=1)
        self.assertEqual(m["sample_count"], 1)
        self.assertIsNone(m["movement_efficiency"])
        self.assertTrue(m["target_reached"])

    def test_repetition_window_isolation(self):
        a = self.make_assessment(
            [(0, 0), (50, 0), (100, 0)]
        )
        m = a.calculate_metrics_for_window(100, 200, object_id=1)
        self.assertEqual(m["sample_count"], 2)
        self.assertAlmostEqual(m["path_length_px"], 50)

    def test_wrong_object_excluded(self):
        a = self.make_assessment([(0, 0), (100, 0)])
        self.assertIsNone(
            a.calculate_metrics_for_window(0, 100, object_id=2)
        )


if __name__ == "__main__":
    unittest.main()
