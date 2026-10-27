import unittest

from src.assessment.calibration_display import calibration_view


class CalibrationDisplayTests(unittest.TestCase):

    def test_new_export_ready(self):
        result = calibration_view({
            "status": "READY",
            "reference_available": True,
            "reference_type": "observed_calibration_posture",
            "stability": {"status": "STABLE"},
        })
        self.assertTrue(result["reference_available"])
        self.assertEqual(result["stability_status"], "STABLE")
        self.assertEqual(result["baseline"], "calibrated_reference")

    def test_old_export_ready(self):
        result = calibration_view({
            "status": "READY",
            "reference_posture": {
                "reference_type": "observed_calibration_posture"
            },
        })
        self.assertTrue(result["reference_available"])

    def test_unavailable(self):
        result = calibration_view({
            "status": "UNAVAILABLE",
            "reference_available": False,
        })
        self.assertFalse(result["reference_available"])
        self.assertEqual(
            result["baseline"], "first_valid_task_frame"
        )

    def test_failed_cannot_claim_reference(self):
        result = calibration_view({
            "status": "UNSTABLE",
            "reference_available": True,
        })
        self.assertFalse(result["reference_available"])


if __name__ == "__main__":
    unittest.main()
