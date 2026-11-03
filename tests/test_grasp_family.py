import unittest
from src.grasp import grasp_family as gf

TINY = {"enabled": True, "features": ["thumb_index"], "medians": [0.5],
        "tree": {"feat": "thumb_index", "thr": 0.25, "n": 2,
                 "le": {"leaf": "PRECISION", "n": 1}, "gt": {"leaf": "POWER", "n": 1}}}


class TestFamily(unittest.TestCase):
    def setUp(self):
        gf.set_model(TINY)

    def tearDown(self):
        gf.load_model()

    def test_close_thumb_index_is_precision(self):
        out = [gf.classify_family("Right", i * 200, {"thumb_index": 0.1}) for i in range(4)]
        self.assertEqual(out[-1], "PRECISION")

    def test_wide_is_power(self):
        out = [gf.classify_family("Right", i * 200, {"thumb_index": 0.8}) for i in range(4)]
        self.assertEqual(out[-1], "POWER")

    def test_missing_feature_uses_median(self):
        self.assertEqual(gf.classify_family("Left", 0, {}), "POWER")

    def test_disabled_model_reports_nothing(self):
        gf.set_model({**TINY, "enabled": False})
        self.assertIsNone(gf.classify_family("Right", 0, {"thumb_index": 0.1}))

    def test_assessment(self):
        self.assertEqual(gf.family_assessment("POWER", False, ["cylindrical", "handle"]), "APPROPRIATE")
        self.assertEqual(gf.family_assessment("PRECISION", False, ["cylindrical", "handle"]), "INAPPROPRIATE")
        self.assertEqual(gf.family_assessment("POWER", True, ["cylindrical"]), "NOT_ASSESSED")
        self.assertEqual(gf.family_assessment(None, False, ["cylindrical"]), "NOT_ASSESSED")
        self.assertEqual(gf.family_assessment("POWER", False, []), "NOT_ASSESSED")

    def test_mixed_expected_families_accept_both(self):
        self.assertEqual(gf.family_assessment("PRECISION", False, ["cylindrical", "tripod"]), "APPROPRIATE")


if __name__ == "__main__":
    unittest.main()
