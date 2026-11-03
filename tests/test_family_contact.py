import os, unittest
from src.grasp import grasp_family as gf

M = {"enabled": True, "features": ["thumb_index"], "medians": [0.5],
     "tree": {"feat": "thumb_index", "thr": 0.236, "le": {"leaf": "PRECISION"}, "gt": {"leaf": "POWER"}}}


def run(fingertips, hand):
    gf.set_model(M)
    out = None
    for i in range(6):
        out = gf.classify_family(hand, i * 100, {"thumb_index": 1.2}, fingertips=fingertips)
    return out


class TestContactRule(unittest.TestCase):
    def test_wide_gap_with_two_fingertips_is_precision(self):
        self.assertEqual(run(2, "A"), "PRECISION")

    def test_wide_gap_with_five_fingertips_is_power(self):
        self.assertEqual(run(5, "B"), "POWER")

    def test_no_object_contact_info_keeps_shape_rule(self):
        self.assertEqual(run(None, "C"), "POWER")
        self.assertEqual(run(0, "D"), "POWER")

    def test_rule_can_be_switched_off(self):
        os.environ["FAMILY_CONTACT_RULE"] = "off"
        try:
            self.assertEqual(run(2, "E"), "POWER")
        finally:
            del os.environ["FAMILY_CONTACT_RULE"]
        gf.load_model()


if __name__ == "__main__":
    unittest.main()
