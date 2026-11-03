import unittest
from src.grasp import grasp_guess as gg

DATA = {"enabled": True, "sequences": 1, "rounds": [1], "proto": {
    "cylindrical": {"thumb_index": [0.9, 0.1]},
    "whole_hand_power": {"thumb_index": [0.3, 0.1]},
    "tip_pinch": {"thumb_index": [0.1, 0.05]},
    "tripod": {"thumb_index": [0.2, 0.05]}}}


def run(hand, feats, family):
    gg.set_prototypes(DATA)
    out = None
    for i in range(6):
        out = gg.best_guess(hand, i * 100, feats, family)
    return out


class TestGuess(unittest.TestCase):
    def test_power_picks_closest_power_grasp(self):
        self.assertEqual(run("A", {"thumb_index": 0.85}, "POWER"), "cylindrical")

    def test_precision_never_returns_a_power_grasp(self):
        self.assertEqual(run("B", {"thumb_index": 0.19}, "PRECISION"), "tripod")

    def test_no_family_or_features_gives_none(self):
        self.assertIsNone(run("C", {"thumb_index": 0.5}, None))
        self.assertIsNone(run("D", None, "POWER"))

    def test_disabled_gives_none(self):
        gg.set_prototypes(None)
        try:
            self.assertIsNone(gg.best_guess("E", 0, {"thumb_index": 0.5}, "POWER"))
        finally:
            gg.load()


if __name__ == "__main__":
    unittest.main()
