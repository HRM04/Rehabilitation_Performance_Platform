import unittest
from src.preprocessing.grasp_voting import GraspVoter


class TestGraspVoter(unittest.TestCase):
    def run_seq(self, labels, step=200, window=5, hand="Right"):
        v = GraspVoter(window=window)
        return [v.vote(hand, i * step, l, [l])[0] for i, l in enumerate(labels)]

    def test_single_flicker_removed(self):
        out = self.run_seq(["cylindrical"] * 4 + ["pincer"] + ["cylindrical"] * 3)
        self.assertNotIn("pincer", out)

    def test_real_change_follows_after_three_frames(self):
        out = self.run_seq(["cylindrical"] * 5 + ["tripod"] * 5)
        self.assertEqual(out[-1], "tripod")
        self.assertEqual(out[5], "cylindrical")

    def test_ambiguous_is_not_hidden(self):
        out = self.run_seq(["AMBIGUOUS"] * 5)
        self.assertEqual(out[-1], "AMBIGUOUS")

    def test_disabled_with_window_one(self):
        out = self.run_seq(["a", "b", "a", "b"], window=1)
        self.assertEqual(out, ["a", "b", "a", "b"])

    def test_old_votes_expire(self):
        v = GraspVoter(window=5)
        for i in range(4):
            v.vote("Right", i * 100, "cylindrical", ["cylindrical"])
        lab, _ = v.vote("Right", 10000, "pincer", ["pincer"])
        self.assertEqual(lab, "pincer")

    def test_hands_are_separate(self):
        v = GraspVoter(window=5)
        for i in range(4):
            v.vote("Left", i * 100, "cylindrical", ["cylindrical"])
        lab, _ = v.vote("Right", 500, "pincer", ["pincer"])
        self.assertEqual(lab, "pincer")

    def test_candidates_follow_label(self):
        v = GraspVoter(window=5)
        for i in range(3):
            v.vote("Right", i * 100, "AMBIGUOUS", ["a", "b"])
        lab, cands = v.vote("Right", 400, "UNKNOWN", [])
        self.assertEqual((lab, cands), ("AMBIGUOUS", ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
