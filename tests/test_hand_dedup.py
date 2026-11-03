import unittest
from types import SimpleNamespace as NS
from src.preprocessing.hand_dedup import dedupe_hands


def hand(cx, cy, label, score):
    pts = [NS(x=cx / 1280.0, y=cy / 720.0) for _ in range(21)]
    return NS(landmark=pts), NS(classification=[NS(label=label, score=score)])


def results_of(*hs):
    return NS(multi_hand_landmarks=[h[0] for h in hs], multi_handedness=[h[1] for h in hs])


class TestHandDedup(unittest.TestCase):
    def test_same_hand_twice_keeps_stronger(self):
        r = dedupe_hands(results_of(hand(400, 300, "Left", 0.99), hand(420, 310, "Left", 0.90)))
        self.assertEqual(len(r.multi_hand_landmarks), 1)
        self.assertEqual(r.multi_handedness[0].classification[0].score, 0.99)

    def test_two_far_hands_same_label_relabelled(self):
        r = dedupe_hands(results_of(hand(200, 300, "Left", 0.99), hand(900, 300, "Left", 0.95)))
        labels = sorted(h.classification[0].label for h in r.multi_handedness)
        self.assertEqual(labels, ["Left", "Right"])
        self.assertEqual(len(r.multi_hand_landmarks), 2)

    def test_normal_two_hands_untouched(self):
        r = dedupe_hands(results_of(hand(200, 300, "Left", 0.99), hand(900, 300, "Right", 0.95)))
        self.assertEqual(len(r.multi_hand_landmarks), 2)

    def test_single_hand_and_none(self):
        self.assertEqual(len(dedupe_hands(results_of(hand(200, 300, "Left", 0.9))).multi_hand_landmarks), 1)
        self.assertIsNone(dedupe_hands(NS(multi_hand_landmarks=None, multi_handedness=None)).multi_hand_landmarks)


if __name__ == "__main__":
    unittest.main()
