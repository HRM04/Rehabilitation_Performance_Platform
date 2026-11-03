import os, unittest
from src.preprocessing.handedness import final_label, raw_label_for


class TestHandedness(unittest.TestCase):
    def test_default_keeps_mediapipe_label(self):
        os.environ.pop("HANDEDNESS_SWAP", None)
        self.assertEqual(final_label("Left"), "Left")
        self.assertEqual(final_label("Right"), "Right")

    def test_swap_switch_restores_old_behaviour(self):
        os.environ["HANDEDNESS_SWAP"] = "on"
        try:
            self.assertEqual(final_label("Left"), "Right")
            self.assertEqual(raw_label_for("Right"), "Left")
        finally:
            del os.environ["HANDEDNESS_SWAP"]


if __name__ == "__main__":
    unittest.main()
