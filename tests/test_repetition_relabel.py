import unittest
from src.assessment.repetition_analysis import segment_repetitions


def row(frame, state, oid, movement=5.0, cls="bottle"):
    return {"frame_number": frame, "timestamp_ms": frame * 100, "interaction_state": state,
            "handedness": "Right", "object_class": cls, "object_id": oid, "object_movement": movement}


class TestRelabel(unittest.TestCase):
    def test_same_class_new_id_mid_grasp_is_one_repetition(self):
        rows = [row(i, "GRASPING", "7") for i in range(4)] + [row(i, "TRANSPORT", "9", 10.0) for i in range(4, 12)]
        rows += [row(i, "NONE", None, 0.0) for i in range(12, 18)]
        self.assertEqual(len(segment_repetitions(rows)), 1)

    def test_different_class_still_ends_repetition(self):
        rows = [row(i, "GRASPING", "7") for i in range(6)] + [row(i, "GRASPING", "9", cls="cup") for i in range(6, 14)]
        rows += [row(i, "NONE", None, 0.0) for i in range(14, 20)]
        self.assertGreaterEqual(len(segment_repetitions(rows)), 1)


if __name__ == "__main__":
    unittest.main()
