import unittest
from src.object_detection.hand_object_interaction import pinch_contact_at_object

BOX = {"bounding_box": (100, 100, 200, 300)}


def hand(**pos):
    names = ("THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP")
    return {n: {"x_px": pos.get(n, (500, 500))[0], "y_px": pos.get(n, (500, 500))[1]} for n in names}


class TestPinchContact(unittest.TestCase):
    def test_thumb_and_index_on_object_with_precision_shape(self):
        h = hand(THUMB_TIP=(120, 150), INDEX_FINGER_TIP=(180, 160))
        self.assertTrue(pinch_contact_at_object(h, BOX, 2, "PRECISION"))

    def test_two_fingers_without_thumb_do_not_count(self):
        h = hand(INDEX_FINGER_TIP=(120, 150), MIDDLE_FINGER_TIP=(130, 160))
        self.assertFalse(pinch_contact_at_object(h, BOX, 2, "PRECISION"))

    def test_power_shape_or_no_object_do_not_count(self):
        h = hand(THUMB_TIP=(120, 150), INDEX_FINGER_TIP=(180, 160))
        self.assertFalse(pinch_contact_at_object(h, BOX, 2, "POWER"))
        self.assertFalse(pinch_contact_at_object(h, None, 2, "PRECISION"))


if __name__ == "__main__":
    unittest.main()
