import unittest

from src.assessment.game_trajectory import GameTrajectory


class TestGameTrajectoryIdentityRecovery(unittest.TestCase):

    def setUp(self):
        self.game = GameTrajectory(
            start_point=(200, 400),
            target_center=(1000, 400),
            object_name="cell phone",
        )
        self.game.start_attempt(object_id=8)
        self.game.last_seen_position = (500, 400)

    @staticmethod
    def detection(object_id, center):
        return {
            "object_id": object_id,
            "center": center,
            "class_name": "cell phone",
            "confidence": 0.8,
        }

    def test_original_id_preferred_over_duplicate(self):
        original = self.detection(8, (510, 400))
        duplicate = self.detection(11, (510, 455))

        selected = self.game.recover_object(
            [original, duplicate], contact_object_ids={8, 11}
        )

        self.assertIs(selected, original)
        self.assertEqual(self.game.active_object_id, 8)

    def test_original_id_preferred_regardless_of_order(self):
        original = self.detection(8, (510, 400))
        duplicate = self.detection(11, (510, 455))

        selected = self.game.recover_object(
            [duplicate, original], contact_object_ids={8, 11}
        )

        self.assertIs(selected, original)
        self.assertEqual(self.game.active_object_id, 8)

    def test_unconfirmed_new_id_is_not_adopted(self):
        duplicate = self.detection(11, (510, 455))

        selected = self.game.recover_object([duplicate])

        self.assertIsNone(selected)
        self.assertEqual(self.game.active_object_id, 8)

    def test_original_id_continues_when_alone(self):
        original = self.detection(8, (515, 402))

        selected = self.game.recover_object([original], contact_object_ids={8})

        self.assertIs(selected, original)
        self.assertEqual(self.game.active_object_id, 8)

    def test_distant_original_id_is_rejected(self):
        distant = self.detection(8, (900, 600))

        selected = self.game.recover_object([distant])

        self.assertIsNone(selected)
        self.assertEqual(self.game.active_object_id, 8)

    def test_missing_detection_does_not_change_identity(self):
        selected = self.game.recover_object([])

        self.assertIsNone(selected)
        self.assertEqual(self.game.active_object_id, 8)


if __name__ == "__main__":
    unittest.main()
