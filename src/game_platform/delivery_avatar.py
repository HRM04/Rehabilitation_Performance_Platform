"""Animated delivery avatar for Delivery Dash Version 2."""

import math
import time

import cv2


class DeliveryAvatar:
    def __init__(self):
        self.last_position = None
        self.last_move_time = 0.0
        self.walk_phase = 0.0

    def draw(self, frame, position, carrying=True):
        x, y = map(int, position)
        now = time.monotonic()

        if self.last_position is not None:
            distance = math.dist(position, self.last_position)

            if distance > 2:
                self.walk_phase += min(distance / 20.0, 1.0)
                self.last_move_time = now

        self.last_position = tuple(position)

        walking = now - self.last_move_time < 0.2
        swing = int(12 * math.sin(self.walk_phase * 2.5)) if walking else 0

        # Ground shadow
        cv2.ellipse(
            frame, (x, y + 48), (30, 8),
            0, 0, 360, (115, 145, 145), -1
        )

        # Legs
        cv2.line(
            frame, (x - 8, y + 15),
            (x - 12 + swing, y + 45),
            (75, 80, 110), 7
        )
        cv2.line(
            frame, (x + 8, y + 15),
            (x + 12 - swing, y + 45),
            (75, 80, 110), 7
        )

        # Body
        cv2.ellipse(
            frame, (x, y - 7), (21, 30),
            0, 0, 360, (210, 120, 75), -1
        )

        # Head
        cv2.circle(
            frame, (x, y - 52), 19,
            (160, 195, 225), -1
        )

        # Hair
        cv2.ellipse(
            frame, (x, y - 62), (20, 11),
            0, 180, 360, (55, 65, 90), -1
        )

        # Face
        cv2.circle(frame, (x + 6, y - 53), 2, (40, 45, 65), -1)
        cv2.ellipse(
            frame, (x + 7, y - 46), (5, 3),
            0, 0, 180, (60, 75, 100), 1
        )

        # Arms
        cv2.line(
            frame, (x - 16, y - 20),
            (x + 18, y - 5),
            (160, 195, 225), 6
        )

        if carrying:
            # Package
            cv2.rectangle(
                frame, (x + 12, y - 22),
                (x + 48, y + 13),
                (65, 155, 230), -1
            )
            cv2.rectangle(
                frame, (x + 25, y - 22),
                (x + 33, y + 13),
                (120, 215, 255), -1
            )
            cv2.rectangle(
                frame, (x + 12, y - 22),
                (x + 48, y + 13),
                (60, 90, 130), 2
            )

        return frame
