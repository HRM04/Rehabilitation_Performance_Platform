"""Top-down vertical neighborhood scenery for Delivery Dash."""

import math

import cv2
import numpy as np


class VerticalScenery:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        self.house_colors = [
            (180, 170, 245),
            (175, 225, 255),
            (190, 235, 175),
            (225, 195, 255),
            (170, 215, 245),
        ]

    def draw_house(self, frame, x, y, number):
        """Draw a small house viewed from above."""
        color = self.house_colors[number % len(self.house_colors)]

        cv2.rectangle(
            frame, (x - 80, y - 65),
            (x + 80, y + 65),
            (80, 100, 120), -1
        )

        cv2.rectangle(
            frame, (x - 73, y - 58),
            (x + 73, y + 58),
            color, -1
        )

        # Roof ridge.
        cv2.line(
            frame, (x, y - 58),
            (x, y + 58),
            (100, 115, 145), 5
        )

        # Front path leading to the road.
        if x < self.width // 2:
            cv2.rectangle(
                frame, (x + 80, y - 12),
                (510, y + 12),
                (195, 205, 210), -1
            )
        else:
            cv2.rectangle(
                frame, (770, y - 12),
                (x - 80, y + 12),
                (195, 205, 210), -1
            )

        cv2.putText(
            frame, str(number),
            (x - 14, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (50, 60, 85), 2,
            cv2.LINE_AA
        )

    def draw(self, scroll=0.0):
        """Draw a vertically scrolling neighborhood."""
        frame = np.full(
            (self.height, self.width, 3),
            (150, 205, 130),
            dtype=np.uint8
        )

        # Vertical street.
        cv2.rectangle(
            frame, (510, 0),
            (770, self.height),
            (165, 175, 190), -1
        )

        # Sidewalks.
        cv2.rectangle(
            frame, (485, 0),
            (510, self.height),
            (205, 215, 215), -1
        )
        cv2.rectangle(
            frame, (770, 0),
            (795, self.height),
            (205, 215, 215), -1
        )

        # Road markings move downward as the player progresses.
        offset = int(scroll) % 140

        for y in range(-140 + offset, self.height + 140, 140):
            cv2.line(
                frame, (640, y),
                (640, y + 65),
                (245, 245, 245), 6
            )

        # Houses are placed along both sides of the road.
        # Determine visible rows for either scrolling direction.
        first_row = math.floor((scroll - 260) / 300)
        last_row = math.ceil(
            (scroll + self.height + 260) / 300
        )

        for row in range(first_row, last_row + 1):
            y = int(140 + row * 300 - scroll)

            if -120 <= y <= self.height + 120:
                self.draw_house(frame, 285, y, row * 2 + 1)
                self.draw_house(frame, 995, y, row * 2 + 2)

                for x in (100, 445, 835, 1170):
                    cv2.circle(
                        frame, (x, y + 105),
                        32, (75, 155, 85), -1
                    )
                    cv2.circle(
                        frame, (x - 9, y + 95),
                        18, (95, 185, 100), -1
                    )

        return frame
