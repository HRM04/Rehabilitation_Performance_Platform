"""Scrolling scenery for Delivery Dash Version 2."""

import math

import cv2
import numpy as np


class WorldScenery:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # OpenCV uses BGR colors.
        self.themes = [
            {
                "sky": (245, 225, 175),
                "grass": (145, 210, 130),
                "house": (165, 210, 250),
                "roof": (100, 110, 215),
            },
            {
                "sky": (245, 225, 175),
                "grass": (145, 210, 130),
                "house": (220, 195, 245),
                "roof": (165, 115, 205),
            },
            {
                "sky": (245, 225, 175),
                "grass": (145, 210, 130),
                "house": (180, 235, 215),
                "roof": (115, 175, 130),
            },
        ]

    def draw_house(self, frame, x, theme, number):
        x = int(x)

        if x < -200 or x > self.width + 200:
            return

        cv2.rectangle(
            frame, (x - 75, 300), (x + 75, 420),
            theme["house"], -1
        )

        roof = np.array([
            [x - 95, 305],
            [x, 225],
            [x + 95, 305],
        ], dtype=np.int32)

        cv2.fillPoly(frame, [roof], theme["roof"])

        cv2.rectangle(
            frame, (x - 18, 355), (x + 22, 420),
            (100, 140, 175), -1
        )

        for wx in (-48, 48):
            cv2.rectangle(
                frame, (x + wx - 13, 325),
                (x + wx + 13, 350),
                (255, 255, 220), -1
            )

        cv2.putText(
            frame, str(number), (x - 12, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

    def draw_tree(self, frame, x, y=380):
        x = int(x)

        if x < -100 or x > self.width + 100:
            return

        cv2.rectangle(
            frame, (x - 10, y - 10),
            (x + 10, y + 70),
            (85, 120, 150), -1
        )

        for dx, dy, radius in [
            (-25, -25, 35),
            (25, -25, 35),
            (0, -55, 45),
        ]:
            cv2.circle(
                frame, (x + dx, y + dy),
                radius, (85, 165, 75), -1
            )

    def draw_vertical(self, scroll=0.0):
        """Scroll complete neighborhood scenes vertically."""
        import numpy as np

        progress = max(0.0, float(scroll)) / self.width
        current = int(progress)
        fraction = progress - current

        # Generate complete scenes using the existing renderer.
        current_scene = self.draw(current * self.width)
        next_scene = self.draw((current + 1) * self.width)

        shift = int(fraction * self.height)

        if shift == 0:
            return current_scene

        frame = np.empty_like(current_scene)

        # Current neighborhood moves down; next enters from above.
        frame[shift:] = current_scene[:self.height - shift]
        frame[:shift] = next_scene[self.height - shift:]

        return frame

    def draw(self, scroll=0.0):
        frame = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # Draw adjacent neighborhoods so scrolling is seamless.
        first_neighborhood = max(0, int(scroll // self.width))

        for neighborhood in range(
            first_neighborhood,
            first_neighborhood + 3,
        ):
            theme = self.themes[neighborhood % len(self.themes)]
            left = int(neighborhood * self.width - scroll)
            right = left + self.width

            if right < 0 or left > self.width:
                continue

            visible_left = max(0, left)
            visible_right = min(self.width, right)

            if visible_right <= visible_left:
                continue

            frame[:, visible_left:visible_right] = theme["sky"]

            cv2.rectangle(
                frame,
                (visible_left, 350),
                (visible_right, self.height),
                theme["grass"], -1
            )

            # Rolling hills.
            for hill_x in (300, 750, 1150):
                cx = int(left + hill_x)

                cv2.ellipse(
                    frame, (cx, 395),
                    (220, 95), 0, 180, 360,
                    (160, 205, 135), -1
                )

            # Clouds.
            for cx, cy in [
                (180, 125),
                (650, 105),
                (1050, 145),
            ]:
                screen_x = int(left + cx)

                for dx, dy, radius in [
                    (-35, 8, 30),
                    (0, -10, 43),
                    (35, 8, 30),
                ]:
                    cv2.circle(
                        frame,
                        (screen_x + dx, cy + dy),
                        radius,
                        (255, 255, 255), -1
                    )

            # Houses and trees.
            self.draw_house(
                frame, left + 1050,
                theme, neighborhood + 1
            )

            for tree_x in (130, 480, 800):
                self.draw_tree(frame, left + tree_x)

            # Flowers.
            for flower_x in range(70, self.width, 95):
                fx = int(left + flower_x)

                if 0 <= fx < self.width:
                    fy = 575 + int(
                        12 * math.sin(flower_x * 0.04)
                    )
                    cv2.circle(
                        frame, (fx, fy),
                        6, (190, 100, 245), -1
                    )
                    cv2.circle(
                        frame, (fx, fy),
                        3, (80, 220, 255), -1
                    )

        # Continuous road in the foreground.
        cv2.rectangle(
            frame, (0, 425),
            (self.width, 535),
            (165, 180, 195), -1
        )

        for x in range(
            -120, self.width + 120, 120
        ):
            shifted = int(
                (x - scroll * 0.3) % (self.width + 120)
            ) - 60

            cv2.line(
                frame,
                (shifted, 480),
                (shifted + 60, 480),
                (245, 245, 245), 5
            )

        return frame
