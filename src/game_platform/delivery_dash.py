"""Delivery Dash: patient-facing rehabilitation game prototype."""

import math
import time

import cv2
import numpy as np

from src.assessment.game_trajectory import GameTrajectory
from src.game_platform.world_scenery import WorldScenery
from src.game_platform.vertical_scenery import VerticalScenery
from src.game_platform.delivery_avatar import DeliveryAvatar


class DeliveryDash:
    def __init__(
        self,
        target_repetitions=10,
        width=1280,
        height=720,
        object_name="bottle",
        trajectory_preset="horizontal",
    ):
        self.width = width
        self.height = height
        self.target_repetitions = max(1, int(target_repetitions))
        self.object_name = object_name
        self.trajectory_preset = trajectory_preset
        self.completed = 0
        self.package_position = [220.0, 470.0]
        self.destination = (1050, 470)
        self.start = (220, 470)
        self.celebration_until = 0.0
        self.running = True

        # Visual world progression is independent of physical coordinates.
        self.world_scroll = 0.0
        self.scroll_from = 0.0
        self.scroll_to = 0.0
        self.scroll_started_at = None
        self.scroll_duration = 1.8
        self.neighborhood_width = self.width
        self.scenery = WorldScenery(self.width, self.height)
        self.vertical_scenery = VerticalScenery(self.width, self.height)
        self.avatar = DeliveryAvatar()

        self.trajectory = GameTrajectory(
            object_name=self.object_name,
            start_point=self.start,
            target_center=self.destination,
            target_radius=70,
        )

        self.trajectory.set_preset(
            self.trajectory_preset,
            frame_width=self.width,
            frame_height=self.height,
        )

        self.start = self.trajectory.start_point
        self.destination = self.trajectory.target_center
        self.package_position = [
            float(self.start[0]),
            float(self.start[1]),
        ]

        # Keyboard simulation only; camera integration comes later.
        self.trajectory.start_attempt(object_id=1)

    def reset_package(self):
        self.package_position = [float(self.start[0]), float(self.start[1])]

    def move_package(self, dx, dy):
        if self.completed >= self.target_repetitions:
            return

        self.package_position[0] = max(
            70, min(self.width - 70, self.package_position[0] + dx)
        )
        self.package_position[1] = max(
            170, min(self.height - 90, self.package_position[1] + dy)
        )

        self.trajectory.update(
            tuple(map(int, self.package_position)),
            object_id=1,
        )

        distance = math.dist(self.package_position, self.destination)

        if distance < 85:
            self.trajectory.end_attempt(outcome="COMPLETED")
            self.completed += 1
            self.celebration_until = time.monotonic() + 1.3
            self.reset_package()

            if self.completed < self.target_repetitions:
                self.scroll_from = self.world_scroll
                self.scroll_to = (
                    self.completed * self.neighborhood_width
                )
                self.scroll_started_at = time.monotonic()
                self.trajectory.reset()
                self.trajectory.start_attempt(object_id=1)

    def draw(self):
        if self.scroll_started_at is not None:
            elapsed = time.monotonic() - self.scroll_started_at
            t = min(1.0, elapsed / self.scroll_duration)
            smooth = t * t * (3.0 - 2.0 * t)
            self.world_scroll = (
                self.scroll_from
                + (self.scroll_to - self.scroll_from) * smooth
            )
            if t >= 1.0:
                self.scroll_started_at = None

        if self.trajectory_preset == "vertical":
            frame = self.vertical_scenery.draw(-self.world_scroll)
        else:
            frame = self.scenery.draw(self.world_scroll)

        # Avatar follows the simulated tracked-object position.
        self.avatar.draw(frame, self.package_position)

        # Header
        cv2.rectangle(
            frame, (0, 0), (self.width, 85),
            (70, 85, 130), -1
        )
        cv2.putText(
            frame, "DELIVERY DASH!", (40, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3,
            (255, 255, 255), 3, cv2.LINE_AA
        )

        progress = f"{self.completed} / {self.target_repetitions}"
        cv2.putText(
            frame, progress, (990, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            (255, 255, 255), 3, cv2.LINE_AA
        )

        # Progress bar
        cv2.rectangle(
            frame, (400, 30), (850, 55),
            (215, 215, 215), -1
        )
        progress_width = int(
            450 * self.completed / self.target_repetitions
        )
        cv2.rectangle(
            frame, (400, 30), (400 + progress_width, 55),
            (100, 220, 120), -1
        )

        if self.completed >= self.target_repetitions:
            message = "MISSION COMPLETE!"
        elif time.monotonic() < self.celebration_until:
            message = "AWESOME DELIVERY!"
        else:
            message = "Deliver the package to the house!"

        cv2.putText(
            frame, message, (230, 630),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            (50, 70, 110), 3, cv2.LINE_AA
        )

        cv2.putText(
            frame, "Arrow keys: move  |  R: restart  |  Q: quit",
            (300, 680), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (60, 80, 100), 2, cv2.LINE_AA
        )

        self.trajectory.draw(frame)
        return frame

    def run(self):
        window = "Delivery Dash - Version 2"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.width, self.height)

        while self.running:
            cv2.imshow(window, self.draw())
            key = cv2.waitKeyEx(30)

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                self.completed = 0
                self.world_scroll = 0.0
                self.scroll_from = 0.0
                self.scroll_to = 0.0
                self.scroll_started_at = None
                self.reset_package()
                self.trajectory.reset()
                self.trajectory.start_attempt(object_id=1)
            elif key in (81, 2424832, 63234):
                self.move_package(-20, 0)
            elif key in (83, 2555904, 63235):
                self.move_package(20, 0)
            elif key in (82, 2490368, 63232):
                self.move_package(0, -20)
            elif key in (84, 2621440, 63233):
                self.move_package(0, 20)

        cv2.destroyWindow(window)


if __name__ == "__main__":
    DeliveryDash(target_repetitions=3).run()
