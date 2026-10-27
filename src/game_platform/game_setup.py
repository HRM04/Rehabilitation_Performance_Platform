"""Interactive setup screen for Delivery Dash."""

import cv2
import numpy as np

from src.game_platform.game_config import (
    GameConfig,
    TRAJECTORIES,
    REPETITION_OPTIONS,
    available_objects,
)


def draw_text(frame, text, x, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(
        frame, str(text), (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def choose_game_config():
    objects = available_objects()
    trajectories = list(TRAJECTORIES.keys())

    object_index = 0
    trajectory_index = 0
    repetition_index = 2

    window = "Delivery Dash - Game Setup"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    try:
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            frame[:] = (145, 105, 65)

            cv2.rectangle(
                frame, (70, 75), (1210, 650),
                (205, 165, 105), -1
            )

            draw_text(
                frame, "DELIVERY DASH!",
                390, 155, 1.5
            )
            draw_text(
                frame, "Choose your rehabilitation adventure",
                325, 205, 0.8
            )

            selected_object = objects[object_index]
            selected_trajectory = trajectories[trajectory_index]
            selected_repetitions = REPETITION_OPTIONS[repetition_index]

            draw_text(frame, "OBJECT", 180, 285, 0.8)
            draw_text(
                frame,
                selected_object["name"],
                180, 330, 1.1
            )
            draw_text(
                frame,
                f"Status: {selected_object['status']}",
                180, 370, 0.6
            )

            draw_text(frame, "TRAJECTORY", 700, 285, 0.8)
            draw_text(
                frame,
                TRAJECTORIES[selected_trajectory],
                700, 330, 1.1
            )

            draw_text(frame, "REPETITIONS", 180, 460, 0.8)
            draw_text(
                frame,
                str(selected_repetitions),
                180, 510, 1.3
            )

            draw_text(
                frame,
                "UP/DOWN: Object    LEFT/RIGHT: Trajectory",
                180, 565, 0.65
            )
            draw_text(
                frame,
                "1-4: Repetitions    ENTER: Start    Q: Quit",
                180, 605, 0.65
            )

            cv2.imshow(window, frame)
            key = cv2.waitKeyEx(30)

            if key in (ord("q"), ord("Q"), 27):
                return None

            elif key in (13, 10):
                return GameConfig(
                    object_name=selected_object["id"],
                    trajectory=selected_trajectory,
                    repetitions=selected_repetitions,
                ).validate()

            elif key in (2490368, 63232, 82):
                object_index = (object_index - 1) % len(objects)

            elif key in (2621440, 63233, 84):
                object_index = (object_index + 1) % len(objects)

            elif key in (2424832, 63234, 81):
                trajectory_index = (
                    trajectory_index - 1
                ) % len(trajectories)

            elif key in (2555904, 63235, 83):
                trajectory_index = (
                    trajectory_index + 1
                ) % len(trajectories)

            elif ord("1") <= key <= ord("4"):
                repetition_index = key - ord("1")

    finally:
        cv2.destroyWindow(window)


if __name__ == "__main__":
    result = choose_game_config()
    print("Selected configuration:", result)
