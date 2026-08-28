import csv
import cv2
from datetime import datetime
from pathlib import Path
from src.capture.video_playback import play_video
import mediapipe as mp
import time

from src.grasp.grasp_features import calculate_thumb_index_distance
from src.grasp.grasp_classifier import classify_pinch
from src.data.landmark_logger import (create_landmark_logger,log_hand_landmarks)

# -----------------------------
# MediaPipe setup
# -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Configuration
# -----------------------------

CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30

OUTPUT_DIR = Path("data/raw/video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"{timestamp}_camera_test.mp4"

LANDMARK_DIR = Path("data/raw/landmarks")
LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

landmark_output_path = (
    LANDMARK_DIR / f"{timestamp}_camera_test_landmarks.csv"
)




# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")



# Request capture settings
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)



# -----------------------------
# Create video writer
# -----------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():
    camera.release()
    raise RuntimeError("Could not create video writer.")


print("Recording started.")
print(f"Saving to: {output_path}")
print("Press Q to stop recording.")


# open CSV file for writing hand landmarks
csv_file, csv_writer = create_landmark_logger(landmark_output_path)

frame_number = 0

start_time = time.perf_counter()


# -----------------------------
# Capture loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    frame_number += 1
    timestamp_ms = int((time.perf_counter() - start_time) * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
    else:
        num_hands = 0



    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            handedness = results.multi_handedness[hand_index]

            hand_label = handedness.classification[0].label
            hand_confidence = handedness.classification[0].score

            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"

            # Draw the hand skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------
            # Store all 21 landmarks
            # -----------------------------

            hand_data = {}

            for landmark_name in mp_hands.HandLandmark:

                landmark = hand_landmarks.landmark[landmark_name.value]

                x_px = int(landmark.x * actual_width)
                y_px = int(landmark.y * actual_height)

                hand_data[landmark_name.name] = {
                    "x_norm": landmark.x,
                    "y_norm": landmark.y,
                    "z_rel": landmark.z,
                    "x_px": x_px,
                    "y_px": y_px
                }


            # -----------------------------
            # Wrist-relative coordinates
            # -----------------------------

            wrist_x = hand_data["WRIST"]["x_px"]
            wrist_y = hand_data["WRIST"]["y_px"]

            for landmark_name, coordinates in hand_data.items():

                coordinates["x_rel"] = coordinates["x_px"] - wrist_x
                coordinates["y_rel"] = coordinates["y_px"] - wrist_y


            # -----------------------------
            # Scale normalization
            # -----------------------------

            middle_mcp_x = hand_data["MIDDLE_FINGER_MCP"]["x_rel"]
            middle_mcp_y = hand_data["MIDDLE_FINGER_MCP"]["y_rel"]

            reference_distance = (
                middle_mcp_x ** 2 + middle_mcp_y ** 2
            ) ** 0.5

            if reference_distance > 0:

                for landmark_name, coordinates in hand_data.items():

                    coordinates["x_scaled"] = (
                        coordinates["x_rel"] / reference_distance
                    )

                    coordinates["y_scaled"] = (
                        coordinates["y_rel"] / reference_distance
                    )
            else:
                continue


            # -----------------------------
            # Thumb-index distance
            # -----------------------------

            thumb_index_distance = calculate_thumb_index_distance(hand_data)


            # -----------------------------
            # Pinch detection
            # -----------------------------
            predicted_grasp, grasp_confidence = classify_pinch(thumb_index_distance)

            grasp_text = (f"{hand_label}: {predicted_grasp} "
                        f"({grasp_confidence:.1f}%)")

            cv2.putText(
                frame,
                grasp_text,
                (wrist_x, wrist_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            log_hand_landmarks(
            csv_writer,
            output_path.stem,
            frame_number,
            timestamp_ms,
            num_hands,
            hand_label,
            hand_confidence,
            hand_data,
            mp_hands.HandLandmark,
            thumb_index_distance,
            predicted_grasp,
            grasp_confidence)


            

    cv2.putText(
        frame,
        f"Hands detected: {num_hands}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    

    writer.write(frame)

    cv2.imshow("Camera Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break



# -----------------------------
# Cleanup
# -----------------------------


camera.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

print("Recording stopped.")
print(f"Video saved to: {output_path}")
print(f"Landmark data saved to: {landmark_output_path}")



play_video(output_path)
