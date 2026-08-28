import csv
import cv2
from datetime import datetime
from pathlib import Path
from video_playback import play_video
import mediapipe as mp
import time



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

PINCH_THRESHOLD = 0.55


# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")


# print("Reached camera settings section")

# Request capture settings
camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
camera.set(cv2.CAP_PROP_FPS, FPS)

actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = camera.get(cv2.CAP_PROP_FPS)

print(f"Camera resolution: {actual_width}x{actual_height}")
print(f"Camera FPS: {actual_fps}")

# print("Finished camera settings section")

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


#open CSV file for writing hand landmarks
csv_file = open(
    landmark_output_path,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_file = open(
    landmark_output_path,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

# Write column headings
csv_writer.writerow([
    "session_name",
    "frame_number",
    "timestamp_ms",
    "detected_hand_count",
    "hand_label",
    "hand_confidence",
    "landmark_number",
    "landmark_name",
    "x_norm",
    "y_norm",
    "z_rel",
    "x_px",
    "y_px",
    "x_rel",
    "y_rel",
    "x_scaled",
    "y_scaled"
])

frame_number = 0

start_time = time.perf_counter()

# thumb_index_distances = {
#     "Left": [],
#     "Right": []
# }

thumb_index_distances = {
    "Left": {
        "open": [],
        "pinch": []
    },
    "Right": {
        "open": [],
        "pinch": []
    }
}

grasp_state = None
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

    print(f"Hands detected: {num_hands}")


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


            # -----------------------------
            # Thumb-index distance
            # -----------------------------

            thumb_tip = hand_data["THUMB_TIP"]
            index_tip = hand_data["INDEX_FINGER_TIP"]

            x_distance = (
                thumb_tip["x_scaled"] - index_tip["x_scaled"]
            )

            y_distance = (
                thumb_tip["y_scaled"] - index_tip["y_scaled"]
            )

            thumb_index_distance = (
                x_distance ** 2 + y_distance ** 2
            ) ** 0.5

            UNCERTAIN_MARGIN = 0.05
            PINCH_THRESHOLD = 0.55

            lower_threshold = PINCH_THRESHOLD - UNCERTAIN_MARGIN
            upper_threshold = PINCH_THRESHOLD + UNCERTAIN_MARGIN

            if thumb_index_distance < lower_threshold:
                predicted_grasp = "PINCH"

            elif thumb_index_distance > upper_threshold:
                predicted_grasp = "NOT PINCH"

            else:
                predicted_grasp = "UNCERTAIN"

            # -----------------------------
            # Pinch detection
            # -----------------------------

            # if thumb_index_distance < PINCH_THRESHOLD:
            #     predicted_grasp = "PINCH"
            # else:
            #     predicted_grasp = "NOT PINCH"


            

            if grasp_state is not None:
                thumb_index_distances[hand_label][grasp_state].append(
                    thumb_index_distance
                )

                grasp_text = f"{hand_label}: {predicted_grasp}"

                cv2.putText(
                    frame,
                    grasp_text,
                    (wrist_x, wrist_y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        for landmark_number, landmark_name in enumerate(
            mp_hands.HandLandmark
        ):

            coordinates = hand_data[landmark_name.name]

            csv_writer.writerow([
                output_path.stem,
                frame_number,
                timestamp_ms,
                num_hands,
                hand_label,
                hand_confidence,
                landmark_number,
                landmark_name.name,
                coordinates["x_norm"],
                coordinates["y_norm"],
                coordinates["z_rel"],
                coordinates["x_px"],
                coordinates["y_px"],
                coordinates["x_rel"],
                coordinates["y_rel"],
                coordinates["x_scaled"],
                coordinates["y_scaled"]
            ])


        print(
            "\n\nIndex scaled:",
            hand_data["INDEX_FINGER_TIP"]["x_scaled"],
            hand_data["INDEX_FINGER_TIP"]["y_scaled"]
        )


        print("\n\nPixels:", hand_data["INDEX_FINGER_TIP"]["x_rel"],
                hand_data["INDEX_FINGER_TIP"]["y_rel"])

        print("\n\nScaled:", hand_data["INDEX_FINGER_TIP"]["x_scaled"],
                hand_data["INDEX_FINGER_TIP"]["y_scaled"])
        print("\n\nWrist:", hand_data["WRIST"])
        print("Index tip:", hand_data["INDEX_FINGER_TIP"])
        print(hand_data["INDEX_FINGER_TIP"])



            

    cv2.putText(
        frame,
        f"Hands detected: {num_hands}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if grasp_state is not None:
        cv2.putText(
            frame,
            f"Calibration state: {grasp_state.upper()}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    writer.write(frame)

    cv2.imshow("Camera Capture", frame)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

    key = cv2.waitKey(1) & 0xFF

    if key == ord("o"):
        grasp_state = "open"
        print("Calibration state: OPEN")

    elif key == ord("p"):
        grasp_state = "pinch"
        print("Calibration state: PINCH")

    elif key == ord("q"):
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

print("\nThumb-index distance measurements:")

print("\nThumb-index distance summary:")

for hand, states in thumb_index_distances.items():

    for state, distances in states.items():

        if distances:

            minimum = min(distances)
            maximum = max(distances)
            average = sum(distances) / len(distances)

            print(f"\n{hand.upper()} - {state.upper()}")
            print(f"Samples: {len(distances)}")
            print(f"Minimum: {minimum:.3f}")
            print(f"Average: {average:.3f}")
            print(f"Maximum: {maximum:.3f}")

play_video(output_path)
