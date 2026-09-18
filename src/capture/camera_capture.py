import csv
import cv2
from datetime import datetime
from pathlib import Path
from src.capture.video_playback import play_video
import mediapipe as mp
import time

from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance, 
                                    calculate_finger_angles,calculate_thumb_opposition, calculate_thumb_index_side_distance,
                                    calculate_thumb_fingertip_distances, calculate_hand_center)
from src.grasp.grasp_classifier import (classify_pinch, detect_cylindrical_grasp, detect_hook_grasp, 
                                        detect_large_aperture_palmar_grasp, detect_multidigit_precision_grasp, detect_pincer_grasp, detect_spherical_grasp,
                                        detect_tip_pinch, detect_whole_hand_power_grip, detect_handle_grasp, detect_tripod_grasp,
                                        detect_lateral_pinch)
from src.data.landmark_logger import (create_landmark_logger,log_hand_landmarks)
from src.object_detection.yolo_detector import (detect_objects, draw_detected_objects)
from src.object_detection.hand_object_interaction import (find_nearest_object, is_hand_near_object, 
                                                          count_fingertips_near_object, calculate_object_movement, 
                                                          determine_interaction_state, InteractionStateTracker)
from src.tasks.task_config import get_task
from src.object_detection.yolo_detector import (detect_objects, draw_detected_objects, filter_objects_by_class,)
from src.data.frame_record import create_frame_record
from src.data.interaction_logger import (create_interaction_logger, log_frame_record,)
from src.object_detection.object_detection_tracker import (ObjectDetectionTracker)

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

CURRENT_TASK_ID = "bottle_transport"

current_task = get_task(
    CURRENT_TASK_ID
)

permitted_objects = current_task["permitted_objects"]
expected_grasps = current_task["expected_grasps"]

OUTPUT_DIR = Path("data/raw/video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"{timestamp}_camera_test.mp4"

LANDMARK_DIR = Path("data/raw/landmarks")
LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

landmark_output_path = (
    LANDMARK_DIR / f"{timestamp}_camera_test_landmarks.csv"
)

INTERACTION_DIR = Path("data/raw/interactions")
INTERACTION_DIR.mkdir(parents=True, exist_ok=True)

interaction_output_path = (
    INTERACTION_DIR
    / f"{timestamp}_interaction_data.csv"
)

SESSION_ID = timestamp
TRIAL_ID = "trial_001"
REPETITION_NUMBER = 1


object_detection_tracker = ObjectDetectionTracker(
    lost_after_frames=3,
    uncertain_threshold=0.25
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
print(
    f"Current task: {current_task['task_name']}"
)

print(
    f"Permitted objects: {permitted_objects}"
)

print(
    f"Expected grasps: {expected_grasps}"
)
print(f"Saving to: {output_path}")
print("Press Q to stop recording.")


# open CSV file for writing hand landmarks
csv_file, csv_writer = create_landmark_logger(landmark_output_path)

interaction_csv_file, interaction_csv_writer = (
    create_interaction_logger(
        interaction_output_path
    )
)

frame_number = 0

start_time = time.perf_counter()

previous_object_center = None
object_movement = 0.0

interaction_tracker = InteractionStateTracker(
    required_frames=3
)
# -----------------------------
# Capture loop
# -----------------------------

while True:

    success, frame = camera.read()

    object_movement = 0.0

    if not success:
        print("Failed to capture frame.")
        break


    frame_number += 1
    timestamp_ms = int((time.perf_counter() - start_time) * 1000)

    detected_objects = detect_objects(frame)

    if frame_number % 30 == 0:
        # print(
        #     "YOLO:",
        #     [obj["class_name"] for obj in detected_objects]
        # )

        print(
            "YOLO:",
            [
                (
                    obj["class_name"],
                    round(obj["confidence"], 2)
                )
                for obj in detected_objects
            ]
        )

    if frame_number % 30 == 0:
        print(
            "Permitted:",
            [
                obj["class_name"]
                for obj in permitted_detected_objects
            ]
        )

    permitted_detected_objects, irrelevant_detected_objects = (filter_objects_by_class(detected_objects, permitted_objects))

    object_outcome = object_detection_tracker.update(permitted_detected_objects)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)
    draw_detected_objects(
        frame,
        detected_objects
    )
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

            hand_center = calculate_hand_center(hand_data)

            wrist_x = hand_data["WRIST"]["x_px"]
            wrist_y = hand_data["WRIST"]["y_px"]

            hand_near_object = False
            fingertip_count = 0
            object_name = None
            object_distance = None

            nearest_object, object_distance = find_nearest_object(
                hand_center,
                permitted_detected_objects
            )

            if nearest_object is not None:

                object_name = nearest_object["class_name"]

                # -----------------------------
                # Object movement
                # -----------------------------
                current_object_center = nearest_object["center"]

                object_movement = calculate_object_movement(
                    previous_object_center,
                    current_object_center
                )

                previous_object_center = current_object_center

                # -----------------------------
                # Hand-object proximity
                # -----------------------------
                hand_near_object = is_hand_near_object(
                    hand_center,
                    nearest_object
                )

                # -----------------------------
                # Fingertip-object proximity
                # -----------------------------
                fingertip_count = count_fingertips_near_object(
                    hand_data,
                    nearest_object
                )

                # cv2.putText(
                #     frame,
                #     interaction_text,
                #     (wrist_x, wrist_y - 90),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 255, 0),
                #     2
                # )

            

                # cv2.putText(
                #     frame,
                #     interaction_text,
                #     (wrist_x, wrist_y - 90),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 255, 0),
                #     2
                # )

            # -----------------------------
            # Wrist-relative coordinates
            # -----------------------------

            

            for landmark_name, coordinates in hand_data.items():

                coordinates["x_rel"] = coordinates["x_px"] - wrist_x
                coordinates["y_rel"] = coordinates["y_px"] - wrist_y


            if nearest_object is not None:

                object_name = nearest_object["class_name"]

                cv2.putText(
                    frame,
                    f"Nearest object: {object_name} ({object_distance:.0f}px)",
                    (wrist_x, wrist_y - 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
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



            # -----------------------------
            # Cylindrical grasp detection
            # -----------------------------
            is_cylindrical, cylindrical_confidence = detect_cylindrical_grasp(hand_data)

            grasp_text = (f"{hand_label}: {predicted_grasp} "
                        f"({grasp_confidence:.1f}%)")


            # -----------------------------
            # Spherical grasp detection
            # -----------------------------
            finger_angles = calculate_finger_angles(hand_data)

            thumb_distances = calculate_thumb_opposition(hand_data)

            thumb_middle_distance = thumb_distances["thumb_middle"]

            finger_spread = calculate_finger_spread(hand_data)
            is_spherical, spherical_confidence = detect_spherical_grasp(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance,
                finger_spread
            )



            # -----------------------------
            # Large-aperture palmar grasp detection
            # -----------------------------
            hand_aperture = calculate_hand_aperture(hand_data)
            is_large_palmar, large_palmar_confidence = (
                detect_large_aperture_palmar_grasp(
                    finger_angles,
                    finger_spread,
                    hand_aperture,
                    thumb_index_distance,
                    thumb_middle_distance
                )
            )



            # -----------------------------
            # Whole-hand power grip detection
            # -----------------------------
            is_power_grip, power_grip_confidence = detect_whole_hand_power_grip(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance,
                hand_aperture
            )

            # -----------------------------
            # Handle grasp detection
            # -----------------------------
            is_handle_grasp, handle_grasp_confidence = detect_handle_grasp(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance,
                hand_aperture,
                finger_spread
            )



            # -----------------------------
            # Hook grasp detection
            # -----------------------------
            is_hook_grasp, hook_grasp_confidence = detect_hook_grasp(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance,
                hand_aperture
            )

            # -----------------------------
            # Pincer grasp detection
            # -----------------------------
            is_pincer, pincer_confidence = detect_pincer_grasp(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance
            )

            # -----------------------------
            # Tip pinch detection
            # -----------------------------
            is_tip_pinch, tip_pinch_confidence = detect_tip_pinch(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance
            )



            # -----------------------------
            # Tripod grasp detection
            # -----------------------------
            is_tripod, tripod_confidence = detect_tripod_grasp(
                finger_angles,
                thumb_index_distance,
                thumb_middle_distance
            )



            # -----------------------------
            # Lateral pinch detection
            # -----------------------------
            thumb_index_side_distance = (
                calculate_thumb_index_side_distance(hand_data)
            )
            is_lateral_pinch, lateral_pinch_confidence = detect_lateral_pinch(
                finger_angles,
                thumb_index_distance,
                thumb_index_side_distance
            )




            # -----------------------------
            # Multidigit precision grasp detection
            # -----------------------------
            thumb_fingertip_distances = (
                calculate_thumb_fingertip_distances(hand_data)
            )
            is_multidigit_precision, multidigit_precision_confidence = (
                detect_multidigit_precision_grasp(
                    finger_angles,
                    thumb_fingertip_distances
                )
            )


            grasp_detected = any([
                is_cylindrical,
                is_spherical,
                is_large_palmar,
                is_power_grip,
                is_handle_grasp,
                is_hook_grasp,
                is_pincer,
                is_tip_pinch,
                is_tripod,
                is_lateral_pinch,
                is_multidigit_precision,
            ])

            candidate_interaction_state = determine_interaction_state(
                hand_near_object=hand_near_object,
                fingertips_near_object=fingertip_count,
                grasp_detected=grasp_detected
            )

            interaction_state = interaction_tracker.update(
                candidate_interaction_state
            )

            frame_record = create_frame_record(
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,

                session_id=SESSION_ID,
                trial_id=TRIAL_ID,
                repetition_number=REPETITION_NUMBER,

                task_id=CURRENT_TASK_ID,
                task_name=current_task["task_name"],

                hand_label=hand_label,
                hand_confidence=hand_confidence,
                hand_center=hand_center,

                grasp_detected=grasp_detected,
                interaction_state=interaction_state,

                detected_object=nearest_object,
                object_distance=object_distance,
                object_movement=object_movement,
                fingertip_count=fingertip_count,

                object_outcome=object_outcome,
            )

            log_frame_record(
                interaction_csv_writer,
                frame_record
            )

            if frame_number % 30 == 0:
                print(frame_record)

            if object_name is not None:
                interaction_text = (
                    f"Raw: {candidate_interaction_state} | "
                    f"Stable: {interaction_state}"
                )
            else:
                interaction_text = (
                    f"Raw: {candidate_interaction_state} | "
                    f"Stable: {interaction_state}"
                )

            cv2.putText(
                frame,
                interaction_text,
                (wrist_x, wrist_y - 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            if object_name is not None:

                cv2.putText(
                    frame,
                    f"Object movement: {object_movement:.1f}px",
                    (wrist_x, wrist_y - 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


            if object_name is not None:

                cv2.putText(
                    frame,
                    f"Nearest: {object_name} ({object_distance:.0f}px)",
                    (wrist_x, wrist_y - 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            cv2.putText(
                    frame,
                    f"Object movement: {object_movement:.1f}px",
                    (wrist_x, wrist_y - 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            
            cv2.putText(
                frame,
                interaction_text,
                (wrist_x, wrist_y - 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        
            cv2.putText(
                frame,
                grasp_text,
                (wrist_x, wrist_y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Cylindrical: {cylindrical_confidence:.1f}%",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Spherical: {spherical_confidence:.1f}%",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Large Palmar: {is_large_palmar} ({large_palmar_confidence:.1f}%)",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Power Grip: {is_power_grip} ({power_grip_confidence:.1f}%)",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Handle Grasp: {is_handle_grasp} ({handle_grasp_confidence:.1f}%)",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )



            cv2.putText(
                frame,
                f"Hook Grasp: {is_hook_grasp} ({hook_grasp_confidence:.1f}%)",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )



            cv2.putText(
                frame,
                f"Pincer: {is_pincer} ({pincer_confidence:.1f}%)",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Tip Pinch: {is_tip_pinch} ({tip_pinch_confidence:.1f}%)",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )




            cv2.putText(
                frame,
                f"Tripod: {is_tripod} ({tripod_confidence:.1f}%)",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )



            cv2.putText(
                frame,
                f"Lateral Pinch: {is_lateral_pinch} ({lateral_pinch_confidence:.1f}%)",
                (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )



            cv2.putText(
                frame,
                f"Multidigit Precision: {is_multidigit_precision} "
                f"({multidigit_precision_confidence:.1f}%)",
                (20, 410),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
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


    cv2.putText(
        frame,
        f"Permitted objects: {len(permitted_detected_objects)}",
        (30, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Irrelevant objects: {len(irrelevant_detected_objects)}",
        (30, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
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
interaction_csv_file.close()

print("Recording stopped.")
print(f"Video saved to: {output_path}")
print(f"Landmark data saved to: {landmark_output_path}")
print(
    f"Interaction data saved to: "
    f"{interaction_output_path}"
)


play_video(output_path)
