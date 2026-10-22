import csv
import json
import cv2
from datetime import datetime
from pathlib import Path
from src.capture.video_playback import play_video
import mediapipe as mp
import time
import numpy as np

from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance, 
                                    calculate_finger_angles,calculate_thumb_opposition, calculate_thumb_index_side_distance,
                                    calculate_thumb_fingertip_distances, calculate_hand_center)
from src.grasp.grasp_classifier import (classify_pinch, detect_cylindrical_grasp, detect_hook_grasp, 
                                        detect_large_aperture_palmar_grasp, detect_multidigit_precision_grasp, detect_pincer_grasp, detect_spherical_grasp,
                                        detect_tip_pinch, detect_whole_hand_power_grip, detect_handle_grasp, detect_tripod_grasp,
                                        detect_lateral_pinch)
from src.data.landmark_logger import (create_landmark_logger,log_hand_landmarks,log_no_hand_frame)
from src.object_detection.yolo_detector import (detect_objects, draw_detected_objects)
from src.object_detection.hand_object_interaction import (find_nearest_object, is_hand_near_object, 
                                                          count_fingertips_near_object, determine_interaction_state, InteractionStateTracker)
from src.tasks.task_config import get_task
from src.object_detection.yolo_detector import (detect_objects, draw_detected_objects, filter_objects_by_class,)
from src.data.frame_record import create_frame_record
from src.data.interaction_logger import (create_interaction_logger, log_frame_record, log_no_hand_interaction,)
from src.object_detection.object_detection_tracker import (ObjectDetectionTracker)
from src.object_detection.object_tracker import (track_objects, ObjectIdentityTracker,)
from src.data.object_logger import (create_object_logger, log_detected_objects,)
from src.object_detection.object_movement_tracker import ObjectMovementTracker
from src.object_detection.duplicate_filter import filter_duplicate_objects
from src.object_detection.object_segmentation import ObjectSegmenter
from src.object_detection.geometry_association import attach_object_geometry
from src.object_detection.interaction_event_tracker import (InteractionEventTracker,)
from src.pose.pose_detector import UpperBodyPoseDetector
from src.pose.pose_logger import create_pose_logger, log_pose_frame
from src.pose.joint_angles import calculate_upper_body_angles
from src.assessment.game_trajectory import GameTrajectory
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

CAMERA_INDEX = 0  # Updated below from session configuration
WIDTH = 1280
HEIGHT = 720
FPS = 30

from src.platform_config.camera_bridge import load_camera_session

SESSION_CONFIG, current_task = load_camera_session()
CURRENT_TASK_ID = current_task["task_id"]
CAMERA_INDEX = SESSION_CONFIG.camera_index
WIDTH = SESSION_CONFIG.frame_width
HEIGHT = SESSION_CONFIG.frame_height

# -----------------------------
# Trajectory testing mode
# -----------------------------

TRAJECTORY_MODE_ENABLED = True

trajectory_assessment = GameTrajectory(
    object_name=SESSION_CONFIG.object_id,
    grace_seconds=3.0,
    start_point=(250, 400),
    target_center=(1000, 400),
    target_radius=70,
)

trajectory_assessment.set_preset(
    SESSION_CONFIG.trajectory,
    frame_width=SESSION_CONFIG.frame_width,
    frame_height=SESSION_CONFIG.frame_height,
)

# Task configuration is supplied by the platform bridge.

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

SESSION_ID = SESSION_CONFIG.session_id
TRIAL_ID = "trial_001"
REPETITION_NUMBER = 1

# Trajectory session state
TRAJECTORY_TARGET_REPETITIONS = SESSION_CONFIG.target_repetitions
trajectory_successful_repetitions = 0
trajectory_attempt_history = []
trajectory_session_complete = False
trajectory_next_start_ready_at = 0.0
trajectory_attempt_start_ms = None

trajectory_results_path = (
    Path("data")
    / "trajectory_results"
    / f"{SESSION_ID}_trajectory_results.json"
)


object_detection_tracker = ObjectDetectionTracker(
    lost_after_frames=90,
    uncertain_threshold=0.25
)

identity_tracker = ObjectIdentityTracker(
    max_missing_frames=90,
    max_match_distance=100,
)

movement_tracker = ObjectMovementTracker()

# Experimental object-geometry settings
SEGMENTATION_ENABLED = False

segmenter = None

if SEGMENTATION_ENABLED:
    segmenter = ObjectSegmenter(confidence=0.25)

# Create an independent object-detection log
OBJECT_LOG_DIR = Path("data/raw/objects")
OBJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)

object_csv_path = (
    OBJECT_LOG_DIR
    / f"{SESSION_ID}_object_data.csv"
)

object_csv_file, object_csv_writer = create_object_logger(
    object_csv_path
)

# Create a separate CSV for upper-body pose landmarks.
pose_output_dir = Path("data/raw/pose")
pose_output_dir.mkdir(parents=True, exist_ok=True)

pose_output_path = pose_output_dir / f"{SESSION_ID}_pose_landmarks.csv"

pose_csv_file, pose_csv_writer = create_pose_logger(pose_output_path)


pose_detector = UpperBodyPoseDetector()

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

interaction_trackers = {
    "Left": InteractionStateTracker(required_frames=3),
    "Right": InteractionStateTracker(required_frames=3),
}

interaction_event_tracker = InteractionEventTracker(
    manipulation_threshold=2.0,
    transport_threshold=10.0,
)

last_associated_object_ids = {
    "Left": None,
    "Right": None,
}

# Track consecutive frames without a valid hand-object association.
association_missing_frames = {
    "Left": 0,
    "Right": 0,
}

ASSOCIATION_LOSS_RESET_FRAMES = 5

# Display-only memory: does not affect event detection or CSV logging.
display_events = {
    "Left": None,
    "Right": None,
}

HAND_LOSS_RESET_FRAMES = 5

hand_missing_frames = {
    "Left": 0,
    "Right": 0,
}

# -----------------------------
# Capture loop
# -----------------------------


try:
    perf_totals = {
        "pose": 0.0,
        "yolo": 0.0,
        "segmentation": 0.0,
        "hands": 0.0,
        "frame": 0.0,
    }
    perf_count = 0

    while True:

        perf_frame_start = time.perf_counter()

        success, frame = camera.read()

        if not success:
            print("Failed to capture frame.")
            break

        # Mirror the camera horizontally for intuitive user interaction.
        # Apply before detection, tracking, and trajectory rendering
        # so all components share the same coordinate system.
        frame = cv2.flip(frame, 1)

        # Detect upper-body landmarks in the current camera frame.
        perf_start = time.perf_counter()
        upper_body_landmarks = pose_detector.detect(frame)
        perf_totals["pose"] += time.perf_counter() - perf_start

        # Calculate preliminary 2D joint angles from the detected pose.
        upper_body_angles = calculate_upper_body_angles(
            upper_body_landmarks
        )


        # Draw the upper-body skeleton when pose landmarks are available.
        if upper_body_landmarks is not None:
            frame_height, frame_width = frame.shape[:2]

            # Connect the shoulders, arms, and torso.
            upper_body_connections = [
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
            ]

            for start_name, end_name in upper_body_connections:
                start = upper_body_landmarks[start_name]
                end = upper_body_landmarks[end_name]

                # Do not draw a connection if either landmark is poorly visible.
                if start["visibility"] < 0.5 or end["visibility"] < 0.5:
                    continue

                start_point = (
                    int(start["x"] * frame_width),
                    int(start["y"] * frame_height),
                )
                end_point = (
                    int(end["x"] * frame_width),
                    int(end["y"] * frame_height),
                )

                pass  # Game UI: drawing disabled
                # GAME UI HIDDEN: cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

            # Draw a circle at each sufficiently visible landmark.
            for landmark in upper_body_landmarks.values():
                if landmark["visibility"] < 0.5:
                    continue

                point = (
                    int(landmark["x"] * frame_width),
                    int(landmark["y"] * frame_height),
                )

                pass  # Game UI: drawing disabled
                # GAME UI HIDDEN: cv2.circle(frame, point, 5, (255, 0, 255), -1)


                # Display the calculated elbow and shoulder angles.
                angle_labels = [
                    ("Left elbow", upper_body_angles["left_elbow"]),
                    ("Right elbow", upper_body_angles["right_elbow"]),
                    ("Left shoulder", upper_body_angles["left_shoulder"]),
                    ("Right shoulder", upper_body_angles["right_shoulder"]),
                ]

                for index, (joint_name, angle) in enumerate(angle_labels):
                    angle_text = (
                        f"{joint_name}: {angle:.1f} deg"
                        if angle is not None
                        else f"{joint_name}: N/A"
                    )

                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.putText(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: angle_text,
                        # GAME UI HIDDEN: (20, 350 + index * 25),
                        # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                        # GAME UI HIDDEN: 0.55,
                        # GAME UI HIDDEN: (255, 255, 0),
                        # GAME UI HIDDEN: 2,
                    # GAME UI HIDDEN: )

        frame_number += 1
        timestamp_ms = int((time.perf_counter() - start_time) * 1000)

        # Whether the active trajectory object has a valid
        # hand-object interaction in the current frame.
        trajectory_interaction_active = False

        # Save upper-body landmarks using the current frame's timestamp.
        log_pose_frame(
            writer=pose_csv_writer,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            upper_body_landmarks=upper_body_landmarks,
            upper_body_angles=upper_body_angles,
        )

        # Detect objects using YOLO.
        perf_start = time.perf_counter()
        raw_detected_objects = track_objects(frame)
        perf_totals["yolo"] += time.perf_counter() - perf_start

        # TEMPORARY DIAGNOSTIC: Check whether YOLO detects the bottle
        # before duplicate filtering and application-level identity tracking.

        if frame_number % 15 == 0:
            raw_bottles = [
                obj for obj in raw_detected_objects
                if obj["class_name"] == "bottle"
            ]

            print(
                f"[DETECTION CHECK] Frame {frame_number} | "
                f"Raw bottles: {len(raw_bottles)} | "
                f"YOLO IDs: {[obj.get('track_id') for obj in raw_bottles]} | "
                f"Confidence: {[round(obj['confidence'], 2) for obj in raw_bottles]}"
            )

        # Flag near-identical detections before assigning application IDs.
        detected_objects, probable_duplicates = filter_duplicate_objects(
            raw_detected_objects,
            iou_threshold=0.90
        )

        # Assign application-level IDs to the retained detections.
        detected_objects = identity_tracker.update(detected_objects)

        # Temporary diagnostic: inspect application-level object identity.
        for obj in detected_objects:
            if obj["class_name"] == "bottle":
                print(
                    f"[IDENTITY DEBUG] "
                    f"Frame: {frame_number} | "
                    f"YOLO ID: {obj.get('track_id')} | "
                    f"Object ID: {obj.get('object_id')} | "
                    f"Status: {obj.get('identity_status')} | "
                    f"Centre: {obj.get('center')}"
                )

        # -------------------------------------------------
        # Experimental object segmentation and geometry
        # -------------------------------------------------

        if SEGMENTATION_ENABLED and permitted_objects:

            perf_start = time.perf_counter()
            segmented_objects = segmenter.segment_objects(
                frame,
                allowed_classes=permitted_objects,
            )
            perf_totals["segmentation"] += (
                time.perf_counter() - perf_start
            )

            detected_objects = attach_object_geometry(
                tracked_objects=detected_objects,
                segmented_objects=segmented_objects,
                min_iou=0.30,
            )

        # -------------------------------------------------
        # Per-object movement
        # -------------------------------------------------

        object_movements = movement_tracker.update(detected_objects)

        # Log object detections independently of hand detection
        log_detected_objects(
            writer=object_csv_writer,
            detected_objects=detected_objects,
            session_id=SESSION_ID,
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            task_id=CURRENT_TASK_ID,
            task_name=current_task["task_name"],
        )

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

        permitted_detected_objects, irrelevant_detected_objects = (filter_objects_by_class(detected_objects, permitted_objects))

       
        # Temporary tracking-ID diagnostic
        if frame_number % 30 == 0:
            print(
                "TRACKING DEBUG:",
                [
                    {
                        "class": obj["class_name"],
                        "track_id": obj.get("track_id"),
                        "status": obj.get("tracking_status")
                    }
                    for obj in permitted_detected_objects
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


        if TRAJECTORY_MODE_ENABLED and not trajectory_session_complete:
            trajectory_assessment.prepare_retry()

            # Preserve the completed trajectory briefly
            # before preparing the next repetition.
            if (
                trajectory_assessment.outcome == "COMPLETED"
                and time.monotonic() >= trajectory_next_start_ready_at
            ):
                trajectory_assessment.reset()

        # Start confirmation uses the detected object,
        # independently of grasp recognition.
        if (
            TRAJECTORY_MODE_ENABLED
            and not trajectory_session_complete
            and not trajectory_assessment.attempt_active
            and trajectory_assessment.outcome == "WAITING"
        ):
            start_candidates = [
                obj for obj in permitted_detected_objects
                if obj.get("center") is not None
                and obj.get("object_id") is not None
            ]

            if len(start_candidates) == 1:
                candidate = start_candidates[0]
                was_active = trajectory_assessment.attempt_active

                trajectory_assessment.consider_start(
                    object_id=candidate["object_id"],
                    object_center=candidate["center"],
                )

                if (
                    not was_active
                    and trajectory_assessment.attempt_active
                ):
                    trajectory_attempt_start_ms = timestamp_ms
            else:
                trajectory_assessment.clear_start_confirmation()

        object_outcome = object_detection_tracker.update(permitted_detected_objects)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        perf_start = time.perf_counter()
        results = hands.process(rgb_frame)
        perf_totals["hands"] += time.perf_counter() - perf_start
        draw_detected_objects(
            frame,
            detected_objects
        )

        # Display YOLO and application-level object IDs
        for tracked_object in permitted_detected_objects:
            yolo_id = tracked_object.get("track_id")
            object_id = tracked_object.get("object_id")

            x1, y1, x2, y2 = tracked_object["bounding_box"]

            pass  # Game UI: drawing disabled
            # GAME UI HIDDEN: cv2.putText(
                # GAME UI HIDDEN: frame,
                # GAME UI HIDDEN: f"YOLO: {yolo_id} | Object: {object_id}",
                # GAME UI HIDDEN: (x1 + 5, y1 + 25),
                # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                # GAME UI HIDDEN: 0.6,
                # GAME UI HIDDEN: (255, 0, 0),
                # GAME UI HIDDEN: 2
            # GAME UI HIDDEN: )


            # Draw segmentation geometry for this tracked object.
            geometry = tracked_object.get("geometry")

            if geometry is not None:
                polygon = geometry.get("polygon")
                oriented_box = geometry.get("oriented_box")

                # Green: estimated visible object contour.
                if polygon is not None:
                    contour_points = np.rint(polygon).astype(np.int32)
                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.polylines(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: [contour_points],
                        # GAME UI HIDDEN: isClosed=True,
                        # GAME UI HIDDEN: color=(0, 255, 0),
                        # GAME UI HIDDEN: thickness=2,
                    # GAME UI HIDDEN: )

                # Red: rotated bounding box.
                if oriented_box is not None:
                    box_points = np.asarray(
                        oriented_box,
                        dtype=np.int32,
                    )

                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.polylines(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: [box_points],
                        # GAME UI HIDDEN: isClosed=True,
                        # GAME UI HIDDEN: color=(0, 0, 255),
                        # GAME UI HIDDEN: thickness=2,
                    # GAME UI HIDDEN: )

                # Display segmentation-derived object measurements.
                angle = geometry["orientation_deg"]
                major_axis = geometry["major_axis_px"]
                minor_axis = geometry["minor_axis_px"]
                aspect_ratio = geometry["aspect_ratio"]

                if geometry["orientation_reliable"]:
                    angle_text = f"{angle:.1f} deg"
                else:
                    angle_text = "Uncertain"

                geometry_labels = [
                    f"Orientation: {angle_text}",
                    f"Major axis: {major_axis:.1f} px",
                    f"Minor axis: {minor_axis:.1f} px",
                    f"Aspect ratio: {aspect_ratio:.2f}",
                ]

                # Position the labels beside the object's standard bounding box.
                label_x = min(x2 + 10, frame.shape[1] - 230)
                label_x = max(5, label_x)

                label_y = max(25, y1 + 20)

                for label_index, label_text in enumerate(geometry_labels):
                    text_y = label_y + (label_index * 23)

                    if text_y >= frame.shape[0]:
                        break

                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.putText(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: label_text,
                        # GAME UI HIDDEN: (label_x, text_y),
                        # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                        # GAME UI HIDDEN: 0.5,
                        # GAME UI HIDDEN: (0, 255, 255),
                        # GAME UI HIDDEN: 2,
                        # GAME UI HIDDEN: cv2.LINE_AA,
                    # GAME UI HIDDEN: )

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
        else:
            num_hands = 0

        # Identify which hands were detected in this frame.
        detected_hand_labels = set()

        if results.multi_hand_landmarks:
            for handedness in results.multi_handedness:
                hand_label = handedness.classification[0].label

                # Apply the same handedness correction as the hand loop.
                if hand_label == "Left":
                    hand_label = "Right"
                else:
                    hand_label = "Left"

                detected_hand_labels.add(hand_label)

        # Update absence counters independently for each hand.
        for hand_label in hand_missing_frames:

            if hand_label in detected_hand_labels:
                hand_missing_frames[hand_label] = 0

            else:
                hand_missing_frames[hand_label] += 1
                association_missing_frames[hand_label] += 1

                # A brief missing-hand observation is not a confirmed release.
                # Keep the previous event history temporarily, but do not
                # update it while the hand is not visible.

                if hand_missing_frames[hand_label] == HAND_LOSS_RESET_FRAMES:
                    association_missing_frames[hand_label] = ASSOCIATION_LOSS_RESET_FRAMES

                    # The interruption has lasted too long to preserve
                    # the previous interaction sequence.
                    interaction_event_tracker.mark_unobserved(hand_label)

                    last_associated_object_ids[hand_label] = None

                    interaction_trackers[hand_label].reset()


        if num_hands == 0:

            log_no_hand_frame(
                csv_writer,
                output_path.stem,
                frame_number,
                timestamp_ms
            )

            log_no_hand_interaction(
                writer=interaction_csv_writer,
                frame_number=frame_number,
                timestamp_ms=timestamp_ms,
                session_id=SESSION_ID,
                trial_id=TRIAL_ID,
                repetition_number=REPETITION_NUMBER,
                task_id=CURRENT_TASK_ID,
                task_name=current_task["task_name"],
                object_outcome=object_outcome
            )



        if results.multi_hand_landmarks:

            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

                handedness = results.multi_handedness[hand_index]

                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                if hand_label == "Left":
                    hand_label = "Right"
                else:
                    hand_label = "Left"

                hand_missing_frames[hand_label] = 0

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
                object_movement = 0.0

                nearest_object, object_distance = find_nearest_object(
                    hand_center,
                    permitted_detected_objects
                )

                # A nearby detection is only a candidate for hand-object association.
                # Do not treat a distant object as the hand's interaction target.
                associated_object = None

                if nearest_object is not None:
                    palm_near_object = is_hand_near_object(
                        hand_center,
                        nearest_object
                    )

                    candidate_fingertip_count = count_fingertips_near_object(
                        hand_data,
                        nearest_object
                    )

                    if palm_near_object or candidate_fingertip_count >= 2:
                        associated_object = nearest_object

                if associated_object is not None:
                    object_name = associated_object["class_name"]

                    associated_object_id = associated_object.get("object_id")

                    object_movement = object_movements.get(
                        associated_object_id,
                        0.0
                    )

                    # -----------------------------
                    # Hand-object proximity
                    # -----------------------------
                    hand_near_object = is_hand_near_object(
                        hand_center,
                        associated_object
                    )

                    # Keep your existing fingertip and grasp calculations
                    # below this line, inside the same if block.
                    # -----------------------------
                    # Hand-object proximity
                    # -----------------------------
                    hand_near_object = (
                        is_hand_near_object(hand_center, nearest_object)
                        or candidate_fingertip_count >= 2
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


                # if nearest_object is not None:
                #     cv2.putText(
                #         frame,
                #         f"Nearest object: {object_name} ({object_distance:.0f}px)",
                #         (wrist_x, wrist_y - 90),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.6,
                #         (255, 255, 0),
                #         2
                #     )
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

                # grasp_text = (f"{hand_label}: {predicted_grasp} "
                #             f"({grasp_confidence:.1f}%)")


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

                # -----------------------------------------
                # Preliminary unified grasp classification
                # -----------------------------------------

                grasp_candidates = {
                    "cylindrical": is_cylindrical,
                    "spherical": is_spherical,
                    "large_palmar": is_large_palmar,
                    "whole_hand_power": is_power_grip,
                    "handle": is_handle_grasp,
                    "hook": is_hook_grasp,
                    "pincer": is_pincer,
                    "tip_pinch": is_tip_pinch,
                    "tripod": is_tripod,
                    "lateral_pinch": is_lateral_pinch,
                    "multidigit_precision": is_multidigit_precision,
                }

                grasp_confidences = {
                    "cylindrical": cylindrical_confidence,
                    "spherical": spherical_confidence,
                    "large_palmar": large_palmar_confidence,
                    "whole_hand_power": power_grip_confidence,
                    "handle": handle_grasp_confidence,
                    "hook": hook_grasp_confidence,
                    "pincer": pincer_confidence,
                    "tip_pinch": tip_pinch_confidence,
                    "tripod": tripod_confidence,
                    "lateral_pinch": lateral_pinch_confidence,
                    "multidigit_precision": multidigit_precision_confidence,
                }

                detected_grasp_types = [
                    grasp_name
                    for grasp_name, is_detected in grasp_candidates.items()
                    if is_detected
                ]

                if len(detected_grasp_types) == 1:
                    detected_grasp_type = detected_grasp_types[0]

                elif len(detected_grasp_types) > 1:
                    detected_grasp_type = max(
                        detected_grasp_types,
                        key=lambda grasp: grasp_confidences[grasp]
                    )

                else:
                    detected_grasp_type = "UNKNOWN"

                # Display the preliminary grasp classification for this hand.
                candidate_text = ", ".join(detected_grasp_types) or "none"

                # cv2.putText(
                #     frame,
                #     f"{hand_label} grasp: {detected_grasp_type}",
                #     (20, 180 if hand_label == "Left" else 250),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (255, 255, 255),
                #     2,
                # )

                # cv2.putText(
                #     frame,
                #     f"Candidates: {candidate_text[:75]}",
                #     (20, 205 if hand_label == "Left" else 275),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.45,
                #     (255, 255, 255),
                #     1,
                # )

                # -----------------------------------------
                # Preliminary task-specific grasp assessment
                # -----------------------------------------

                associated_object_class = (
                    associated_object.get("class_name")
                    if associated_object is not None
                    else None
                )

                if (
                    associated_object is None
                    or not hand_near_object
                    or associated_object_class not in permitted_objects
                ):
                    grasp_appropriateness = "NOT_ASSESSED"

                elif detected_grasp_type in ("UNKNOWN", "AMBIGUOUS"):
                    grasp_appropriateness = "UNCERTAIN"

                elif detected_grasp_type in expected_grasps:
                    grasp_appropriateness = "APPROPRIATE"

                else:
                    grasp_appropriateness = "INAPPROPRIATE"

                # cv2.putText(
                #     frame,
                #     f"{hand_label} appropriateness: {grasp_appropriateness}",
                #     (20, 225 if hand_label == "Left" else 295),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.55,
                #     (255, 255, 255),
                #     2,
                # )

                candidate_interaction_state = determine_interaction_state(
                    hand_near_object=hand_near_object,
                    fingertips_near_object=fingertip_count,
                    grasp_detected=grasp_detected
                )

                interaction_state = interaction_trackers[hand_label].update(
                    candidate_interaction_state
                )

                # Temporary diagnostic: report changes in the grasp state.
                if interaction_state == "GRASPING":
                    print(
                        f"[GRASP DEBUG] "
                        f"Hand: {hand_label} | "
                        f"State: {interaction_state} | "
                        f"Object ID: "
                        f"{associated_object.get('object_id') if associated_object else None} | "
                        f"Grasp detected: {grasp_detected} | "
                        f"Fingertips near object: {fingertip_count} | "
                        f"Hand near object: {hand_near_object}"
                    )

                # Generate temporal events for the associated hand-object pair.
                interaction_event = None

                if associated_object is not None:
                    association_missing_frames[hand_label] = 0

                    current_object_id = associated_object.get("object_id")
                    previous_object_id = last_associated_object_ids[hand_label]

                    # A different associated object starts a new observed
                    # interaction sequence for this hand.
                    if (
                        previous_object_id is not None
                        and previous_object_id != current_object_id
                    ):
                        interaction_event_tracker.mark_unobserved(hand_label)

                    if current_object_id is not None:
                        interaction_event = interaction_event_tracker.update(
                            hand_label=hand_label,
                            object_id=current_object_id,
                            interaction_state=interaction_state,
                            object_movement=object_movement,
                            timestamp_ms=timestamp_ms,
                        )
                    else:
                        interaction_event_tracker.mark_unobserved(hand_label)

                    last_associated_object_ids[hand_label] = current_object_id

                else:
                    # No valid hand-object association in this frame.
                    association_missing_frames[hand_label] += 1

                    previous_object_id = last_associated_object_ids[hand_label]

                    # Missing object association is not evidence of release.
                    # Preserve event history during brief detection gaps.
                    # Do not generate RELEASE until an observed transition
                    # occurs while the object is associated.

                    # An extended association interruption is unknown,
                    # not a confirmed release.
                    if (
                        association_missing_frames[hand_label]
                        >= ASSOCIATION_LOSS_RESET_FRAMES
                    ):
                        interaction_event_tracker.mark_unobserved(
                            hand_label
                        )

                        last_associated_object_ids[hand_label] = None


                # -------------------------------------------------
                # Trajectory attempt control
                # -------------------------------------------------

                if TRAJECTORY_MODE_ENABLED and associated_object is not None:

                    current_object_id = associated_object.get("object_id")

                    transport_states = {
                        "GRASPING",
                        "HOLDING",
                        "MANIPULATION",
                        "TRANSPORT",
                    }

                    if (
                        current_object_id is not None
                        and interaction_state in transport_states
                    ):

                        if (
                            trajectory_assessment.attempt_active
                            and current_object_id
                            == trajectory_assessment.active_object_id
                        ):
                            trajectory_interaction_active = True

                    
                   

                # Remember the latest detected event for the video overlay.
                # This does not affect event detection or CSV logging.
                if interaction_event is not None:
                    display_events[hand_label] = interaction_event

                # Duration is available only when an observed release occurs.
                grasp_duration_ms = None

                if interaction_event == "RELEASE":
                    object_id = (
                        associated_object.get("object_id")
                        if associated_object is not None
                        else last_associated_object_ids[hand_label]
                    )

                    if object_id is not None:
                        history = interaction_event_tracker.pair_history.get(
                            (hand_label, object_id)
                        )

                        if history is not None:
                            grasp_duration_ms = history.get(
                                "last_grasp_duration_ms"
                            )

                                # -----------------------------------------
                # Live hand-object interaction overlay
                # -----------------------------------------
                associated_object_id = (
                    associated_object.get("object_id")
                    if associated_object is not None
                    else None
                )

                # Retrieve the latest event for the current hand-object pair.
                latest_event = None

                if associated_object_id is not None:
                    event_history = interaction_event_tracker.pair_history.get(
                        (hand_label, associated_object_id)
                    )

                    if event_history is not None:
                        latest_event = event_history.get("last_event")

                # -----------------------------------------
                # Live observed grasp-duration display
                # -----------------------------------------
                grasp_timer_text = "-"

                if associated_object_id is not None:
                    history = interaction_event_tracker.pair_history.get(
                        (hand_label, associated_object_id)
                    )

                    if history is not None:
                        grasp_start_ms = history.get("grasp_start_ms")

                        if (
                            interaction_state == "GRASPING"
                            and grasp_start_ms is not None
                        ):
                            elapsed_ms = max(
                                0,
                                timestamp_ms - grasp_start_ms
                            )

                            grasp_timer_text = f"{elapsed_ms / 1000:.1f} s"

                # Display the completed duration on the release frame.
                if grasp_duration_ms is not None:
                    grasp_timer_text = f"{grasp_duration_ms / 1000:.1f} s (released)"

                interaction_labels = [
                    f"{hand_label} hand | Object: {associated_object_id}",
                    f"State: {interaction_state}",
                    f"Latest event: {display_events[hand_label] or '-'}",
                    f"Grasp duration: {grasp_timer_text}",
                ]

                # Put each hand's information in a separate area.
                panel_x = 350 if hand_label == "Left" else frame.shape[1] - 340
                panel_y = 35

                for label_index, label_text in enumerate(interaction_labels):
                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.putText(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: label_text,
                        # GAME UI HIDDEN: (panel_x, panel_y + label_index * 24),
                        # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                        # GAME UI HIDDEN: 0.55,
                        # GAME UI HIDDEN: (255, 255, 255),
                        # GAME UI HIDDEN: 2,
                        # GAME UI HIDDEN: cv2.LINE_AA,
                    # GAME UI HIDDEN: )
                
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

                    detected_object=associated_object,
                    object_distance=object_distance,
                    object_movement=object_movement,
                    fingertip_count=fingertip_count,

                    object_outcome=object_outcome,
                    interaction_event=interaction_event,

                    grasp_duration_ms=grasp_duration_ms,
                    detected_grasp_type=detected_grasp_type,
                    detected_grasp_types=detected_grasp_types,
                    grasp_appropriateness=grasp_appropriateness,
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

                pass  # Game UI: drawing disabled
                # GAME UI HIDDEN: cv2.putText(
                    # GAME UI HIDDEN: frame,
                    # GAME UI HIDDEN: interaction_text,
                    # GAME UI HIDDEN: (wrist_x, wrist_y - 240),
                    # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                    # GAME UI HIDDEN: 0.6,
                    # GAME UI HIDDEN: (255, 255, 0),
                    # GAME UI HIDDEN: 2
                # GAME UI HIDDEN: )

                if object_name is not None:

                    pass  # Game UI: drawing disabled
                    # GAME UI HIDDEN: cv2.putText(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: f"Object movement: {object_movement:.1f}px",
                        # GAME UI HIDDEN: (wrist_x, wrist_y - 150),
                        # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                        # GAME UI HIDDEN: 0.6,
                        # GAME UI HIDDEN: (255, 255, 0),
                        # GAME UI HIDDEN: 2
                    # GAME UI HIDDEN: )


                # if object_name is not None:

                #     cv2.putText(
                #         frame,
                #         f"Nearest: {object_name} ({object_distance:.0f}px)",
                #         (wrist_x, wrist_y - 120),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.6,
                #         (255, 255, 0),
                #         2
                #     )

                pass  # Game UI: drawing disabled
                # GAME UI HIDDEN: cv2.putText(
                        # GAME UI HIDDEN: frame,
                        # GAME UI HIDDEN: f"Object movement: {object_movement:.1f}px",
                        # GAME UI HIDDEN: (wrist_x, wrist_y - 180),
                        # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                        # GAME UI HIDDEN: 0.6,
                        # GAME UI HIDDEN: (255, 255, 0),
                        # GAME UI HIDDEN: 2
                    # GAME UI HIDDEN: )

                
                pass  # Game UI: drawing disabled
                # GAME UI HIDDEN: cv2.putText(
                    # GAME UI HIDDEN: frame,
                    # GAME UI HIDDEN: interaction_text,
                    # GAME UI HIDDEN: (wrist_x, wrist_y - 210),
                    # GAME UI HIDDEN: cv2.FONT_HERSHEY_SIMPLEX,
                    # GAME UI HIDDEN: 0.6,
                    # GAME UI HIDDEN: (255, 255, 0),
                    # GAME UI HIDDEN: 2
                # GAME UI HIDDEN: )
            
                # cv2.putText(
                #     frame,
                #     grasp_text,
                #     (wrist_x, wrist_y - 60),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2
                # )

                # cv2.putText(
                #     frame,
                #     f"Cylindrical: {cylindrical_confidence:.1f}%",
                #     (20, 170),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )

                # cv2.putText(
                #     frame,
                #     f"Spherical: {spherical_confidence:.1f}%",
                #     (20, 110),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )


                # cv2.putText(
                #     frame,
                #     f"Large Palmar: {is_large_palmar} ({large_palmar_confidence:.1f}%)",
                #     (20, 140),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )


                # cv2.putText(
                #     frame,
                #     f"Power Grip: {is_power_grip} ({power_grip_confidence:.1f}%)",
                #     (20, 200),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )

                # cv2.putText(
                #     frame,
                #     f"Handle Grasp: {is_handle_grasp} ({handle_grasp_confidence:.1f}%)",
                #     (20, 230),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )



                # cv2.putText(
                #     frame,
                #     f"Hook Grasp: {is_hook_grasp} ({hook_grasp_confidence:.1f}%)",
                #     (20, 260),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )



                # cv2.putText(
                #     frame,
                #     f"Pincer: {is_pincer} ({pincer_confidence:.1f}%)",
                #     (20, 290),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )


                # cv2.putText(
                #     frame,
                #     f"Tip Pinch: {is_tip_pinch} ({tip_pinch_confidence:.1f}%)",
                #     (20, 320),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )




                # cv2.putText(
                #     frame,
                #     f"Tripod: {is_tripod} ({tripod_confidence:.1f}%)",
                #     (20, 350),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )



                # cv2.putText(
                #     frame,
                #     f"Lateral Pinch: {is_lateral_pinch} ({lateral_pinch_confidence:.1f}%)",
                #     (20, 380),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )



                # cv2.putText(
                #     frame,
                #     f"Multidigit Precision: {is_multidigit_precision} "
                #     f"({multidigit_precision_confidence:.1f}%)",
                #     (20, 410),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (255, 255, 255),
                #     2
                # )

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


                

        # cv2.putText(
        #     frame,
        #     f"Hands detected: {num_hands}",
        #     (30, 80),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2
        # )


        # cv2.putText(
        #     frame,
        #     f"Permitted objects: {len(permitted_detected_objects)}",
        #     (30, 450),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (255, 255, 255),
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"Irrelevant objects: {len(irrelevant_detected_objects)}",
        #     (30, 480),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (255, 255, 255),
        #     2
        # )

        # Reset start confirmation when the candidate
        # object disappears or leaves the start region.
        if (
            TRAJECTORY_MODE_ENABLED
            and not trajectory_assessment.attempt_active
            and trajectory_assessment.outcome == "WAITING"
        ):
            candidate_id = trajectory_assessment.start_candidate_id

            if candidate_id is not None:
                candidate_visible = any(
                    obj.get("object_id") == candidate_id
                    for obj in permitted_detected_objects
                )

                if not candidate_visible:
                    trajectory_assessment.clear_start_confirmation()

        # -------------------------------------------------
        # Track the active game object once per frame
        # -------------------------------------------------

        if TRAJECTORY_MODE_ENABLED and trajectory_assessment.attempt_active:

            trajectory_object = trajectory_assessment.recover_object(
                permitted_detected_objects
            )

            object_visible = trajectory_object is not None

            if object_visible:
                trajectory_center = trajectory_object.get("center")

                if trajectory_center is not None:
                    trajectory_assessment.update(
                        trajectory_center,
                        object_id=trajectory_object.get("object_id"),
                        timestamp_ms=timestamp_ms,
                    )

            attempt_ended = trajectory_assessment.update_attempt_state(
                interaction_active=trajectory_interaction_active,
                object_visible=object_visible,
            )

            if attempt_ended:
                trajectory_metrics = trajectory_assessment.get_metrics()

                print(
                    "\nTrajectory attempt ended:",
                    trajectory_assessment.outcome,
                )

                for metric_name, metric_value in trajectory_metrics.items():
                    print(f"{metric_name}: {metric_value}")

                attempt_outcome = trajectory_assessment.outcome

                attempt_end_ms = timestamp_ms

                attempt_duration_ms = (
                    attempt_end_ms - trajectory_attempt_start_ms
                    if trajectory_attempt_start_ms is not None
                    else None
                )

                attempt_record = {
                    "attempt_number": len(trajectory_attempt_history) + 1,
                    "repetition_number": REPETITION_NUMBER,
                    "outcome": attempt_outcome,
                    "start_timestamp_ms": trajectory_attempt_start_ms,
                    "end_timestamp_ms": attempt_end_ms,
                    "duration_ms": attempt_duration_ms,
                    "progress_percent": round(
                        trajectory_assessment.progress_fraction * 100, 2
                    ),
                    "metrics": trajectory_metrics,
                    "trajectory_points": [
                        list(point)
                        for point in trajectory_assessment.actual_path
                    ],
                    "trajectory_samples": [
                        {
                            "timestamp_ms": sample.get("timestamp_ms"),
                            "object_id": sample.get("object_id"),
                            "center": list(sample["center"]),
                        }
                        for sample in trajectory_assessment.trajectory_samples
                    ],
                    "recorded_at": datetime.now().isoformat(),
                    "trajectory_config": {
                        "preset": getattr(
                            trajectory_assessment,
                            "preset_name",
                            SESSION_CONFIG.trajectory,
                        ),
                        "start_point": list(
                            trajectory_assessment.start_point
                        ),
                        "target_center": list(
                            trajectory_assessment.target_center
                        ),
                        "target_radius": trajectory_assessment.target_radius,
                        "waypoints": [
                            list(point)
                            for point in trajectory_assessment.waypoints
                        ],
                        "desired_path": [
                            list(point)
                            for point in trajectory_assessment.desired_path
                        ],
                        "coordinate_system": "image_pixels",
                    },
                }

                trajectory_attempt_history.append(attempt_record)

                if attempt_outcome == "COMPLETED":
                    trajectory_successful_repetitions += 1

                    if (
                        trajectory_successful_repetitions
                        >= TRAJECTORY_TARGET_REPETITIONS
                    ):
                        trajectory_session_complete = True
                        print("\\nSESSION COMPLETE")
                    else:
                        REPETITION_NUMBER = (
                            trajectory_successful_repetitions + 1
                        )
                        trajectory_next_start_ready_at = (
                            time.monotonic() + 2.0
                        )
                        print(
                            f"Repetition {trajectory_successful_repetitions}"
                            f"/{TRAJECTORY_TARGET_REPETITIONS} complete"
                        )

                trajectory_results_path.parent.mkdir(
                    parents=True, exist_ok=True
                )

                session_results = {
                    "session_id": SESSION_ID,
                    "patient_id": SESSION_CONFIG.patient_id,
                    "object_id": SESSION_CONFIG.object_id,
                    "trajectory": getattr(
                        trajectory_assessment,
                        "preset_name",
                        SESSION_CONFIG.trajectory,
                    ),
                    "trajectory_config": {
                        "preset": getattr(
                            trajectory_assessment,
                            "preset_name",
                            SESSION_CONFIG.trajectory,
                        ),
                        "start_point": list(
                            trajectory_assessment.start_point
                        ),
                        "target_center": list(
                            trajectory_assessment.target_center
                        ),
                        "target_radius": trajectory_assessment.target_radius,
                        "waypoints": [
                            list(point)
                            for point in trajectory_assessment.waypoints
                        ],
                        "desired_path": [
                            list(point)
                            for point in trajectory_assessment.desired_path
                        ],
                        "coordinate_system": "image_pixels",
                    },
                    "target_repetitions": TRAJECTORY_TARGET_REPETITIONS,
                    "successful_repetitions":
                        trajectory_successful_repetitions,
                    "session_complete": trajectory_session_complete,
                    "attempts": trajectory_attempt_history,
                }

                trajectory_results_path.write_text(
                    json.dumps(session_results, indent=2, default=str)
                )

                print("Results saved:", trajectory_results_path)
                trajectory_attempt_start_ms = None

        # -------------------------------------------------
        # Draw trajectory testing mode
        # -------------------------------------------------

        if TRAJECTORY_MODE_ENABLED:
            trajectory_assessment.draw(frame)

            if trajectory_session_complete:
                repetition_text = "SESSION COMPLETE"
            else:
                repetition_text = (
                    f"REP {REPETITION_NUMBER}/"
                    f"{TRAJECTORY_TARGET_REPETITIONS}"
                )

            cv2.putText(
                frame,
                repetition_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if trajectory_session_complete
                else (255, 255, 255),
                2,
            )

            # Always display the centre of a currently
            # detected permitted object, even before
            # the game begins.
            for detected_object in permitted_detected_objects:
                center = detected_object.get("center")

                if center is None:
                    continue

                cx, cy = map(int, center)

                # Filled red dot with white border.
                cv2.circle(
                    frame, (cx, cy), 9,
                    (255, 255, 255), 2,
                )
                cv2.circle(
                    frame, (cx, cy), 6,
                    (0, 0, 255), -1,
                )

        writer.write(frame)

        cv2.imshow("Camera Capture", frame)

        # -------------------------------------------------
        # Trajectory preset keyboard controls
        # -------------------------------------------------
        trajectory_presets = {
            ord("1"): "horizontal",
            ord("2"): "vertical",
            ord("3"): "diagonal",
            ord("4"): "curved",
            ord("5"): "s_shaped",
        }

        key = cv2.waitKey(1) & 0xFF

        perf_totals["frame"] += (
            time.perf_counter() - perf_frame_start
        )
        perf_count += 1

        if perf_count % 30 == 0:
            print("\n[PERFORMANCE] Last 30 frames (average ms)")
            for name in ("pose", "yolo", "segmentation", "hands", "frame"):
                average_ms = perf_totals[name] * 1000 / perf_count
                print(f"  {name:<15}: {average_ms:8.2f}")

            average_frame_ms = (
                perf_totals["frame"] * 1000 / perf_count
            )
            if average_frame_ms > 0:
                print(
                    f"  Approx. FPS: {1000 / average_frame_ms:.2f}"
                )

            perf_totals = {
                name: 0.0 for name in perf_totals
            }
            perf_count = 0

        if key in trajectory_presets:
            if (
                not trajectory_assessment.attempt_active
                and not trajectory_session_complete
            ):
                selected_preset = trajectory_presets[key]

                trajectory_assessment.set_preset(
                    selected_preset,
                    frame_width=1280,
                    frame_height=720,
                )

                print(f"[TRAJECTORY] Selected: {selected_preset}")
            else:
                print(
                    "[TRAJECTORY] Cannot change preset "
                    "while an attempt is active."
                )

        # key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break



    # -----------------------------
    # Cleanup
    # -----------------------------

finally:
    camera.release()
    writer.release()
    csv_file.close()
    object_csv_file.close()
    interaction_csv_file.close()
    hands.close()
    cv2.destroyAllWindows()
    pose_csv_file.close()
    pose_detector.close()

print(f"Pose landmark data saved to: {pose_output_path}")


print("Recording stopped.")
print(f"Video saved to: {output_path}")
print(f"Landmark data saved to: {landmark_output_path}")
print(
    f"Interaction data saved to: "
    f"{interaction_output_path}"
)


play_video(output_path)
