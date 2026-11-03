import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "models" / "rgb_webcam_export_E8B"

MODEL_PATH = EXPORT / "mobilenetv3_grasp_medium.pt"
LABELS_PATH = EXPORT / "class_to_index.json"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
INPUT_SIZE = 224
DETECTION_WIDTH = 960

# Medium crop parameters from E6C
CROP_SCALE = 2.8
MIN_CROP_SIZE = 260

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("=" * 65)
print("E8B — LIVE RGB GRASP RECOGNITION")
print("=" * 65)

for path in [MODEL_PATH, LABELS_PATH, HAND_MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

# ------------------------------------------------------------
# 1. Load trained model
# ------------------------------------------------------------

model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
model.eval()

class_to_idx = json.loads(LABELS_PATH.read_text())
labels = [
    name for name, idx in sorted(
        class_to_idx.items(),
        key=lambda item: item[1]
    )
]

print("RGB model loaded.")
print("Classes:", len(labels))

# ------------------------------------------------------------
# 2. Initialize MediaPipe Tasks
# ------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35,
)

# ------------------------------------------------------------
# 3. Crop preparation
# ------------------------------------------------------------

def make_medium_crop(frame, landmarks):
    height, width = frame.shape[:2]

    xs = np.array([p.x * width for p in landmarks])
    ys = np.array([p.y * height for p in landmarks])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    hand_size = max(x_max - x_min, y_max - y_min)

    # Crop is defined in original-frame pixel coordinates.
    crop_size = max(
        hand_size * CROP_SCALE,
        MIN_CROP_SIZE
    )

    half = crop_size / 2

    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x2 = int(round(center_x + half))
    y2 = int(round(center_y + half))

    # Preserve the requested square crop, including at edges.
    side = max(1, x2 - x1, y2 - y1)

    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    crop = padded[
        y1 + pad_top:y2 + pad_top,
        x1 + pad_left:x2 + pad_left
    ]

    if crop.size == 0:
        return None, None

    crop = cv2.resize(
        crop,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA
    )

    display_box = (
        max(0, x1),
        max(0, y1),
        min(width - 1, x2),
        min(height - 1, y2)
    )

    return crop, display_box


def preprocess_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    image = crop_rgb.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    tensor = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    )

    return tensor.unsqueeze(0)


def predict_grasp(crop):
    tensor = preprocess_crop(crop)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=3)

    return [
        (labels[index], float(score))
        for score, index in zip(
            scores.tolist(),
            indices.tolist()
        )
    ]


# ------------------------------------------------------------
# 4. Open webcam
# ------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check macOS camera permissions."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nCamera opened.")
print("Press Q to quit.")
print("Press S to save a screenshot.")
print("Predictions are experimental, not clinical assessments.")

last_timestamp_ms = -1
start_time = time.monotonic()
frame_count = 0

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Camera frame unavailable.")
                break

            frame_count += 1

            height, width = frame.shape[:2]

            # MediaPipe runs on a resized image for speed.
            detection_scale = min(
                1.0,
                DETECTION_WIDTH / width
            )

            if detection_scale < 1:
                detection_frame = cv2.resize(
                    frame,
                    None,
                    fx=detection_scale,
                    fy=detection_scale
                )
            else:
                detection_frame = frame

            detection_rgb = cv2.cvtColor(
                detection_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(detection_rgb)
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            predictions_per_hand = []

            for hand_idx, landmarks in enumerate(
                result.hand_landmarks
            ):
                crop, box = make_medium_crop(
                    frame,
                    landmarks
                )

                if crop is None:
                    continue

                predictions = predict_grasp(crop)
                predictions_per_hand.append(predictions)

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                top_label, top_score = predictions[0]

                label_text = (
                    f"Hand {hand_idx + 1}: "
                    f"{top_label} {top_score:.1%}"
                )

                cv2.putText(
                    frame,
                    label_text,
                    (max(5, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            # Top-three predictions in a readable side panel.
            panel_height = max(
                110,
                35 + len(predictions_per_hand) * 105
            )

            panel_height = min(panel_height, height)

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (min(width, 520), panel_height),
                (25, 25, 25),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.75,
                frame,
                0.25,
                0,
                frame
            )

            cv2.putText(
                frame,
                "E8B - RGB GRASP RECOGNITION",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y = 58

            if not predictions_per_hand:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2
                )

            for hand_idx, predictions in enumerate(
                predictions_per_hand
            ):
                cv2.putText(
                    frame,
                    f"Hand {hand_idx + 1}",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                y += 25

                for label, score in predictions:
                    cv2.putText(
                        frame,
                        f"{label}: {score:.1%}",
                        (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 255, 200),
                        1
                    )
                    y += 23

                y += 10

            elapsed = max(
                time.monotonic() - start_time,
                1e-6
            )

            fps = frame_count / elapsed

            cv2.putText(
                frame,
                f"Average FPS: {fps:.1f}",
                (15, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "E8B - Live RGB Grasp Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                output_dir = (
                    ROOT / "outputs" / "rgb_webcam_E8B"
                )
                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                filename = (
                    output_dir
                    / f"screenshot_{int(time.time())}.jpg"
                )

                cv2.imwrite(str(filename), frame)
                print("Screenshot saved:", filename)

finally:
    cap.release()
    cv2.destroyAllWindows()

print("E8B webcam test finished.")
