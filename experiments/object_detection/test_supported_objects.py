import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from src.object_detection.yolo_detector import detect_objects

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "object_compatibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUTPUT_DIR / f"{session_id}_detections.csv"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counts = Counter()
frame_number = 0
last_save_message = ""
save_message_until = 0

print("\nOBJECT COMPATIBILITY TEST")
print("Show different objects to the camera.")
print("S = save screenshot")
print("Q = quit")
print("Results CSV:", csv_path)

try:
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "timestamp",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ])

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            detections = detect_objects(frame)

            for obj in detections:
                name = obj["class_name"]
                confidence = obj["confidence"]
                x1, y1, x2, y2 = obj["bounding_box"]

                counts[name] += 1

                writer.writerow([
                    frame_number,
                    datetime.now().isoformat(),
                    name,
                    confidence,
                    x1, y1, x2, y2,
                ])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{name}: {confidence:.0%}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Objects detected: {len(detections)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if time.monotonic() < save_message_until:
                cv2.putText(
                    frame,
                    last_save_message,
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Object Compatibility Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                screenshot_path = (
                    OUTPUT_DIR
                    / f"{session_id}_frame_{frame_number}.jpg"
                )

                saved = cv2.imwrite(
                    str(screenshot_path),
                    frame,
                )

                if saved:
                    last_save_message = "Screenshot saved"
                    print("Saved:", screenshot_path)
                else:
                    last_save_message = "Screenshot FAILED"
                    print("Screenshot failed:", screenshot_path)

                save_message_until = time.monotonic() + 2

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\nDETECTION SUMMARY")
print("-" * 45)

for name, count in counts.most_common():
    print(f"{name:20} {count:5} detected frames")

print("\nCSV saved:", csv_path)
