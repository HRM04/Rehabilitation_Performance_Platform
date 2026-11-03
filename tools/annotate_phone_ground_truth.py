import copy
import json
from datetime import datetime
from pathlib import Path

import cv2

VIDEO = Path("data/raw/video/20261027_193035_raw_camera.mp4")
OUTPUT = Path("data/ground_truth/20261027_phone_annotations.json")

FRAMES = [
    1, 31, 61, 91, 131, 162,
    172, 182, 189, 192, 202,
    212, 222, 232, 242, 252, 262, 272,
    248, 270, 282, 290, 293,
    171, 173, 188, 190, 247, 249, 289,
]

FRAMES = sorted(set(FRAMES))

WINDOW = "Phone Ground Truth Annotation"

COLORS = {
    "visible_box": (0, 255, 0),
    "full_box": (255, 0, 255),
}


def blank_annotation(frame_number):
    return {
        "frame_number": frame_number,
        "phone_present": None,
        "visible_box": None,
        "full_box": None,
        "occluded": False,
        "truncated": False,
        "uncertain": False,
        "completed": False,
        "notes": "",
    }


def load_annotations():
    if not OUTPUT.exists():
        return {
            str(number): blank_annotation(number)
            for number in FRAMES
        }

    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    existing = data.get("annotations", {})

    return {
        str(number): {
            **blank_annotation(number),
            **existing.get(str(number), {}),
        }
        for number in FRAMES
    }


def save_annotations(annotations):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_video": str(VIDEO),
        "coordinate_system": "mirrored_image_pixels",
        "box_format": "xyxy",
        "frame_numbering": "1_based",
        "annotation_policy": {
            "visible_box": "Bounding box of visible phone extent",
            "full_box": "Estimated complete phone extent",
            "occlusion": "Object partially hidden by another object",
            "truncation": "Object extends outside image boundaries",
        },
        "updated_at": datetime.now().isoformat(),
        "annotations": annotations,
    }

    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)


def read_frame(capture, frame_number):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = capture.read()

    if not ok:
        raise RuntimeError(
            f"Unable to read frame {frame_number}"
        )

    return frame


def normalize_box(start, end, width, height):
    x1, y1 = start
    x2, y2 = end

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    return [x1, y1, x2, y2]


def main():
    if not VIDEO.exists():
        raise SystemExit(f"STOP: Video not found: {VIDEO}")

    capture = cv2.VideoCapture(str(VIDEO))

    if not capture.isOpened():
        raise SystemExit("STOP: Could not open video.")

    annotations = load_annotations()
    index = 0
    selected_box = "visible_box"
    dragging = False
    drag_start = None
    drag_end = None
    history = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 850)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal dragging, drag_start, drag_end

        frame_height, frame_width = userdata["shape"]
        panel_height = 100

        # Convert displayed panel coordinates to image coordinates.
        image_x = max(0, min(x, frame_width - 1))
        image_y = max(0, min(y - panel_height, frame_height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < panel_height:
                return

            dragging = True
            drag_start = (image_x, image_y)
            drag_end = (image_x, image_y)

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_end = (image_x, image_y)

        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            drag_end = (image_x, image_y)

            box = normalize_box(
                drag_start,
                drag_end,
                frame_width,
                frame_height,
            )

            if box is not None:
                number = FRAMES[index]
                record = annotations[str(number)]
                history.append(
                    (number, copy.deepcopy(record))
                )
                record[selected_box] = box
                record["phone_present"] = True
                record["completed"] = False

    try:
        while True:
            number = FRAMES[index]
            record = annotations[str(number)]
            frame = read_frame(capture, number)
            height, width = frame.shape[:2]

            cv2.setMouseCallback(
                WINDOW,
                on_mouse,
                {"shape": (height, width)},
            )

            display = frame.copy()

            for field, color in COLORS.items():
                box = record.get(field)

                if box is None:
                    continue

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    display,
                    "VISIBLE" if field == "visible_box"
                    else "FULL",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if dragging and drag_start and drag_end:
                cv2.rectangle(
                    display,
                    drag_start,
                    drag_end,
                    COLORS[selected_box],
                    2,
                )

            panel = cv2.copyMakeBorder(
                display,
                100, 0, 0, 0,
                cv2.BORDER_CONSTANT,
                value=(30, 30, 30),
            )

            status = (
                f"Frame {number} ({index + 1}/{len(FRAMES)}) "
                f"| Selected: {selected_box} "
                f"| Present: {record['phone_present']} "
                f"| Completed: {record['completed']}"
            )

            flags_text = (
                f"Occluded: {record['occluded']} "
                f"| Truncated: {record['truncated']} "
                f"| Uncertain: {record['uncertain']}"
            )

            cv2.putText(
                panel, status, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1,
            )

            cv2.putText(
                panel, flags_text, (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1,
            )

            cv2.putText(
                panel,
                "1 Visible | 2 Full | P Present | A Absent | "
                "O Occluded | T Truncated | X Uncertain | U Undo",
                (12, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1,
            )

            cv2.imshow(WINDOW, panel)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                continue

            if key == ord("1"):
                selected_box = "visible_box"

            elif key == ord("2"):
                selected_box = "full_box"

            elif key == ord("u"):
                for position in range(len(history) - 1, -1, -1):
                    previous_number, previous = history[position]
                    if previous_number == number:
                        annotations[str(number)] = previous
                        history.pop(position)
                        save_annotations(annotations)
                        break

            elif key in (
                ord("p"), ord("a"), ord("o"),
                ord("t"), ord("c"), ord("x"), ord("m")
            ):
                history.append((number, copy.deepcopy(record)))

                if key == ord("p"):
                    record["phone_present"] = True

                elif key == ord("a"):
                    record.update({
                        "phone_present": False,
                        "visible_box": None,
                        "full_box": None,
                        "occluded": False,
                        "truncated": False,
                        "uncertain": False,
                    })

                elif key == ord("o"):
                    record["occluded"] = not record["occluded"]

                elif key == ord("t"):
                    record["truncated"] = not record["truncated"]

                elif key == ord("x"):
                    record["uncertain"] = not record["uncertain"]

                elif key == ord("c"):
                    if record["visible_box"] is not None:
                        record["full_box"] = list(
                            record["visible_box"]
                        )

                elif key == ord("m"):
                    print(
                        f"\\nEnter note for frame {number} "
                        "(press Enter to finish):",
                        flush=True,
                    )
                    try:
                        record["notes"] = input("> ").strip()
                    except EOFError:
                        print("Note entry cancelled.")

                record["completed"] = False
                save_annotations(annotations)

            elif key == ord("n"):
                if record["phone_present"] is None:
                    print(
                        f"Frame {number}: mark present or absent."
                    )
                    continue

                if record["phone_present"] is True:
                    if (
                        record["visible_box"] is None
                        or record["full_box"] is None
                    ):
                        print(
                            f"Frame {number}: both boxes required."
                        )
                        continue

                record["completed"] = True
                save_annotations(annotations)
                index = min(index + 1, len(FRAMES) - 1)

            elif key == ord("b"):
                save_annotations(annotations)
                index = max(index - 1, 0)

            elif key == ord("q"):
                save_annotations(annotations)
                break

    finally:
        save_annotations(annotations)
        capture.release()
        cv2.destroyAllWindows()

    completed = sum(
        item["completed"]
        for item in annotations.values()
    )

    print(f"Annotations saved: {OUTPUT}")
    print(f"Completed: {completed}/{len(FRAMES)}")


if __name__ == "__main__":
    main()
