import json
import zipfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "datasets/hograspnet_sample/GraspNet_S1_2_Labeling_data.zip"
OUT = Path(__file__).resolve().parent / "dataset_report.txt"

IMPLEMENTED = [
    "cylindrical", "spherical", "large_palmar",
    "whole_hand_power", "handle", "hook",
    "pincer", "tip_pinch", "tripod",
    "lateral_pinch", "multidigit_precision",
]

def shape(value):
    if isinstance(value, list):
        return [len(value), shape(value[0]) if value else None]
    if isinstance(value, dict):
        return {k: shape(v) for k, v in list(value.items())[:8]}
    return type(value).__name__

def main():
    labels = Counter()
    class_ids = {}
    subjects = Counter()
    objects = Counter()
    camera_files = set()
    annotation_shapes = Counter()
    errors = 0

    with zipfile.ZipFile(ZIP_PATH) as archive:
        files = [
            name for name in archive.namelist()
            if name.lower().endswith(".json")
        ]

        for name in files:
            try:
                record = json.loads(archive.read(name))
            except (ValueError, UnicodeDecodeError):
                errors += 1
                continue

            actor = record.get("actor", {})
            subjects[str(actor.get("id", "UNKNOWN"))] += 1

            obj = record.get("object", {})
            objects[str(obj.get("name", "UNKNOWN"))] += 1

            images = record.get("images", {})
            for path in images.get("file_name", []):
                camera_files.add(path.split("/")[1] if "/" in path else path)

            for ann in record.get("annotations", []):
                label = str(ann.get("class_name", "UNKNOWN"))
                labels[label] += 1
                class_ids.setdefault(label, set()).add(ann.get("class_id"))

                data = ann.get("data")
                annotation_shapes[str(shape(data))] += 1

    lines = [
        "HOGRASPNET DATASET INSPECTION",
        "=" * 50,
        f"JSON files: {len(files)}",
        f"Unreadable files: {errors}",
        f"Subjects: {dict(subjects)}",
        f"Objects: {dict(objects)}",
        f"Camera/path groups: {sorted(camera_files)}",
        "",
        "GRASP LABELS",
        "-" * 50,
    ]

    for label, count in labels.most_common():
        lines.append(
            f"{label:35} count={count:5} ids={sorted(class_ids[label], key=str)}"
        )

    lines += [
        "",
        "ANNOTATION DATA SHAPES",
        "-" * 50,
    ]

    for s, count in annotation_shapes.most_common(15):
        lines.append(f"{s}: {count}")

    lines += [
        "",
        "OUR 11 IMPLEMENTED GRASPS",
        "-" * 50,
        *IMPLEMENTED,
        "",
        "NOTE: Dataset-to-system mapping is not yet validated.",
        "Label similarity alone does not establish equivalence.",
    ]

    report = "\n".join(lines)
    OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {OUT}")

if __name__ == "__main__":
    main()
