import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "datasets/hograspnet_sample/GraspNet_S1_2_Labeling_data.zip"
OUTPUT = Path(__file__).resolve().parent / "hograspnet_s1_landmarks.npz"

X, y, objects, subjects, sources = [], [], [], [], []
invalid = Counter()

with zipfile.ZipFile(ARCHIVE) as zf:
    names = sorted(
        name for name in zf.namelist()
        if name.lower().endswith(".json")
    )

    for name in names:
        try:
            record = json.loads(zf.read(name))
        except (ValueError, UnicodeDecodeError):
            invalid["unreadable_json"] += 1
            continue

        annotations = record.get("annotations", [])

        for annotation in annotations:
            try:
                points = np.asarray(annotation["data"], dtype=np.float32)

                if points.shape != (1, 21, 3):
                    invalid["wrong_shape"] += 1
                    continue

                points = points[0]

                if not np.isfinite(points).all():
                    invalid["nonfinite_coordinates"] += 1
                    continue

                label = annotation.get("class_name")
                if not isinstance(label, str) or not label.strip():
                    invalid["missing_label"] += 1
                    continue

                X.append(points)
                y.append(label)
                objects.append(record.get("object", {}).get("name", "UNKNOWN"))
                subjects.append(record.get("actor", {}).get("id", "UNKNOWN"))
                sources.append(name)

            except (KeyError, TypeError, ValueError):
                invalid["malformed_annotation"] += 1

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=str)

np.savez_compressed(
    OUTPUT,
    landmarks=X,
    labels=y,
    objects=np.asarray(objects, dtype=str),
    subjects=np.asarray(subjects, dtype=str),
    sources=np.asarray(sources, dtype=str),
)

print("DATASET PREPARATION COMPLETE")
print("Valid samples:", len(y))
print("Landmark shape:", X.shape)
print("Invalid annotations:", dict(invalid))
print("Subjects:", sorted(set(subjects)))
print("\nClass counts:")

for label, count in Counter(y).most_common():
    print(f"  {label}: {count}")

print("\nSaved:", OUTPUT)
