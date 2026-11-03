import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
ZIP = ROOT / "datasets/hograspnet_sample/GraspNet_S1_2_Labeling_data.zip"

data = np.load(EXP / "hograspnet_s1_landmarks.npz")
X = data["landmarks"]
y = data["labels"]
sources = data["sources"]

print("=== 1. DUPLICATE LANDMARK ANALYSIS ===")

groups = defaultdict(list)
for i, sample in enumerate(X):
    groups[sample.tobytes()].append(i)

duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
conflicting = []

for indices in duplicate_groups:
    labels = set(y[indices])
    if len(labels) > 1:
        conflicting.append((indices, labels))

print("Total samples:", len(X))
print("Unique landmark configurations:", len(groups))
print("Duplicate groups:", len(duplicate_groups))
print("Redundant samples:", sum(len(g) - 1 for g in duplicate_groups))
print("Groups with conflicting labels:", len(conflicting))

print("\n=== 2. DUPLICATES BY CLASS ===")
duplicates_by_class = Counter()

for indices in duplicate_groups:
    for i in indices[1:]:
        duplicates_by_class[str(y[i])] += 1

for label, count in duplicates_by_class.most_common():
    print(f"{label:25} {count}")

print("\n=== 3. RECORDING GROUPS ===")
recordings = defaultdict(list)

for i, source in enumerate(sources):
    recording = Path(str(source)).parts[0]
    recordings[recording].append(i)

print("Recording groups:", len(recordings))

for recording, indices in sorted(
    recordings.items(), key=lambda item: -len(item[1])
)[:15]:
    labels = Counter(y[indices])
    print(recording, "samples=", len(indices), "labels=", dict(labels))

print("\n=== 4. FRAME METADATA EXAMPLES ===")

with zipfile.ZipFile(ZIP) as zf:
    for source in sources[:3]:
        record = json.loads(zf.read(str(source)))
        print({
            "source": str(source),
            "frame_num": record.get("images", {}).get("frame_num"),
            "image_files": record.get("images", {}).get("file_name"),
            "annotation_type": [
                a.get("type") for a in record.get("annotations", [])
            ],
        })

print("\n=== 5. CROSS-RECORDING DUPLICATES ===")
cross_recording = 0

for indices in duplicate_groups:
    recording_names = {
        Path(str(sources[i])).parts[0] for i in indices
    }
    if len(recording_names) > 1:
        cross_recording += 1

print("Duplicate groups spanning recordings:", cross_recording)

print("\nCONFLICT EXAMPLES:")
for indices, labels in conflicting[:5]:
    print("Labels:", sorted(labels))
    print("Sources:", [str(sources[i]) for i in indices[:5]])

print("\nNOTE:")
print("Landmark ordering cannot be verified from array shape alone.")
print("We must inspect the dataset specification or landmark definitions.")
