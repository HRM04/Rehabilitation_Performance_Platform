from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "hograspnet_s1_landmarks.npz")

X = data["landmarks"]
y = data["labels"]
objects = data["objects"]
subjects = data["subjects"]
sources = data["sources"]

# Keep the first occurrence of each exact landmark configuration.
# The previous audit found no conflicting labels.
seen = set()
keep = []

for i, sample in enumerate(X):
    key = sample.tobytes()
    if key not in seen:
        seen.add(key)
        keep.append(i)

keep = np.asarray(keep, dtype=int)

X = X[keep]
y = y[keep]
objects = objects[keep]
subjects = subjects[keep]
sources = sources[keep]

recordings = np.asarray([
    Path(str(source)).parts[0]
    for source in sources
])

class_groups = defaultdict(set)
for label, recording in zip(y, recordings):
    class_groups[str(label)].add(str(recording))

report = [
    "DEDUPLICATED DATASET REPORT",
    "=" * 55,
    f"Unique samples: {len(X)}",
    f"Recording groups: {len(set(recordings))}",
    f"Subjects: {sorted(set(subjects))}",
    "",
    "CLASS COVERAGE",
    "-" * 55,
]

for label, count in Counter(y).most_common():
    groups = len(class_groups[str(label)])
    status = (
        "CANDIDATE FOR GROUPED EVALUATION"
        if groups >= 2
        else "TRAINING / EXPLORATORY ONLY"
    )
    report.append(
        f"{label:25} samples={count:4} "
        f"recordings={groups:2} | {status}"
    )

output = HERE / "hograspnet_s1_clean.npz"

np.savez_compressed(
    output,
    landmarks=X,
    labels=y,
    objects=objects,
    subjects=subjects,
    sources=sources,
    recordings=recordings,
)

report_path = HERE / "clean_dataset_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved dataset:", output)
print("Saved report:", report_path)
print("\nNOTE: Grouped evaluation eligibility is provisional.")
print("Two recordings do not guarantee a reliable held-out test.")
print("Landmark ordering and coordinate units still need verification.")
