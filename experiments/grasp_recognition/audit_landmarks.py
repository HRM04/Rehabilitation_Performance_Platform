from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
data = np.load(ROOT / "hograspnet_s1_landmarks.npz")

X = data["landmarks"]
y = data["labels"]
objects = data["objects"]
sources = data["sources"]

print("=== 1. RAW COORDINATE AUDIT ===")
for axis, name in enumerate(("x", "y", "z")):
    values = X[:, :, axis]
    print(
        f"{name}: min={values.min():.4f}, "
        f"max={values.max():.4f}, "
        f"median={np.median(values):.4f}"
    )

print("\nFirst sample, first 5 landmarks:")
print(X[0, :5])

print("\n=== 2. WRIST-RELATIVE NORMALIZATION ===")
relative = X - X[:, :1, :]
scale = np.linalg.norm(relative[:, 9, :], axis=1)

print("Wrist-to-landmark-9 distance:")
print("min:", scale.min())
print("median:", np.median(scale))
print("max:", scale.max())
print("Near-zero scales:", np.sum(scale < 1e-6))

safe_scale = np.maximum(scale, 1e-6)
normalized = relative / safe_scale[:, None, None]

print("Normalized shape:", normalized.shape)
print("All finite:", bool(np.isfinite(normalized).all()))

print("\n=== 3. DUPLICATE CHECK ===")
flat = X.reshape(len(X), -1)

exact_unique = len(np.unique(flat, axis=0))
rounded_unique = len(np.unique(np.round(flat, 2), axis=0))

print("Total samples:", len(X))
print("Exact duplicates:", len(X) - exact_unique)
print("Duplicates after rounding to 2 decimals:", len(X) - rounded_unique)

print("\n=== 4. CLASS / OBJECT COVERAGE ===")
coverage = defaultdict(set)

for label, obj in zip(y, objects):
    coverage[label].add(obj)

for label, count in Counter(y).most_common():
    print(
        f"{label:25} samples={count:4} "
        f"objects={len(coverage[label]):2}"
    )

print("\n=== 5. SOURCE / GROUP AUDIT ===")
groups = Counter()
for source in sources:
    parts = Path(str(source)).parts
    group = parts[0] if parts else "UNKNOWN"
    groups[group] += 1

print("Top source groups:", groups.most_common(12))
print("Unique source files:", len(set(sources)))

print("\nIMPORTANT:")
print("Only Subject S1 is available.")
print("Do not report subject-independent evaluation.")
print("Nearby frames may be correlated even without exact duplicates.")
print("Dataset coordinate units and landmark ordering remain unverified.")
