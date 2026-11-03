"""Grasp labels restricted to frames where a hand is in contact with an object.

Uses the interaction logs. Run from the project folder:
    python tools/verify_contact_grasps.py
"""
import csv, glob
from collections import Counter, defaultdict

CONTACT = {"CONTACT", "GRASPING"}
by_obj = defaultdict(Counter)
appr = defaultdict(Counter)
reasons = Counter()
combos = Counter()
state_counts = Counter()
n = 0
files = sorted(glob.glob("data/raw/interactions/*.csv"))
for p in files:
    with open(p, errors="ignore") as f:
        for r in csv.DictReader(f):
            st = r.get("interaction_state")
            if st:
                state_counts[st] += 1
            if st not in CONTACT or not r.get("detected_grasp_type"):
                continue
            n += 1
            cls = r.get("object_class") or "(none)"
            by_obj[cls][r["detected_grasp_type"]] += 1
            appr[cls][r.get("grasp_appropriateness") or ""] += 1
            if r.get("appropriateness_reason"):
                reasons[r["appropriateness_reason"]] += 1
            if r["detected_grasp_type"] == "AMBIGUOUS":
                combos[r.get("detected_grasp_types", "")] += 1

print("interaction files:", len(files))
print("interaction states (all hand frames):", dict(state_counts))
print("contact frames (CONTACT or GRASPING) with a grasp label:", n)
for cls, c in sorted(by_obj.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
    tot = sum(c.values())
    print("\n%s (%d frames)" % (cls, tot))
    print("  labels:        ", ", ".join("%s %.0f%%" % (k, 100 * v / tot) for k, v in c.most_common(6)))
    print("  appropriateness:", ", ".join("%s %.0f%%" % (k or "blank", 100 * v / tot) for k, v in appr[cls].most_common()))
print("\nMost common AMBIGUOUS candidate sets in contact frames:")
for k, v in combos.most_common(8):
    print("  %5d  %s" % (v, k))
if reasons:
    print("\nappropriateness_reason (new logs only):", dict(reasons.most_common(8)))
