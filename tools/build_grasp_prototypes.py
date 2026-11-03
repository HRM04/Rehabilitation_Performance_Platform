"""Pick ONE best-guess grasp using the family model plus hand-shape ranking, scored on labelled recordings.
Run from the project folder:  python tools/evaluate_best_guess.py
Training uses every round except the last; all strategies are scored on the last round only."""
import csv, glob, importlib.util, os, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("vr", os.path.join("tools", "verify_grasp_replay.py"))
vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
from src.grasp import grasp_classifier as gc
from src.grasp import grasp_family as gf
from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance,
    calculate_finger_angles, calculate_thumb_opposition, calculate_thumb_index_side_distance, calculate_thumb_fingertip_distances)
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother
from src.preprocessing.grasp_voting import GraspVoter
NAMES = vr.NAMES
POWER = {"cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook"}
FAM = {n: ("POWER" if n in POWER else "PRECISION") for n in NAMES}
LINES = []; FEAT = {}
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LINES.append(s)
def scores(hd):
    tid = calculate_thumb_index_distance(hd); fa = calculate_finger_angles(hd)
    tmd = calculate_thumb_opposition(hd)["thumb_middle"]; fs = calculate_finger_spread(hd)
    ap = calculate_hand_aperture(hd); tis = calculate_thumb_index_side_distance(hd); tfd = calculate_thumb_fingertip_distances(hd)
    r = {"cylindrical": gc.detect_cylindrical_grasp(hd), "spherical": gc.detect_spherical_grasp(fa, tid, tmd, fs),
         "large_palmar": gc.detect_large_aperture_palmar_grasp(fa, fs, ap, tid, tmd), "whole_hand_power": gc.detect_whole_hand_power_grip(fa, tid, tmd, ap),
         "handle": gc.detect_handle_grasp(fa, tid, tmd, ap, fs), "hook": gc.detect_hook_grasp(fa, tid, tmd, ap),
         "pincer": gc.detect_pincer_grasp(fa, tid, tmd), "tip_pinch": gc.detect_tip_pinch(fa, tid, tmd),
         "tripod": gc.detect_tripod_grasp(fa, tid, tmd), "lateral_pinch": gc.detect_lateral_pinch(fa, tid, tis),
         "multidigit_precision": gc.detect_multidigit_precision_grasp(fa, tfd)}
    FEAT.clear()
    for k, v in (("thumb_index", tid), ("thumb_middle", tmd), ("aperture", ap), ("side_dist", tis), ("spread", fs)):
        if isinstance(v, (int, float)): FEAT[k] = float(v)
    for k, v in fa.items():
        if isinstance(v, (int, float)): FEAT["angle_" + str(k)] = float(v)
    return {n: (bool(v[0]), float(v[1] or 0.0)) for n, v in r.items()}
paths = sorted(glob.glob(os.path.join("data", "ground_truth", "grasp_labels_*.csv")))
if not paths: print("No labelled recordings in data/ground_truth/."); sys.exit(0)
seqs = defaultdict(lambda: defaultdict(dict))
for p in paths:
    with open(p) as f:
        for r in csv.DictReader(f):
            seqs[(r["session"], r["true_grasp"], int(r["rep"]), r["hand_label"])][(float(r["timestamp_ms"]), int(r["frame_number"]))][r["landmark_name"]] = (float(r["x_norm"]), float(r["y_norm"]), float(r["z_rel"]))
data = []
for (sess, true, rep, hand), frames in seqs.items():
    sm = HandLandmarkSmoother(enabled=True); out = []
    for (t, fn), pts in sorted(frames.items()):
        if len(pts) != 21: continue
        ts = t / 1000.0; last = sm._last_t.get(hand)
        if last is not None and ts - last > sm.max_gap_s: sm.reset(hand)
        sm._last_t[hand] = ts
        sp = {n: tuple(sm.smooth_values(hand, i, ax, v, ts) for ax, v in zip("xyz", p)) for i, (n, p) in enumerate(pts.items())}
        hd = vr.build(sp)
        if hd is None: continue
        sc_ = scores(hd); out.append((t, sc_, dict(FEAT)))
    if out: data.append((true, rep, hand, out))

rounds = sorted({d[1] for d in data})
vals = defaultdict(lambda: defaultdict(list))
for true, rep, hand, fr in data:
    for t, sc, f in fr:
        for k, v in f.items(): vals[true][k].append(v)
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(len(v) * p))]
proto = {g: {k: [q(v, .5), max((q(v, .75) - q(v, .25)) / 1.349, 0.05)] for k, v in vals[g].items() if len(v) >= 5} for g in vals}
import json
out = os.path.join("src", "grasp", "grasp_prototypes.json")
json.dump({"enabled": True, "sequences": len(data), "rounds": rounds, "note": "median and spread of each hand measurement per labelled grasp, one person", "proto": proto}, open(out, "w"), indent=1)
print("CREATED   " + os.path.abspath(out))
