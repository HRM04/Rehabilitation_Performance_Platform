"""Offline verification using logged landmarks.

1. Re-runs all 11 grasp detectors on every logged hand frame (raw landmarks)
   and counts how often each fires, which pairs fire together, and why the
   whole-hand power grip rarely or never fires.
2. Re-runs them on landmarks passed through the new smoother and compares
   jitter, lag and label counts.
3. Counts object classes seen in the object logs.

Run from the project folder:  python tools/verify_grasp_replay.py [--limit N]
"""
import argparse, csv, glob, os, sys, statistics
from collections import Counter, defaultdict

root = os.getcwd()
sys.path.insert(0, root)
from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance,
    calculate_finger_angles, calculate_thumb_opposition, calculate_thumb_index_side_distance, calculate_thumb_fingertip_distances)
from src.grasp import grasp_classifier as gc
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother

W, H = 1280, 720
NAMES = ["cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook",
         "pincer", "tip_pinch", "tripod", "lateral_pinch", "multidigit_precision"]


def build(points):
    hd = {}
    for name, (xn, yn, zr) in points.items():
        hd[name] = {"x_norm": xn, "y_norm": yn, "z_rel": zr, "x_px": int(xn * W), "y_px": int(yn * H)}
    wx, wy = hd["WRIST"]["x_px"], hd["WRIST"]["y_px"]
    for c in hd.values():
        c["x_rel"] = c["x_px"] - wx
        c["y_rel"] = c["y_px"] - wy
    m = hd["MIDDLE_FINGER_MCP"]
    ref = (m["x_rel"] ** 2 + m["y_rel"] ** 2) ** 0.5
    if ref <= 0:
        return None
    for c in hd.values():
        c["x_scaled"] = c["x_rel"] / ref
        c["y_scaled"] = c["y_rel"] / ref
    return hd


def classify(hd):
    tid = calculate_thumb_index_distance(hd)
    fa = calculate_finger_angles(hd)
    to = calculate_thumb_opposition(hd)
    tmd = to["thumb_middle"]
    fs = calculate_finger_spread(hd)
    ap = calculate_hand_aperture(hd)
    tis = calculate_thumb_index_side_distance(hd)
    tfd = calculate_thumb_fingertip_distances(hd)
    r = {
        "cylindrical": gc.detect_cylindrical_grasp(hd)[0],
        "spherical": gc.detect_spherical_grasp(fa, tid, tmd, fs)[0],
        "large_palmar": gc.detect_large_aperture_palmar_grasp(fa, fs, ap, tid, tmd)[0],
        "whole_hand_power": gc.detect_whole_hand_power_grip(fa, tid, tmd, ap)[0],
        "handle": gc.detect_handle_grasp(fa, tid, tmd, ap, fs)[0],
        "hook": gc.detect_hook_grasp(fa, tid, tmd, ap)[0],
        "pincer": gc.detect_pincer_grasp(fa, tid, tmd)[0],
        "tip_pinch": gc.detect_tip_pinch(fa, tid, tmd)[0],
        "tripod": gc.detect_tripod_grasp(fa, tid, tmd)[0],
        "lateral_pinch": gc.detect_lateral_pinch(fa, tid, tis)[0],
        "multidigit_precision": gc.detect_multidigit_precision_grasp(fa, tfd)[0],
    }
    valid = [a for a in fa.values() if a is not None]
    diag = {}
    if len(valid) == 4:
        diag = {"flexed>=3": sum(a < 145 for a in valid) >= 3, "strong>=2": sum(a < 110 for a in valid) >= 2,
                "thumb_engaged": (tid < 1.6 or tmd < 1.6), "closed_hand": ap < 1.5, "min_angle": min(valid),
                "aperture": ap}
    return [n for n in NAMES if r[n]], diag


def label(fired):
    return fired[0] if len(fired) == 1 else ("AMBIGUOUS" if fired else "UNKNOWN")


def load(path, limit):
    seqs = defaultdict(list)
    with open(path, errors="ignore") as f:
        for row in csv.DictReader(f):
            if not row.get("hand_label") or not row.get("landmark_name"):
                continue
            key = (int(float(row["frame_number"])), row["hand_label"], float(row["timestamp_ms"]))
            seqs[key].append((row["landmark_name"], float(row["x_norm"]), float(row["y_norm"]), float(row["z_rel"])))
    frames = sorted(seqs.items(), key=lambda kv: kv[0][2])
    return [(k, {n: (x, y, z) for n, x, y, z in v}) for k, v in frames if len(v) == 21]


def jitter(series):
    d2 = [series[i + 1] - 2 * series[i] + series[i - 1] for i in range(1, len(series) - 1)]
    return (sum(v * v for v in d2) / len(d2)) ** 0.5 if d2 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob("data/raw/landmarks/*.csv"))
    print("landmark files:", len(files))
    raw_fire, raw_lab, sm_lab = Counter(), Counter(), Counter()
    pairs, dg = Counter(), Counter()
    n_frames = 0
    jit_raw, jit_sm, lag = [], [], []
    slow_raw, slow_sm = [], []
    for fi, path in enumerate(files):
        frames = load(path, args.limit)
        if args.limit:
            frames = frames[:args.limit]
        smoother = HandLandmarkSmoother(enabled=True)
        by_hand = defaultdict(lambda: ([], []))
        for (fn, hand, ts), pts in frames:
            hd = build(pts)
            if hd is None:
                continue
            fired, diag = classify(hd)
            n_frames += 1
            raw_lab[label(fired)] += 1
            for n in fired:
                raw_fire[n] += 1
            for i in range(len(fired)):
                for j in range(i + 1, len(fired)):
                    pairs[(fired[i], fired[j])] += 1
            for k, v in diag.items():
                if isinstance(v, bool):
                    dg[k] += int(v)
            if diag:
                dg["_n"] += 1
            # smoothed
            t = ts / 1000.0
            last = smoother._last_t.get(hand)
            if last is not None and t - last > smoother.max_gap_s:
                smoother.reset(hand)
            smoother._last_t[hand] = t
            sp = {n: tuple(smoother.smooth_values(hand, idx, ax, val, t) for ax, val in zip("xyz", p))
                  for idx, (n, p) in enumerate(pts.items())}
            hs = build(sp)
            if hs is not None:
                fs, _ = classify(hs)
                sm_lab[label(fs)] += 1
            xr = pts["INDEX_FINGER_TIP"][0] * W
            xs = sp["INDEX_FINGER_TIP"][0] * W
            by_hand[hand][0].append(xr)
            by_hand[hand][1].append(xs)
            lag.append(abs(xr - xs))
        for hand, (a, b) in by_hand.items():
            for i in range(1, len(a)):
                dr = abs(a[i] - a[i - 1])
                if dr < 15:
                    slow_raw.append(dr)
                    slow_sm.append(abs(b[i] - b[i - 1]))
            if len(a) > 5:
                jit_raw.append(jitter(a))
                jit_sm.append(jitter(b))

    print("\nhand frames analysed:", n_frames)
    print("\nDetector firing counts (raw landmarks, re-run offline):")
    for n in NAMES:
        print("  %-22s %6d  (%.1f%%)" % (n, raw_fire[n], 100.0 * raw_fire[n] / max(n_frames, 1)))
    print("\nLabel after arbitration (raw):", dict(raw_lab))
    print("Label after arbitration (smoothed):", dict(sm_lab))
    print("\nMost common co-firing pairs (cause of AMBIGUOUS):")
    for (a, b), c in pairs.most_common(10):
        print("  %s + %s: %d" % (a, b, c))
    if dg.get("_n"):
        print("\nWhole-hand power grip, how often each condition holds (of %d frames with four valid finger angles):" % dg["_n"])
        for k in ("flexed>=3", "strong>=2", "thumb_engaged", "closed_hand"):
            print("  %-14s %.1f%%" % (k, 100.0 * dg[k] / dg["_n"]))
    if jit_raw:
        print("\nIndex fingertip x jitter (RMS second difference, px): raw %.2f, smoothed %.2f (median over sequences)" % (
            statistics.median(jit_raw), statistics.median(jit_sm)))
        print("Mean absolute difference smoothed vs raw (px): %.2f" % (sum(lag) / len(lag)))
    if slow_raw:
        print("Frame-to-frame change on slow frames (<15 px raw), mean px: raw %.2f, smoothed %.2f (%d frame pairs)" % (
            sum(slow_raw) / len(slow_raw), sum(slow_sm) / len(slow_sm), len(slow_raw)))

    cls = Counter()
    for p in glob.glob("data/raw/objects/*.csv"):
        with open(p, errors="ignore") as f:
            for r in csv.DictReader(f):
                if r.get("object_class"):
                    cls[r["object_class"]] += 1
    print("\nObject classes seen in object logs (%d distinct):" % len(cls))
    for k, v in cls.most_common():
        print("  %-14s %d" % (k, v))


if __name__ == "__main__":
    main()
