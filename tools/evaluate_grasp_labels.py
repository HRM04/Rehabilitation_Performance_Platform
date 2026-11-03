"""Accuracy of the grasp classifier against recorded, labelled grasps.

Replays the recordings through the same steps as the live pipeline (smoothing,
11 detectors, arbitration, label averaging) and reports, per true grasp:
how often its own detector fires, how often the final label is correct,
and what it is confused with.

Run from the project folder:  python tools/evaluate_grasp_labels.py [file.csv]
"""
import csv, glob, importlib.util, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("vr", os.path.join("tools", "verify_grasp_replay.py"))
vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother
from src.preprocessing.grasp_voting import GraspVoter

paths = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "ground_truth", "grasp_labels_*.csv")))
if not paths:
    print("No labelled recordings found in data/ground_truth/. Run tools/record_labelled_grasps.py first.")
    sys.exit(0)

seqs = defaultdict(lambda: defaultdict(dict))
for p in paths:
    with open(p) as f:
        for r in csv.DictReader(f):
            key = (r["session"], r["true_grasp"], r["rep"], r["hand_label"])
            fk = (float(r["timestamp_ms"]), int(r["frame_number"]))
            seqs[key][fk][r["landmark_name"]] = (float(r["x_norm"]), float(r["y_norm"]), float(r["z_rel"]))

fire = defaultdict(Counter)        # true -> detector -> count
final_conf = defaultdict(Counter)  # true -> final label
raw_conf = defaultdict(Counter)
nframes = Counter()
for (sess, true, rep, hand), frames in seqs.items():
    sm = HandLandmarkSmoother(enabled=True)
    voter = GraspVoter(window=5)
    for (t, fn), pts in sorted(frames.items()):
        if len(pts) != 21:
            continue
        ts = t / 1000.0
        last = sm._last_t.get(hand)
        if last is not None and ts - last > sm.max_gap_s:
            sm.reset(hand)
        sm._last_t[hand] = ts
        sp = {n: tuple(sm.smooth_values(hand, i, ax, v, ts) for ax, v in zip("xyz", p)) for i, (n, p) in enumerate(pts.items())}
        hd = vr.build(sp)
        if hd is None:
            continue
        fired, _ = vr.classify(hd)
        nframes[true] += 1
        for n in fired:
            fire[true][n] += 1
        raw = vr.label(fired)
        final, _c = voter.vote(hand, t, raw, fired)
        raw_conf[true][raw] += 1
        final_conf[true][final] += 1

order = [n for n in vr.NAMES if nframes[n]]
print("labelled files:", len(paths), "| frames per grasp:", dict(nframes))
print("\nPER GRASP: own detector fires | final label correct | AMBIGUOUS | UNKNOWN | most common wrong label")
tot = ok = 0
for g in order:
    n = nframes[g]
    wrong = Counter({k: v for k, v in final_conf[g].items() if k not in (g, "AMBIGUOUS", "UNKNOWN")})
    top = wrong.most_common(1)
    print("  %-22s %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %s" % (
        g, 100 * fire[g][g] / n, 100 * final_conf[g][g] / n, 100 * final_conf[g]["AMBIGUOUS"] / n,
        100 * final_conf[g]["UNKNOWN"] / n, ("%s (%.0f%%)" % (top[0][0], 100 * top[0][1] / n)) if top else "-"))
    tot += n; ok += final_conf[g][g]
print("\nOverall: final label equals the true grasp on %.1f%% of %d frames" % (100 * ok / max(tot, 1), tot))
print("Before label averaging (raw): %.1f%%" % (100 * sum(raw_conf[g][g] for g in order) / max(tot, 1)))
print("\nDETECTOR FIRING MATRIX (rows = true grasp, columns = detector, % of frames)")
print("%-20s" % "" + "".join("%6s" % n[:5] for n in vr.NAMES))
for g in order:
    print("%-20s" % g + "".join("%6.0f" % (100 * fire[g][n] / nframes[g]) for n in vr.NAMES))
