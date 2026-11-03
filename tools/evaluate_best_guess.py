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
last_rep = max(d[1] for d in data)
train = [d for d in data if d[1] != last_rep] or data
test = [d for d in data if d[1] == last_rep]
P("recordings:", len(paths), "| sequences:", len(data), "| scored on round", last_rep, "(%d sequences)" % len(test))
fired_n, fired_ok = Counter(), Counter()
for true, rep, hand, fr in train:
    for t, sc, _f in fr:
        for n, (f, c) in sc.items():
            if f: fired_n[n] += 1; fired_ok[n] += (n == true)
W = {n: (fired_ok[n] + 0.5) / (fired_n[n] + 5.0) for n in NAMES}
vals = defaultdict(lambda: defaultdict(list))
for true, rep, hand, fr in train:
    for t, sc, f in fr:
        for k, v in f.items(): vals[true][k].append(v)
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(len(v) * p))]
PROTO = {g: {k: (q(v, .5), max((q(v, .75) - q(v, .25)) / 1.349, 0.05)) for k, v in vals[g].items() if len(v) >= 5} for g in vals}
def dist(g, f):
    p = PROTO.get(g)
    if not p: return 99.0
    ds = [min(((f[k] - m) / s) ** 2, 25.0) for k, (m, s) in p.items() if k in f]
    return sum(ds) / len(ds) if ds else 99.0
def fam_pred(f):
    if not gf.MODEL: return "POWER"
    return gf._walk(gf.MODEL["tree"], f, gf.MODEL["features"], gf.MODEL["medians"])
def fired(sc): return [n for n in NAMES if sc[n][0]]
def s0(sc, f):
    x = fired(sc); return (x[0] if len(x) == 1 else ("AMBIGUOUS" if x else "UNKNOWN")), x
def s3(sc, f):
    x = fired(sc)
    return (max(x, key=lambda n: sc[n][1] * W[n]) if x else "UNKNOWN"), x
def b1(sc, f):
    fam = fam_pred(f); c = [n for n in fired(sc) if FAM[n] == fam]
    if not c: return "AMBIGUOUS", [n for n in NAMES if FAM[n] == fam]
    return max(c, key=lambda n: sc[n][1] * W[n]), c
def b2(sc, f):
    c = [n for n in NAMES if FAM[n] == fam_pred(f)]
    return min(c, key=lambda n: dist(n, f)), c
def b3(sc, f):
    fam = fam_pred(f); c = [n for n in fired(sc) if FAM[n] == fam] or [n for n in NAMES if FAM[n] == fam]
    return min(c, key=lambda n: dist(n, f)), c
STR = [("S0 current (reference)", s0), ("S3 reliability (earlier best)", s3), ("B1 family, then detector rank", b1),
       ("B2 family, then hand-shape rank", b2), ("B3 family + fired, hand-shape rank", b3)]
def fam_ok(true, lab, cands):
    tf = true in POWER
    if lab in NAMES: return (lab in POWER) == tf
    if lab == "AMBIGUOUS" and cands: return all((c in POWER) == tf for c in cands)
    return False
def run(fn):
    res = []
    for true, rep, hand, fr in test:
        v = GraspVoter(window=5)
        for t, sc, f in fr:
            lab, cands = fn(sc, f); lab, cands = v.vote(hand, t, lab, cands); res.append((true, lab, cands))
    return res
P("\n%-38s %8s %8s %8s %8s" % ("strategy (held-out round)", "exact", "family", "AMBIG", "UNKNOWN"))
best = None; allres = {}
for name, fn in STR:
    res = run(fn); allres[name] = res; n = max(len(res), 1)
    ex = sum(t == l for t, l, c in res) / n; fm = sum(fam_ok(t, l, c) for t, l, c in res) / n
    P("%-38s %7.1f%% %7.1f%% %7.1f%% %7.1f%%" % (name, 100 * ex, 100 * fm, 100 * sum(l == "AMBIGUOUS" for t, l, c in res) / n, 100 * sum(l == "UNKNOWN" for t, l, c in res) / n))
    if best is None or ex > best[0]: best = (ex, name)
P("\nBest by exact match: %s -- per grasp" % best[1])
by = defaultdict(list)
for t, l, c in allres[best[1]]: by[t].append(l)
for g in NAMES:
    if g in by:
        labs = Counter(by[g]); n = len(by[g]); wr = [(k, v) for k, v in labs.most_common() if k != g]
        P("  %-22s correct %5.1f%% | most common wrong label: %s (%.0f%%)" % (g, 100 * labs[g] / n, wr[0][0] if wr else "-", 100 * wr[0][1] / n if wr else 0))
os.makedirs("reports", exist_ok=True)
out = os.path.abspath("reports/best_guess_%s.txt" % time.strftime("%Y%m%d_%H%M%S"))
open(out, "w").write("\n".join(LINES)); print("\nCREATED   " + out)
print("CREATED   " + os.path.abspath("tools/evaluate_best_guess.py"))
