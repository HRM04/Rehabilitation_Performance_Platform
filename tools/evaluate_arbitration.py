"""Compare ways of choosing ONE grasp label from the 11 detectors, using labelled recordings.

Run from the project folder:  python tools/evaluate_arbitration.py [file.csv ...]

Strategies (all use the same 5-frame label averaging afterwards):
  S0 current        exactly one detector fires -> that label; several -> AMBIGUOUS
  S1 top confidence the fired detector with the highest confidence wins
  S2 top, margin    as S1 but AMBIGUOUS when the best two are within 0.10
  S3 weighted       confidence x detector reliability, reliability measured on the training rounds
Training uses all rounds except the last one; every strategy is scored on the last round only,
so the weighted strategy is not marked on data it learned from. One person, so read it as a baseline.
"""
import csv, glob, importlib.util, os, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("vr", os.path.join("tools", "verify_grasp_replay.py"))
vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
from src.grasp import grasp_classifier as gc
from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance,
    calculate_finger_angles, calculate_thumb_opposition, calculate_thumb_index_side_distance, calculate_thumb_fingertip_distances)
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother
from src.preprocessing.grasp_voting import GraspVoter

NAMES = vr.NAMES
POWER = {"cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook"}
LINES = []
FEAT = {}
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LINES.append(s)

def scores(hd):
    tid = calculate_thumb_index_distance(hd); fa = calculate_finger_angles(hd)
    tmd = calculate_thumb_opposition(hd)["thumb_middle"]; fs = calculate_finger_spread(hd)
    ap = calculate_hand_aperture(hd); tis = calculate_thumb_index_side_distance(hd); tfd = calculate_thumb_fingertip_distances(hd)
    r = {"cylindrical": gc.detect_cylindrical_grasp(hd),
         "spherical": gc.detect_spherical_grasp(fa, tid, tmd, fs),
         "large_palmar": gc.detect_large_aperture_palmar_grasp(fa, fs, ap, tid, tmd),
         "whole_hand_power": gc.detect_whole_hand_power_grip(fa, tid, tmd, ap),
         "handle": gc.detect_handle_grasp(fa, tid, tmd, ap, fs),
         "hook": gc.detect_hook_grasp(fa, tid, tmd, ap),
         "pincer": gc.detect_pincer_grasp(fa, tid, tmd),
         "tip_pinch": gc.detect_tip_pinch(fa, tid, tmd),
         "tripod": gc.detect_tripod_grasp(fa, tid, tmd),
         "lateral_pinch": gc.detect_lateral_pinch(fa, tid, tis),
         "multidigit_precision": gc.detect_multidigit_precision_grasp(fa, tfd)}
    FEAT.clear()
    for k, v in (("thumb_index", tid), ("thumb_middle", tmd), ("aperture", ap), ("side_dist", tis), ("spread", fs)):
        if isinstance(v, (int, float)): FEAT[k] = float(v)
    for k, v in fa.items():
        if isinstance(v, (int, float)): FEAT["angle_" + str(k)] = float(v)
    return {n: (bool(v[0]), float(v[1] or 0.0)) for n, v in r.items()}

paths = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "ground_truth", "grasp_labels_*.csv")))
if not paths:
    print("No labelled recordings in data/ground_truth/. Run tools/record_labelled_grasps.py first."); sys.exit(0)
seqs = defaultdict(lambda: defaultdict(dict))
for p in paths:
    with open(p) as f:
        for r in csv.DictReader(f):
            seqs[(r["session"], r["true_grasp"], int(r["rep"]), r["hand_label"])][(float(r["timestamp_ms"]), int(r["frame_number"]))][r["landmark_name"]] = (float(r["x_norm"]), float(r["y_norm"]), float(r["z_rel"]))

data = []   # (true, rep, hand, [(t, scores)])
for (sess, true, rep, hand), frames in seqs.items():
    sm = HandLandmarkSmoother(enabled=True); out = []
    for (t, fn), pts in sorted(frames.items()):
        if len(pts) != 21: continue
        ts = t / 1000.0
        last = sm._last_t.get(hand)
        if last is not None and ts - last > sm.max_gap_s: sm.reset(hand)
        sm._last_t[hand] = ts
        sp = {n: tuple(sm.smooth_values(hand, i, ax, v, ts) for ax, v in zip("xyz", p)) for i, (n, p) in enumerate(pts.items())}
        hd = vr.build(sp)
        if hd is None: continue
        sc_ = scores(hd); out.append((t, sc_, dict(FEAT)))
    if out: data.append((true, rep, hand, out))
last_rep = max(d[1] for d in data)
train = [d for d in data if d[1] != last_rep] or data
P("recordings:", len(paths), "| sequences:", len(data), "| rounds:", sorted({d[1] for d in data}), "| scored on round", last_rep)

# reliability weights from training rounds: P(true == d | d fired)
fired_n, fired_ok = Counter(), Counter()
for true, rep, hand, fr in train:
    for t, sc, _f in fr:
        for n, (f, c) in sc.items():
            if f: fired_n[n] += 1; fired_ok[n] += (n == true)
W = {n: (fired_ok[n] + 0.5) / (fired_n[n] + 5.0) for n in NAMES}
P("detector reliability on training rounds (share of its firings that were the true grasp):")
P("   " + "  ".join("%s %.2f" % (n[:8], W[n]) for n in NAMES))

def s0(sc):
    f = [n for n in NAMES if sc[n][0]]
    return (f[0] if len(f) == 1 else ("AMBIGUOUS" if f else "UNKNOWN")), f
def s1(sc):
    f = [n for n in NAMES if sc[n][0]]
    if not f: return "UNKNOWN", f
    return max(f, key=lambda n: sc[n][1]), f
def s2(sc):
    f = [n for n in NAMES if sc[n][0]]
    if not f: return "UNKNOWN", f
    r = sorted(f, key=lambda n: -sc[n][1])
    if len(r) > 1 and sc[r[0]][1] - sc[r[1]][1] < 0.10: return "AMBIGUOUS", f
    return r[0], f
def s3(sc):
    f = [n for n in NAMES if sc[n][0]]
    if not f: return "UNKNOWN", f
    return max(f, key=lambda n: sc[n][1] * W[n]), f
STRATS = [("S0 current", s0), ("S1 top confidence", s1), ("S2 top, margin 0.10", s2), ("S3 weighted by reliability", s3)]

def run(fn, seqs_):
    res = []
    for true, rep, hand, fr in seqs_:
        v = GraspVoter(window=5)
        for t, sc, _f in fr:
            lab, cands = fn(sc)
            lab, cands = v.vote(hand, t, lab, cands)
            res.append((true, lab, cands))
    return res

def fam_ok(true, lab, cands):
    tf = true in POWER
    if lab in NAMES: return (lab in POWER) == tf
    if lab == "AMBIGUOUS" and cands: return all((c in POWER) == tf for c in cands)
    return False

test = [d for d in data if d[1] == last_rep]
P("\nSCORED ON ROUND %d ONLY (%d sequences)" % (last_rep, len(test)))
P("%-30s %10s %12s %10s %10s" % ("strategy", "exact", "family", "AMBIG", "UNKNOWN"))
best = None
allres = {}
for name, fn in STRATS:
    res = run(fn, test); allres[name] = res; n = max(len(res), 1)
    ex = sum(t == l for t, l, c in res) / n; fam = sum(fam_ok(t, l, c) for t, l, c in res) / n
    P("%-30s %9.1f%% %11.1f%% %9.1f%% %9.1f%%" % (name, 100 * ex, 100 * fam, 100 * sum(l == "AMBIGUOUS" for t, l, c in res) / n, 100 * sum(l == "UNKNOWN" for t, l, c in res) / n))
    if best is None or ex > best[0]: best = (ex, name)
P("(exact = label equals the true grasp. family = power-type vs precision-type is right, which is what")
P(" 'appropriate grasp family' scoring would use. S3 learned its weights on the other rounds.)")

P("\nBEST STRATEGY BY EXACT MATCH: %s -- per grasp, round %d" % (best[1], last_rep))
res = allres[best[1]]; by = defaultdict(list)
for t, l, c in res: by[t].append(l)
for g in NAMES:
    if g not in by: continue
    labs = Counter(by[g]); n = len(by[g])
    wrong = [(k, v) for k, v in labs.most_common() if k != g]
    P("  %-22s correct %5.1f%% | most common label: %s (%.0f%%)" % (g, 100 * labs[g] / n, wrong[0][0] if wrong else "-", 100 * wrong[0][1] / n if wrong else 0))

P("\nCONFUSION OF BEST STRATEGY (rows = true, columns = predicted, % of frames)")
cols = NAMES + ["AMBIGUOUS", "UNKNOWN"]
P("%-20s" % "" + "".join("%6s" % c[:5] for c in cols))
for g in NAMES:
    if g not in by: continue
    labs = Counter(by[g]); n = len(by[g])
    P("%-20s" % g + "".join("%6.0f" % (100 * labs[c] / n) for c in cols))

P("\nKEY HAND MEASUREMENTS PER TRUE GRASP (median, with 25th to 75th percentile; all rounds)")
featn = sorted({k for d in data for t, sc, f in d[3] for k in f})
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(len(v) * q))]
P("%-22s" % "" + "".join("%-28s" % k[:18] for k in featn))
for g in NAMES:
    vals = defaultdict(list)
    for true, rep, hand, fr in data:
        if true == g:
            for t, sc, f in fr:
                for k, v in f.items(): vals[k].append(v)
    if vals:
        P("%-22s" % g + "".join(("%-28s" % ("%.2f (%.2f-%.2f)" % (pct(vals[k], .5), pct(vals[k], .25), pct(vals[k], .75)))) if vals.get(k) else "%-28s" % "-" for k in featn))

os.makedirs("reports", exist_ok=True)
out = os.path.abspath("reports/arbitration_%s.txt" % time.strftime("%Y%m%d_%H%M%S"))
open(out, "w").write("\n".join(LINES)); print("\nCREATED   " + out)
