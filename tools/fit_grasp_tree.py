"""Fit a small, readable rule tree on the labelled grasp recordings and test it on a held-out round.

Run from the project folder:  python tools/fit_grasp_tree.py [file.csv ...]

Rounds 1..N-1 train the tree, round N is only used for scoring. Depths 2 to 6 are compared so the
simplest tree that holds up on the held-out round can be chosen. The final tree for the chosen
depth is refit on all rounds and saved as JSON together with a printout of its rules.
One person recorded the data, so this is a baseline for that person, not a general result.
"""
import csv, glob, importlib.util, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("vr", os.path.join("tools", "verify_grasp_replay.py"))
vr = importlib.util.module_from_spec(spec); spec.loader.exec_module(vr)
from src.grasp.grasp_features import (calculate_finger_spread, calculate_hand_aperture, calculate_thumb_index_distance,
    calculate_finger_angles, calculate_thumb_opposition, calculate_thumb_index_side_distance, calculate_thumb_fingertip_distances)
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother
from src.preprocessing.grasp_voting import GraspVoter

NAMES = vr.NAMES
POWER = {"cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook"}
LINES = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); LINES.append(s)

def features(hd):
    f = {"thumb_index": calculate_thumb_index_distance(hd), "thumb_middle": calculate_thumb_opposition(hd)["thumb_middle"],
         "aperture": calculate_hand_aperture(hd), "side_dist": calculate_thumb_index_side_distance(hd),
         "spread": calculate_finger_spread(hd)}
    for k, v in calculate_finger_angles(hd).items(): f["angle_" + str(k)] = v
    tfd = calculate_thumb_fingertip_distances(hd)
    if isinstance(tfd, dict):
        for k, v in tfd.items(): f["thumb_to_" + str(k)] = v
    return {k: float(v) for k, v in f.items() if isinstance(v, (int, float))}

paths = sys.argv[1:] or sorted(glob.glob(os.path.join("data", "ground_truth", "grasp_labels_*.csv")))
if not paths:
    print("No labelled recordings in data/ground_truth/."); sys.exit(0)
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
        ts = t / 1000.0
        last = sm._last_t.get(hand)
        if last is not None and ts - last > sm.max_gap_s: sm.reset(hand)
        sm._last_t[hand] = ts
        sp = {n: tuple(sm.smooth_values(hand, i, ax, v, ts) for ax, v in zip("xyz", p)) for i, (n, p) in enumerate(pts.items())}
        hd = vr.build(sp)
        if hd is None: continue
        out.append((t, features(hd)))
    if out: data.append((true, rep, hand, out))
FEATS = sorted({k for d in data for t, f in d[3] for k in f})
last_rep = max(d[1] for d in data)
def matrix(seqlist):
    X, y = [], []
    for true, rep, hand, fr in seqlist:
        for t, f in fr:
            X.append([f.get(k, np.nan) for k in FEATS]); y.append(true)
    return np.array(X, float), np.array(y)
train_seqs = [d for d in data if d[1] != last_rep]; test_seqs = [d for d in data if d[1] == last_rep]
Xtr, ytr = matrix(train_seqs); Xte, yte = matrix(test_seqs)
med = np.nanmedian(Xtr, axis=0)
def fill(X): X = X.copy(); i = np.where(np.isnan(X)); X[i] = np.take(med, i[1]); return X
Xtr, Xte = fill(Xtr), fill(Xte)
P("recordings:", len(paths), "| train frames:", len(ytr), "| held-out frames (round %d):" % last_rep, len(yte), "| features:", len(FEATS))

def gini(w_by_class):
    t = w_by_class.sum()
    return 0.0 if t <= 0 else 1.0 - float(((w_by_class / t) ** 2).sum())

def build(X, y, w, classes, depth, max_depth, min_leaf):
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[c] for c in y])
    def node(rows, d):
        cw = np.bincount(yi[rows], weights=w[rows], minlength=len(classes))
        leaf = {"leaf": classes[int(cw.argmax())], "n": int(len(rows))}
        if d >= max_depth or len(rows) < 2 * min_leaf or (cw > 0).sum() == 1: return leaf
        best = None; base = gini(cw) * cw.sum()
        for j in range(X.shape[1]):
            col = X[rows, j]
            for thr in np.unique(np.quantile(col, np.linspace(0.05, 0.95, 19))):
                m = col <= thr
                if m.sum() < min_leaf or (~m).sum() < min_leaf: continue
                l = np.bincount(yi[rows][m], weights=w[rows][m], minlength=len(classes))
                r = np.bincount(yi[rows][~m], weights=w[rows][~m], minlength=len(classes))
                score = gini(l) * l.sum() + gini(r) * r.sum()
                if best is None or score < best[0]: best = (score, j, float(thr), m)
        if best is None or best[0] >= base - 1e-9: return leaf
        _, j, thr, m = best
        return {"feat": FEATS[j], "thr": round(thr, 3), "n": int(len(rows)), "le": node(rows[m], d + 1), "gt": node(rows[~m], d + 1)}
    return node(np.arange(len(y)), 0)

def weights(y):
    c = Counter(y); n = len(y); k = len(c)
    return np.array([n / (k * c[v]) for v in y])

def predict_one(tree, fdict):
    while "leaf" not in tree:
        tree = tree["le"] if fdict.get(tree["feat"], med[FEATS.index(tree["feat"])]) <= tree["thr"] else tree["gt"]
    return tree["leaf"]

def evaluate(tree, seqlist):
    res = []
    for true, rep, hand, fr in seqlist:
        v = GraspVoter(window=5)
        for t, f in fr:
            lab = predict_one(tree, f)
            lab, _c = v.vote(hand, t, lab, [lab])
            res.append((true, lab))
    n = max(len(res), 1)
    return (sum(a == b for a, b in res) / n, sum((a in POWER) == (b in POWER) for a, b in res) / n, res)

P("\n%-7s %-24s %-24s" % ("depth", "train exact / family", "held-out exact / family"))
trees = {}
for depth in range(2, 7):
    tr = build(Xtr, ytr, weights(ytr), NAMES, 0, depth, 40); trees[depth] = tr
    a = evaluate(tr, train_seqs); b = evaluate(tr, test_seqs)
    P("%-7d %5.1f%% / %5.1f%%          %5.1f%% / %5.1f%%" % (depth, 100 * a[0], 100 * a[1], 100 * b[0], 100 * b[1]))
P("(reference on the same held-out round: current rule 0.0% / 7.5%, best confidence rule 21.6% / 51.0%)")

best_depth = max(trees, key=lambda d: (round(evaluate(trees[d], test_seqs)[0], 3), -d))
P("\nCHOSEN DEPTH (best held-out exact match, simplest on ties):", best_depth)
res = evaluate(trees[best_depth], test_seqs)[2]; by = defaultdict(list)
for a, b in res: by[a].append(b)
P("held-out round, per grasp: correct share | most common wrong label")
for g in NAMES:
    if g in by:
        c = Counter(by[g]); n = len(by[g]); wrong = [(k, v) for k, v in c.most_common() if k != g]
        P("  %-22s %5.1f%% | %s" % (g, 100 * c[g] / n, ("%s (%.0f%%)" % (wrong[0][0], 100 * wrong[0][1] / n)) if wrong else "-"))
P("\nCONFUSION ON HELD-OUT ROUND (rows = true, columns = predicted, % of frames)")
P("%-20s" % "" + "".join("%6s" % c[:5] for c in NAMES))
for g in NAMES:
    if g in by:
        c = Counter(by[g]); n = len(by[g]); P("%-20s" % g + "".join("%6.0f" % (100 * c[k] / n) for k in NAMES))

Xall = np.vstack([Xtr, Xte]); yall = np.concatenate([ytr, yte])
final = build(Xall, yall, weights(yall), NAMES, 0, best_depth, 40)
def show(t, ind=0):
    pad = "  " * ind
    if "leaf" in t: P("%s-> %s  (%d frames)" % (pad, t["leaf"], t["n"])); return
    P("%sif %s <= %s:" % (pad, t["feat"], t["thr"])); show(t["le"], ind + 1)
    P("%selse:" % pad); show(t["gt"], ind + 1)
P("\nFINAL RULES (refit on all rounds):")
show(final)
os.makedirs("data/ground_truth", exist_ok=True)
mp = os.path.abspath("data/ground_truth/grasp_tree_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
json.dump({"features": FEATS, "medians": [float(x) for x in med], "depth": best_depth, "tree": final, "classes": NAMES}, open(mp, "w"), indent=1)
os.makedirs("reports", exist_ok=True)
rp = os.path.abspath("reports/grasp_tree_%s.txt" % time.strftime("%Y%m%d_%H%M%S"))
open(rp, "w").write("\n".join(LINES))
print("\nCREATED   " + mp); print("CREATED   " + rp)
