#!/usr/bin/env python3
"""E30 regression replay: run a recorded session through the current build and list what changed.

  python tools/regression_replay.py --baseline      save today's result as the reference (do this on the frozen build)
  python tools/regression_replay.py                 rebuild and compare
  python tools/regression_replay.py --session 20261102_140116

It rebuilds the unified assessment from the saved trajectory results, pose landmarks and posture calibration,
the same inputs the exporter uses, without writing anything into data/processed. The result is compared with
(1) the unified assessment that was exported when the session was recorded, and
(2) the newest saved baseline for that session.
Every difference is listed with its old and new value. A difference is not automatically a fault: record each
one as expected (with the reason) or investigate it. The posture_calibration block is added by the exporter
and is not compared with the stored export.
"""
import argparse, copy, glob, importlib.util, json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, os.getcwd())
BASE_DIR = Path("data/regression_baselines")
IGNORE_VS_EXPORT = {"posture_calibration"}


def jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
    except ImportError:
        pass
    return str(o)


def newest_session():
    for p in sorted(glob.glob("data/trajectory_results/*_trajectory_results.json"), key=os.path.getmtime, reverse=True):
        sid = os.path.basename(p).split("_trajectory_results")[0]
        if os.path.exists("data/raw/pose/%s_pose_landmarks.csv" % sid):
            return sid
    sys.exit("No session with both trajectory results and pose landmarks found.")


def rebuild(sid):
    from src.assessment.trajectory_pose_adapter import load_pose_frames
    from src.assessment.posture_calibration_store import load_calibration
    from src.assessment.unified_session_assessment import build_unified_session_assessment
    tpath = Path("data/trajectory_results/%s_trajectory_results.json" % sid)
    ppath = Path("data/raw/pose/%s_pose_landmarks.csv" % sid)
    session = json.loads(tpath.read_text(encoding="utf-8"))
    if str(session.get("session_id")) != str(sid):
        sys.exit("Trajectory session id mismatch.")
    pose_frames = load_pose_frames(ppath)
    calibration = load_calibration(sid)
    reference = None
    if calibration is not None and calibration.get("status", "UNAVAILABLE") == "READY":
        reference = calibration.get("reference_posture")
        if not isinstance(reference, dict):
            sys.exit("READY calibration has no valid reference.")
    out = build_unified_session_assessment(copy.deepcopy(session), pose_frames, reference_posture=reference)
    out["schema_version"] = "0.2"   # the exporter stamps this after the builder runs (export_unified_session.py)
    return json.loads(json.dumps(out, default=jsonable))


def diff(a, b, path="", tol=1e-9, out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a: out.append((path + "/" + str(k), "(missing)", b[k]))
            elif k not in b: out.append((path + "/" + str(k), a[k], "(missing)"))
            else: diff(a[k], b[k], path + "/" + str(k), tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): out.append((path + "[length]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)): diff(x, y, "%s[%d]" % (path, i), tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        if not (both_nan or a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))): out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out


def code_hash():
    try:
        spec = importlib.util.spec_from_file_location("bf", "tools/build_freeze.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.code_hash(m.code_files())
    except Exception:
        return None


def short(v, n=70):
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 3] + "..."


def compare(label, old, new, ignore, limit):
    o = {k: v for k, v in old.items() if k not in ignore}
    n = {k: v for k, v in new.items() if k not in ignore}
    d = diff(o, n)
    print("\n%s: %s" % (label, "no differences" if not d else "%d differences" % len(d)))
    for path, a, b in d[:limit]:
        print("   %s: %s -> %s" % (path, short(a), short(b)))
    if len(d) > limit:
        print("   ... %d more in the saved report" % (len(d) - limit))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session"); ap.add_argument("--baseline", action="store_true"); ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    sid = a.session or newest_session()
    new = rebuild(sid)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print("session %s rebuilt (%d top-level fields, %d attempts)" % (sid, len(new), len(new.get("attempts") or [])))
    if a.baseline:
        folder = BASE_DIR / sid; folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("%s_baseline.json" % stamp)
        path.write_text(json.dumps({"session_id": sid, "created": stamp, "code_hash": code_hash(), "assessment": new}, indent=1))
        print("baseline saved: %s\ncode hash %s" % (path, (code_hash() or "unknown")[:16]))
        return
    results = {}
    stored = Path("data/processed/unified_sessions/%s_unified_assessment_v02.json" % sid)
    if stored.exists():
        results["vs_stored_export"] = compare("Rebuilt now vs the export made when the session was recorded (%s)" % stored.name,
                                              json.loads(stored.read_text(encoding="utf-8")), new, IGNORE_VS_EXPORT, a.limit)
    else:
        print("\nNo stored export for this session (%s)." % stored)
    bases = sorted(glob.glob(str(BASE_DIR / sid / "*_baseline.json")))
    if bases:
        b = json.load(open(bases[-1]))
        results["vs_baseline"] = compare("Rebuilt now vs baseline %s (code hash then %s, now %s)" % (
            os.path.basename(bases[-1]), (b.get("code_hash") or "unknown")[:12], (code_hash() or "unknown")[:12]), b["assessment"], new, set(), a.limit)
    else:
        print("\nNo baseline yet. Run with --baseline on the frozen build to create one.")
    total = sum(len(v) for v in results.values())
    os.makedirs("reports", exist_ok=True)
    rep = "reports/regression_replay_%s_%s.json" % (sid, stamp)
    json.dump({"session_id": sid, "generated": stamp, "differences": {k: [{"path": p, "old": x, "new": y} for p, x, y in v] for k, v in results.items()}},
              open(rep, "w"), indent=1, default=str)
    print("\nRESULT: %s\nReport: %s" % ("no differences, no regression" if not total and results else
                                      ("%d differences to record as expected or investigate" % total if results else "nothing to compare"), rep))
    sys.exit(0 if not total else 1)


if __name__ == "__main__":
    main()
