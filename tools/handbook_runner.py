#!/usr/bin/env python3
"""One status line per handbook engineering test (E01-E42), with the evidence behind it.

  python tools/handbook_runner.py                 run the test suite and the export check
  python tools/handbook_runner.py --dashboards    also open every dashboard headless (slower)

Statuses
  PASS     automated tests mapped to this ID ran and all passed
  FAIL     at least one mapped test or check failed
  PARTIAL  something automated passed, a manual part remains
  AUDIT    the source audit produced evidence; a person signs it off in the result sheet
  LIVE     needs a live run with the camera and a person
  GAP      nothing automated exists yet
Basis "tag" means the test name contains the E-ID. Basis "name" means it was matched by keyword, so read
the matched test names in the saved report before quoting it.
"""
import argparse, csv, glob, json, math, os, re, subprocess, sys, time

TITLES = {
 1: "Enumerate active object library", 2: "Enumerate task mappings", 3: "Handedness/flip audit", 4: "Tracker runtime config",
 5: "Smoothing investigation", 6: "Grasp detector audit", 7: "Export availability", 8: "UI implementation check",
 9: "Bounding-box geometry", 10: "Duplicate threshold boundaries", 11: "Legitimate two-object retention",
 12: "Hand/object association", 13: "Temporal state transitions", 14: "Missing hand semantics",
 15: "Grasp-specific positive cases", 16: "Grasp negative/boundary cases", 17: "Appropriateness decision table",
 18: "Grasp-maintenance arithmetic", 19: "Grasp reliability coverage", 20: "Regrasp and interruption", 21: "Path metrics",
 22: "Joint angle and screening", 23: "Posture calibration", 24: "Repetition acceptance", 25: "Task completion",
 26: "Recovery boundaries", 27: "Baseline comparison", 28: "Feedback thresholds", 29: "Data assembly",
 30: "Regression replay", 31: "Export data integrity", 32: "UI export agreement", 33: "Class-level object set",
 34: "Same-class duplicate case", 35: "Identity continuity", 36: "Occlusion duration ladder", 37: "Hand stability and viewpoint",
 38: "Controlled posture motions", 39: "Orientation checkpoint challenge", 40: "Throughput challenge",
 41: "Power and precision grasp set", 42: "Stop/restart robustness"}
EXPLICIT = {
 11: r"TestDuplicateFilter::test_(adjacent_same_class_objects_kept|two_separate_same_class_objects_kept|different_classes_same_box_kept)",
 13: r"test_grasp_voting\.py",
 14: r"test_(missing_frames_do_not_release|missing_data_prevents_score|missing_observations|missing_observation_does_not_create_regrasp)",
 22: r"TestJointScreening", 24: r"TestRepetitions",
 27: r"TestBaselineAndFeedback::test_(relative_change|no_comparison_gives_no_feedback|improvement_not_flagged|missing_values_trigger_nothing)",
 28: r"TestBaselineAndFeedback::test_(duration_boundary|rom_boundary|speed_trunk_reversal|missing_values_trigger_nothing)"}
KEYWORDS = {18: r"quality|maintenance", 19: r"reliab|coverage", 20: r"regrasp|interrupt", 21: r"path|metric|reliab", 26: r"recover|identity|grace"}
AUDIT = {1, 2, 3, 4, 5, 6, 7}
LIVE = set(range(33, 43))


def run_pytest():
    p = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-rA", "-p", "no:cacheprovider"], capture_output=True, text=True)
    res = {}
    for line in p.stdout.splitlines():
        m = re.match(r"(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", line)
        if m:
            res[m.group(2)] = m.group(1)
    tail = [l for l in p.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    return res, (tail[-1] if tail else "no summary line")


def newest(pat):
    fs = [f for f in glob.glob(pat) if os.path.isfile(f) and "volunteer_testing" not in f and "backup" not in f]
    return max(fs, key=os.path.getmtime) if fs else None


def export_check():
    """E31: structural checks on the newest session's exports."""
    notes, bad, sids = [], [], {}
    specs = (("objects", "data/raw/objects/*_object_data.csv"), ("interactions", "data/raw/interactions/*_interaction_data.csv"),
             ("landmarks", "data/raw/landmarks/*_landmarks.csv"), ("pose", "data/raw/pose/*_pose_landmarks.csv"))
    found = 0
    for label, pat in specs:
        f = newest(pat)
        if not f:
            bad.append("%s: no file" % label); continue
        found += 1
        rows = list(csv.reader(open(f, errors="ignore", newline="")))
        head, body = rows[0], rows[1:]
        ragged = sum(1 for r in body if len(r) != len(head))
        d = [dict(zip(head, r)) for r in body]
        nan = sum(1 for r in d for v in r.values() if str(v).lower() in ("nan", "inf", "-inf"))
        dup = len(body) - len({tuple(r) for r in body})
        ts = [float(r["timestamp_ms"]) for r in d if r.get("timestamp_ms")]
        back = sum(1 for a, b in zip(ts, ts[1:]) if b < a)
        s = {r.get("session_id") for r in d if r.get("session_id")}
        if s: sids[label] = s
        problems = [n for n, v in (("ragged rows", ragged), ("NaN/inf", nan), ("duplicate rows", dup), ("time reversals", back)) if v]
        notes.append("%s %s rows=%d cols=%d %s" % (label, os.path.basename(f), len(body), len(head), ("PROBLEMS: " + ", ".join(problems)) if problems else "clean"))
        if problems: bad.append("%s: %s" % (label, ", ".join(problems)))
    tf = newest("data/trajectory_results/*_trajectory_results.json")
    if tf:
        j = json.load(open(tf)); sids["trajectory"] = {j.get("session_id")}
        for a in j.get("attempts", []):
            m = a.get("metrics") or {}
            nonfinite = [k for k, v in m.items() if isinstance(v, float) and not math.isfinite(v)]
            if nonfinite: bad.append("trajectory attempt %s non-finite metrics %s" % (a.get("attempt_number"), nonfinite))
        notes.append("trajectory %s attempts=%d" % (os.path.basename(tf), len(j.get("attempts", []))))
    for label, pat in (("calibration", "data/processed/posture_calibration/*_posture_calibration.json"), ("unified", "data/processed/unified_sessions/*_unified_*.json")):
        f = newest(pat)
        if f: sids[label] = {json.load(open(f)).get("session_id")}
    allids = set().union(*sids.values()) if sids else set()
    if len(allids) > 1: bad.append("session ids differ across files: %s" % {k: sorted(map(str, v)) for k, v in sids.items()})
    notes.append("session ids: %s" % {k: sorted(map(str, v)) for k, v in sids.items()})
    if found == 0: return "GAP", ["no export files found"] + bad
    return ("FAIL" if bad else "PASS"), notes + bad


def assembly_check():
    """E29: do the recorded files for the newest session link up (session, frames, objects, repetitions)?"""
    tf = newest("data/trajectory_results/*_trajectory_results.json")
    of = newest("data/raw/objects/*_object_data.csv")
    inf = newest("data/raw/interactions/*_interaction_data.csv")
    if not (tf and of and inf):
        return "GAP", ["needs trajectory, objects and interactions files for the same session"]
    traj = json.load(open(tf)); sid = traj.get("session_id")
    objs = list(csv.DictReader(open(of, errors="ignore", newline="")))
    ints = list(csv.DictReader(open(inf, errors="ignore", newline="")))
    bad, notes = [], []
    for label, rows in (("objects", objs), ("interactions", ints)):
        ids = {r.get("session_id") for r in rows}
        if ids != {sid}: bad.append("%s session ids %s differ from trajectory %s" % (label, sorted(map(str, ids)), sid))
    ot = {}
    for r in objs: ot.setdefault(r["frame_number"], set()).add(r["timestamp_ms"])
    if any(len(v) > 1 for v in ot.values()): bad.append("objects: a frame number has more than one timestamp")
    only_int = sum(1 for r in ints if r["frame_number"] not in ot)
    clock = sum(1 for r in ints if r["frame_number"] in ot and r["timestamp_ms"] not in ot[r["frame_number"]])
    if only_int: bad.append("%d interaction rows have a frame that is not in the objects file" % only_int)
    if clock: bad.append("%d interaction rows disagree with the objects file on timestamp" % clock)
    oid = {(r["frame_number"], r["object_id"]) for r in objs if r.get("object_id")}
    orphan = sum(1 for r in ints if r.get("object_id") and (r["frame_number"], r["object_id"]) not in oid)
    if orphan: bad.append("%d interaction rows name an object id that is not in the objects file for that frame" % orphan)
    tasks_o = {r.get("task_id") for r in objs if r.get("task_id")}; tasks_i = {r.get("task_id") for r in ints if r.get("task_id")}
    if tasks_o and tasks_i and tasks_o != tasks_i: bad.append("task ids differ: objects %s, interactions %s" % (sorted(tasks_o), sorted(tasks_i)))
    reps_i = {r.get("repetition_number") for r in ints if r.get("repetition_number")}
    reps_t = {str(a.get("repetition_number")) for a in traj.get("attempts", [])}
    if reps_i - reps_t: bad.append("interaction repetitions %s have no trajectory attempt (attempts have %s)" % (sorted(reps_i), sorted(reps_t)))
    sample_ids = {str(x.get("object_id")) for a in traj.get("attempts", []) for x in (a.get("trajectory_samples") or [])}
    missing_ids = sample_ids - {r.get("object_id") for r in objs}
    if missing_ids: bad.append("trajectory object ids %s are not in the objects file" % sorted(missing_ids))
    notes.append("session %s, %d object rows, %d interaction rows, repetitions %s, trajectory object ids %s" % (sid, len(objs), len(ints), sorted(reps_i), sorted(sample_ids)))
    notes.append("structural linkage only; replay through the assessment builders is E30")
    return ("FAIL" if bad else "PASS"), notes + bad


def dashboards():
    import logging
    sys.path.insert(0, os.getcwd())
    logging.disable(logging.CRITICAL)
    from streamlit.testing.v1 import AppTest
    out, ok = [], True
    for f in sorted(glob.glob("src/dashboard/*.py")):
        if "backup" in f.lower() or f.endswith("__init__.py") or "visualization" in f: continue
        try:
            at = AppTest.from_file(os.path.abspath(f), default_timeout=90).run()
            out.append("%s exceptions=%d warnings=%d" % (os.path.basename(f), len(at.exception), len(at.warning)))
            ok = ok and len(at.exception) == 0
        except Exception as e:
            out.append("%s ERROR %r" % (f, e)); ok = False
    return ("PARTIAL" if ok else "FAIL"), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dashboards", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res, tail = run_pytest()
    rows, detail = [], {}
    audits = sorted(glob.glob("reports/readiness_audit*.txt"))
    for n in range(1, 43):
        eid = "E%02d" % n
        tag = [k for k in res if re.search(r"E%02d(?!\d)" % n, k)]
        ex = [k for k in res if n in EXPLICIT and re.search(EXPLICIT[n], k)] if not tag else []
        kw = [k for k in res if n in KEYWORDS and re.search(KEYWORDS[n], k, re.I)] if not (tag or ex) else []
        matched, basis = (tag, "tag") if tag else ((ex, "map") if ex else (kw, "name"))
        status, ev = "GAP", "no automated test or check"
        if matched:
            states = [res[k] for k in matched]
            bad = [k for k in matched if res[k] in ("FAILED", "ERROR")]
            status = "FAIL" if bad else ("PASS" if all(s in ("PASSED", "XPASS") for s in states) else "PARTIAL")
            if status == "PASS" and basis == "name": status = "PASS*"
            ev = "%d tests, %d passed%s" % (len(matched), sum(s == "PASSED" for s in states), (", failed: " + ", ".join(bad[:3])) if bad else "")
            detail[eid] = matched
        elif n in AUDIT:
            status, basis = ("AUDIT", "file") if audits else ("GAP", "-")
            ev = ("source audit output in %d files, sign off in the result sheet" % len(audits)) if audits else "run the readiness audits first"
        elif n in LIVE:
            status, basis, ev = "LIVE", "-", "needs a live run (see handbook)"
        rows.append([eid, TITLES[n], status, basis if matched or n in AUDIT else "-", ev])
    s31, notes31 = export_check()
    rows[30][2:5] = [s31, "check", "; ".join(notes31)[:230]]; detail["E31"] = notes31
    s29, notes29 = assembly_check(); rows[28][2:5] = [s29, "check", "; ".join(notes29)[:230]]; detail["E29"] = notes29
    if a.dashboards:
        s8, notes8 = dashboards(); rows[7][2:5] = [s8, "check", "dashboards render; orientation prompts still manual. " + "; ".join(notes8)[:150]]; detail["E08"] = notes8
    counts = {}
    for r in rows: counts[r[2]] = counts.get(r[2], 0) + 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lines = ["Handbook engineering status  %s   (pytest: %s)" % (time.strftime("%Y-%m-%d %H:%M"), tail), ""]
    lines += ["%-4s %-34s %-8s %-5s %s" % tuple(r) for r in rows]
    lines += ["", "Totals: " + ", ".join("%s %d" % kv for kv in sorted(counts.items())), "* PASS* = matched by keyword only. Check the test names below before quoting it.", "", "Matched tests per ID:"]
    for k, v in detail.items(): lines.append("  %s: %s" % (k, "; ".join(v) if isinstance(v, list) else v))
    os.makedirs("reports", exist_ok=True)
    open("reports/handbook_status_%s.txt" % stamp, "w").write("\n".join(lines) + "\n")
    json.dump({"generated": stamp, "pytest": tail, "rows": rows, "matched": detail}, open("reports/handbook_status_%s.json" % stamp, "w"), indent=1)
    print("\n".join(lines[:2 + len(rows) + 4]))
    print("\nFiles:\n  reports/handbook_status_%s.txt\n  reports/handbook_status_%s.json   (%.0f s)" % (stamp, stamp, time.time() - t0))


if __name__ == "__main__":
    main()
