#!/usr/bin/env python3
"""ArUco pre-check: is this object usable as an ArUco trajectory carrier with the marker taped on?

  1. Tape marker ID 2 on the object.
  2. Run a short Volunteer session (about 15 s): hold the object in view, then slide it across and turn it a little.
  3. python tools/aruco_precheck.py --object spoon

It uses the platform's own recorded output (objects CSV) and the raw video of that session, so it tests exactly what
the real trial would record. It does not change anything in data/. Result lines:
  marker seen        the marker is found in enough frames
  object kept        the platform kept detecting the object with the right label while the marker was visible
  one identity       the platform kept one object id (an id change breaks the recorded path)
  marker is rigid    after allowing the offset to turn with the object, the platform centre follows the marker closely
                     (a loose or flexing marker, or a jumping centre, shows here; this is a review flag, not a fail)
"""
import argparse, csv, glob, importlib.util, math, os, statistics, sys, time
from collections import Counter

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("val", "tools/validate_aruco_trajectory.py")
val = importlib.util.module_from_spec(spec); spec.loader.exec_module(val)
from src.preprocessing.scale_calibration import nearest_video

LOG = "reports/aruco_precheck_log.csv"
THRESH = {"marker_pct": 90.0, "object_pct": 90.0, "id_pct": 95.0, "rigid_px": 30.0}


def newest_objects_session():
    c = glob.glob("data/raw/objects/*_object_data.csv")
    if not c:
        sys.exit("No object CSV found in data/raw/objects. Run a short session first.")
    return os.path.basename(max(c, key=os.path.getmtime)).split("_object_data")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True, help="object class name, for example cup or spoon")
    ap.add_argument("--session"); ap.add_argument("--video"); ap.add_argument("--marker-id", type=int, default=2)
    a = ap.parse_args()
    sid = a.session or newest_objects_session()
    rows = val.read_rows("data/raw/objects/%s_object_data.csv" % sid)
    ft = {}
    for r in rows:
        fn, ts = val.num(r.get("frame_number")), val.num(r.get("timestamp_ms"))
        if fn is not None and ts is not None:
            ft[int(fn)] = ts
    if not ft:
        sys.exit("No frame numbers in the object CSV.")
    video = a.video or nearest_video(sid)
    if not video:
        sys.exit("No raw video found for %s in data/raw/video. Pass --video FILE." % sid)
    print("session %s, object '%s', video %s" % (sid, a.object, os.path.basename(video)))
    track, width, total = val.marker_track(video, set(ft), a.marker_id)
    if not track:
        verdict(a, sid, {"marker_pct": 0.0, "object_pct": None, "id_pct": None, "rigid_px": None, "n": 0, "rot": None})
        return
    lo, hi = min(track), max(track)
    span = [f for f in sorted(ft) if lo <= f <= hi]
    obj = {}
    for r in rows:
        if str(r.get("object_class") or r.get("class_name") or "").strip().lower() != a.object.lower():
            continue
        fn = val.num(r.get("frame_number")); x, y = val.num(r.get("object_center_x")), val.num(r.get("object_center_y"))
        cf = val.num(r.get("object_confidence")) or 0
        if fn is None or x is None or y is None:
            continue
        fn = int(fn)
        if fn not in obj or cf > obj[fn][2]:
            obj[fn] = (x, y, cf, r.get("object_id"))
    marker_pct = 100.0 * sum(1 for f in span if f in track) / len(span)
    in_marker = [f for f in span if f in track]
    object_pct = 100.0 * sum(1 for f in in_marker if f in obj) / len(in_marker)
    ids = Counter(obj[f][3] for f in span if f in obj)
    id_pct = 100.0 * max(ids.values()) / sum(ids.values()) if ids else None
    rotp = [((width - 1) - track[f][0], track[f][1], math.atan2(math.sin(track[f][3]), -math.cos(track[f][3])), obj[f][0], obj[f][1], track[f][2])
            for f in in_marker if f in obj]
    rf = val.rot_fit(rotp)
    print("frames %d-%d: marker in %d of %d, object found in %d of the marker frames, ids seen %s" % (lo, hi, len(in_marker), len(span),
          sum(1 for f in in_marker if f in obj), dict(ids)))
    verdict(a, sid, {"marker_pct": marker_pct, "object_pct": object_pct, "id_pct": id_pct,
                     "rigid_px": rf["rmse"] if rf else None, "n": len(rotp), "rot": rf})


def verdict(a, sid, m):
    def line(name, value, limit, unit, low_is_bad=True, review=False):
        if value is None:
            print("  %-16s no data" % name); return "FAIL"
        ok = value >= limit if low_is_bad else value <= limit
        tag = "PASS" if ok else ("REVIEW" if review else "FAIL")
        print("  %-16s %-6s %.1f%s (limit %s%.0f%s)" % (name, tag, value, unit, ">=" if low_is_bad else "<=", limit, unit)); return tag
    print("\nPre-check for %s:" % a.object)
    t = [line("marker seen", m["marker_pct"], THRESH["marker_pct"], "%"),
         line("object kept", m["object_pct"], THRESH["object_pct"], "%"),
         line("one identity", m["id_pct"], THRESH["id_pct"], "%"),
         line("marker is rigid", m["rigid_px"], THRESH["rigid_px"], " px", low_is_bad=False, review=True)]
    rf = m.get("rot")
    if rf:
        print("  turn range %.0f deg, marker size change %.1f%%, offset in marker frame (%.1f, %.1f) px%s" % (
            rf["rot_range"], rf["side_change"], rf["d"][0], rf["d"][1], "  (tilt suspected)" if rf["side_change"] > 15 else ""))
        if rf["rot_range"] < 20:
            print("  note: the object barely turned in this run; the rotation check is weak. Turn it by about 30 degrees and repeat if rotation will occur in the trials.")
    overall = "NO" if "FAIL" in t else ("REVIEW" if "REVIEW" in t else "YES")
    print("\nUSE %s AS AN ARUCO CARRIER: %s" % (a.object.upper(), overall))
    if overall == "NO":
        print("Fix: move the marker to a flatter, more visible face, keep it away from the grasp area, or use the cup. Do not use this object for the ArUco block.")
    os.makedirs("reports", exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "session_id", "object", "marker_pct", "object_pct", "id_pct", "rigid_rmse_px", "verdict"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), sid, a.object] + [("" if m[k] is None else round(m[k], 1)) for k in ("marker_pct", "object_pct", "id_pct", "rigid_px")] + [overall])
    print("Logged: " + LOG)


if __name__ == "__main__":
    main()
