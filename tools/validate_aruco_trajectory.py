#!/usr/bin/env python3
"""Check the platform's tracked cup path against an ArUco marker carried on the cup.

The marker (default id 2, 50 mm) is detected in the raw session video. Its centre is the reference path.
It is compared with (A) the object centres the platform logged for the tracked cup, frame by frame, and
(B) the red-trace samples saved in the trajectory results file, matched by time.

  python tools/validate_aruco_trajectory.py --label AR05
  python tools/validate_aruco_trajectory.py --session 20261102_140116 --label AR05
  python tools/validate_aruco_trajectory.py --summary

Frame numbers are 1-based, so video index = frame_number - 1. Real time comes from timestamp_ms.
Errors are in pixels of the platform frame. Millimetres use the marker's own size, so they are valid at
the marker plane only. The marker is a position reference, so a fixed offset between the marker and the
cup centre appears as bias; the centred RMSE removes that offset.
"""
import argparse, bisect, csv, glob, json, math, os, statistics, sys
import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from src.preprocessing.scale_calibration import detect_markers, nearest_video

LOG = os.path.join("reports", "aruco_validation_log.csv")
FIELDS = ["label", "session_id", "window", "marker_id", "frames_in_window", "frames_with_marker", "coverage_pct",
          "longest_gap_s", "paired_frames", "mae_px", "rmse_px", "centred_rmse_px", "bias_x_px", "bias_y_px",
          "trace_pairs", "trace_mae_px", "start_diff_px", "end_diff_px", "path_marker_px", "path_platform_px",
          "path_diff_pct", "marker_side_px", "mae_mm_at_marker_plane", "rot_rmse_px", "rot_range_deg", "offset_local_x_px",
          "offset_local_y_px", "side_change_pct"]


def newest_session():
    c = glob.glob("data/trajectory_results/*_trajectory_results.json")
    if not c:
        sys.exit("No trajectory results found in data/trajectory_results.")
    return os.path.basename(max(c, key=os.path.getmtime)).split("_trajectory_results")[0]


def read_rows(path, required=True):
    if not os.path.exists(path):
        if required:
            sys.exit("Missing file: " + path)
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def locate(frame, mid):
    """Marker corners in camera orientation. A mirrored video frame only decodes after flipping it back."""
    for img in (frame, None):
        if img is None:
            img = cv2.flip(frame, 1)
        for i, c in detect_markers(img):
            if i == mid:
                return c
    return None


def marker_track(video, needed, mid):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("Could not open video: " + video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, width, idx, last = {}, None, 0, max(needed) - 1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        width = width or frame.shape[1]
        if (idx + 1) in needed:
            c = locate(frame, mid)
            if c is not None:
                cx, cy = c.mean(axis=0)
                side = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
                out[idx + 1] = (float(cx), float(cy), side, math.atan2(float(c[1][1] - c[0][1]), float(c[1][0] - c[0][0])))
        idx += 1
    cap.release()
    return out, width, total


def err_stats(pairs):
    if not pairs:
        return None
    ex = [p[2] - p[0] for p in pairs]
    ey = [p[3] - p[1] for p in pairs]
    e = [math.hypot(a, b) for a, b in zip(ex, ey)]
    bx, by = statistics.mean(ex), statistics.mean(ey)
    return {"n": len(pairs), "mae": statistics.mean(e), "rmse": math.sqrt(statistics.mean(v * v for v in e)),
            "crmse": math.sqrt(statistics.mean((a - bx) ** 2 + (b - by) ** 2 for a, b in zip(ex, ey))),
            "bias_x": bx, "bias_y": by, "max": max(e), "p95": sorted(e)[int(0.95 * (len(e) - 1))]}


def rot_fit(rows):
    """Rotation-compensated check. rows: (marker_x, marker_y, angle, platform_x, platform_y, marker_side), all in platform orientation.
    Model: platform_centre = marker_centre + R(angle) d, with d fixed in the marker's own frame. d is the mean of R(-angle)(p - m),
    which is the least-squares answer; the residual is the error left once a rigid, rotating offset is allowed."""
    if len(rows) < 3:
        return None
    ds = []
    for mx, my, th, px, py, _ in rows:
        c, sn, dx, dy = math.cos(th), math.sin(th), px - mx, py - my
        ds.append((c * dx + sn * dy, -sn * dx + c * dy))
    d0, d1 = statistics.mean(d[0] for d in ds), statistics.mean(d[1] for d in ds)
    res = []
    for mx, my, th, px, py, _ in rows:
        c, sn = math.cos(th), math.sin(th)
        res.append(math.hypot(px - (mx + c * d0 - sn * d1), py - (my + sn * d0 + c * d1)))
    ang = np.unwrap([r[2] for r in rows])
    sides = [r[5] for r in rows]
    return {"rmse": math.sqrt(statistics.mean(v * v for v in res)), "d": (d0, d1),
            "rot_range": math.degrees(float(max(ang) - min(ang))),
            "side_change": 100.0 * (max(sides) - min(sides)) / statistics.median(sides)}


def path_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def longest_gap_s(found_ts, start, end):
    edges = [start] + sorted(found_ts) + [end]
    return max(b - a for a, b in zip(edges, edges[1:])) / 1000.0


def analyse(a):
    session = a.session or newest_session()
    traj_path = "data/trajectory_results/%s_trajectory_results.json" % session
    if not os.path.exists(traj_path):
        sys.exit("Missing file: " + traj_path)
    traj = json.load(open(traj_path))
    obj_rows = read_rows("data/raw/objects/%s_object_data.csv" % session)
    pose_rows = read_rows("data/raw/pose/%s_pose_landmarks.csv" % session, required=False)
    ft = {}
    for r in pose_rows + obj_rows:
        fn, ts = num(r.get("frame_number")), num(r.get("timestamp_ms"))
        if fn is not None and ts is not None:
            ft[int(fn)] = ts
    if not ft:
        sys.exit("No frame numbers and timestamps found in the session CSVs.")
    video = a.video or nearest_video(session)
    if not video:
        sys.exit("No raw video found for %s in data/raw/video. Pass --video FILE." % session)
    track, width, total = marker_track(video, set(ft), a.marker_id)
    print("session %s\nvideo   %s\nvideo frames %s, logged frames %d-%d (%d), marker id %d seen in %d frames" % (
        session, video, total, min(ft), max(ft), len(ft), a.marker_id, len(track)))
    if total and total != max(ft):
        print("NOTE: video has %d frames but the last logged frame_number is %d" % (total, max(ft)))
    if not track:
        sys.exit("Marker id %d was not found in this video. Check the id, lighting and that the marker faces the camera." % a.marker_id)
    sides = [t[2] for t in track.values()]
    mm_per_px = a.marker_mm / statistics.median(sides)

    attempts = traj.get("attempts") or []
    windows = []
    for k, at in enumerate(attempts, 1):
        s, e = at.get("start_timestamp_ms"), at.get("end_timestamp_ms")
        samples = [x for x in (at.get("trajectory_samples") or []) if x.get("center") and x.get("timestamp_ms") is not None]
        if s is None:
            continue
        e = e if e is not None else max(ft.values())
        ids = {str(x.get("object_id")) for x in samples}
        windows.append(("attempt_%d" % (at.get("attempt_number") or k), s, e, ids, samples))
    if not windows:
        windows = [("all", min(ft.values()), max(ft.values()), set(), [])]
    if a.object_id:
        windows = [(n, s, e, {str(a.object_id)}, sm) for n, s, e, i, sm in windows]

    frames = sorted(ft)
    found_ts = sorted((ft[f], f) for f in track)
    ts_only = [t for t, _ in found_ts]
    result, per_frame = [], []
    pooled_pairs, pooled_alt = [], []
    for name, s, e, ids, samples in windows:
        wf = [f for f in frames if s <= ft[f] <= e]
        plat = {}
        for r in obj_rows:
            fn = num(r.get("frame_number"))
            if fn is None or int(fn) not in ft or not (s <= ft[int(fn)] <= e):
                continue
            if ids and str(r.get("object_id")) not in ids:
                continue
            x, y, cf = num(r.get("object_center_x")), num(r.get("object_center_y")), num(r.get("object_confidence")) or 0
            if x is None or y is None:
                continue
            if int(fn) not in plat or cf > plat[int(fn)][2]:
                plat[int(fn)] = (x, y, cf)
        pairs, alt, rec, rotp = [], [], [], []
        for f in wf:
            if f in track:
                mx, my = (width - 1) - track[f][0], track[f][1]
                if f in plat:
                    pairs.append((mx, my, plat[f][0], plat[f][1]))
                    alt.append((track[f][0], track[f][1], plat[f][0], plat[f][1]))
                    rotp.append((mx, my, math.atan2(math.sin(track[f][3]), -math.cos(track[f][3])), plat[f][0], plat[f][1], track[f][2]))
                rec.append((f, mx, my))
            per_frame.append({"window": name, "frame_number": f, "timestamp_ms": ft[f],
                              "marker_x": round((width - 1) - track[f][0], 1) if f in track else "",
                              "marker_y": round(track[f][1], 1) if f in track else "",
                              "marker_side_px": round(track[f][2], 1) if f in track else "",
                              "object_x": plat[f][0] if f in plat else "", "object_y": plat[f][1] if f in plat else "",
                              "error_px": round(math.hypot((width - 1) - track[f][0] - plat[f][0], track[f][1] - plat[f][1]), 2) if (f in track and f in plat) else ""})
        # red trace samples matched to the nearest marker frame within 150 ms
        trace = []
        for sm in samples:
            ts = sm["timestamp_ms"]
            j = bisect.bisect_left(ts_only, ts)
            cands = [c for c in (j - 1, j) if 0 <= c < len(ts_only)]
            if not cands:
                continue
            c = min(cands, key=lambda q: abs(ts_only[q] - ts))
            if abs(ts_only[c] - ts) <= 150 and s <= ts <= e:
                f = found_ts[c][1]
                trace.append(((width - 1) - track[f][0], track[f][1], float(sm["center"][0]), float(sm["center"][1])))
        st, tr = err_stats(pairs), err_stats(trace)
        pm, pp = path_len([(p[0], p[1]) for p in pairs]), path_len([(p[2], p[3]) for p in pairs])
        ends = trace or pairs
        side_med = statistics.median([track[f][2] for f in wf if f in track]) if any(f in track for f in wf) else None
        row = {"label": a.label, "session_id": session, "window": name, "marker_id": a.marker_id,
               "frames_in_window": len(wf), "frames_with_marker": sum(1 for f in wf if f in track),
               "coverage_pct": round(100.0 * sum(1 for f in wf if f in track) / len(wf), 1) if wf else 0,
               "longest_gap_s": round(longest_gap_s([ft[f] for f in wf if f in track], s, e), 2) if wf else "",
               "paired_frames": len(pairs),
               "mae_px": round(st["mae"], 2) if st else "", "rmse_px": round(st["rmse"], 2) if st else "",
               "centred_rmse_px": round(st["crmse"], 2) if st else "",
               "bias_x_px": round(st["bias_x"], 2) if st else "", "bias_y_px": round(st["bias_y"], 2) if st else "",
               "trace_pairs": len(trace), "trace_mae_px": round(tr["mae"], 2) if tr else "",
               "start_diff_px": round(math.hypot(ends[0][2] - ends[0][0], ends[0][3] - ends[0][1]), 2) if ends else "",
               "end_diff_px": round(math.hypot(ends[-1][2] - ends[-1][0], ends[-1][3] - ends[-1][1]), 2) if ends else "",
               "path_marker_px": round(pm, 1), "path_platform_px": round(pp, 1),
               "path_diff_pct": round(100.0 * (pp - pm) / pm, 1) if pm else "",
               "marker_side_px": round(side_med, 1) if side_med else "",
               "mae_mm_at_marker_plane": round(st["mae"] * (a.marker_mm / side_med), 2) if (st and side_med) else ""}
        rf = rot_fit(rotp)
        row.update({"rot_rmse_px": round(rf["rmse"], 2) if rf else "", "rot_range_deg": round(rf["rot_range"], 1) if rf else "",
                    "offset_local_x_px": round(rf["d"][0], 1) if rf else "", "offset_local_y_px": round(rf["d"][1], 1) if rf else "",
                    "side_change_pct": round(rf["side_change"], 1) if rf else ""})
        result.append(row)
        pooled_pairs += pairs
        pooled_alt += alt
    return session, result, per_frame, err_stats(pooled_pairs), err_stats(pooled_alt), mm_per_px


def show(session, result, pooled, alt, mm_per_px):
    print("\n%-12s %6s %8s %7s %7s %8s %8s %7s %7s %9s" % ("window", "frames", "coverage", "MAE px", "RMSE", "centred", "bias x,y", "end px", "path %", "gap s"))
    for r in result:
        print("%-12s %6s %7s%% %7s %7s %8s %8s %7s %7s %9s" % (
            r["window"], r["frames_in_window"], r["coverage_pct"], r["mae_px"], r["rmse_px"], r["centred_rmse_px"],
            "%s,%s" % (r["bias_x_px"], r["bias_y_px"]), r["end_diff_px"], r["path_diff_pct"], r["longest_gap_s"]))
    for r in result:
        if r.get("rot_rmse_px") not in ("", None):
            print("  %s: rotation-compensated RMSE %s px (fixed offset only: %s px); object turned %s deg; marker size changed %s%%%s" % (
                r["window"], r["rot_rmse_px"], r["centred_rmse_px"], r["rot_range_deg"], r["side_change_pct"],
                "  <- large size change, the object may have tilted toward or away from the camera" if float(r["side_change_pct"]) > 15 else ""))
    if pooled:
        print("\nAll windows: %d paired frames, MAE %.1f px (%.1f mm at the marker plane), RMSE %.1f px, centred RMSE %.1f px, "
              "bias (%.1f, %.1f) px, 95th percentile %.1f px, worst %.1f px" % (
                  pooled["n"], pooled["mae"], pooled["mae"] * mm_per_px, pooled["rmse"], pooled["crmse"],
                  pooled["bias_x"], pooled["bias_y"], pooled["p95"], pooled["max"]))
        if alt and alt["mae"] < 0.5 * pooled["mae"]:
            print("WARNING: comparing without the mirror gives a much lower error (%.1f px). The video may already be in "
                  "platform orientation or the platform may not mirror. Tell Claude before using these numbers." % alt["mae"])
    else:
        print("\nNo frames had both a marker and a logged object centre. Check the object id and class.")


def summary():
    if not os.path.exists(LOG):
        sys.exit("No log yet: " + LOG)
    rows = list(csv.DictReader(open(LOG)))
    print("%d logged windows from %d labels\n" % (len(rows), len({r["label"] for r in rows})))
    print("%-10s %-16s %-11s %8s %7s %8s %8s %8s" % ("label", "session", "window", "coverage", "MAE px", "centred", "end px", "path %"))
    for r in rows:
        print("%-10s %-16s %-11s %7s%% %7s %8s %8s %8s" % (r["label"], r["session_id"], r["window"], r["coverage_pct"],
                                                         r["mae_px"], r["centred_rmse_px"], r["end_diff_px"], r["path_diff_pct"]))
    vals = [float(r["mae_px"]) for r in rows if r["mae_px"] not in ("", None)]
    if vals:
        print("\nMAE across windows: mean %.1f px, median %.1f px, worst %.1f px (n=%d)" % (
            statistics.mean(vals), statistics.median(vals), max(vals), len(vals)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session"); p.add_argument("--label", default="unlabelled")
    p.add_argument("--marker-id", type=int, default=2); p.add_argument("--marker-mm", type=float, default=50.0)
    p.add_argument("--video"); p.add_argument("--object-id")
    p.add_argument("--no-log", action="store_true"); p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        return summary()
    session, result, per_frame, pooled, alt, mm_per_px = analyse(a)
    show(session, result, pooled, alt, mm_per_px)
    os.makedirs("reports", exist_ok=True)
    base = "reports/aruco_%s_%s" % (session, a.label)
    with open(base + "_frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_frame[0].keys())); w.writeheader(); w.writerows(per_frame)
    json.dump({"session_id": session, "label": a.label, "marker_id": a.marker_id, "marker_mm": a.marker_mm,
               "mm_per_px_at_marker_plane": mm_per_px, "windows": result, "pooled": pooled}, open(base + ".json", "w"), indent=1)
    if not a.no_log:
        new = not os.path.exists(LOG)
        if not new:
            head = open(LOG).readline().strip().split(",")
            if head != FIELDS:
                import time as _t
                old = LOG.replace(".csv", "_before_rotation_columns_%s.csv" % _t.strftime("%Y%m%d_%H%M%S"))
                os.rename(LOG, old); new = True
                print("Log columns changed; the old log was kept as " + old)
        with open(LOG, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); new and w.writeheader()
            for r in result:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    print("\nFiles:\n  %s_frames.csv\n  %s.json%s" % (base, base, "" if a.no_log else "\n  " + LOG + " (one row per window added)"))


if __name__ == "__main__":
    main()
