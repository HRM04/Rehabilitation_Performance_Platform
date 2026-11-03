"""Pixel to millimetre scale from an ArUco marker.

Print the marker (make-marker), keep it flat on the task plane in view at the
start of a session, then run `auto`. The scale is the marker's printed side
length divided by its side length in pixels. It is valid only at the depth of
the marker plane, so objects well above or below that plane carry extra error.

Commands (run from the project folder):
  python -m src.preprocessing.scale_calibration make-marker [--size-mm 50] [--id 0]
  python -m src.preprocessing.scale_calibration auto [--marker-mm 50]
  python -m src.preprocessing.scale_calibration from-video VIDEO [--session ID] [--marker-mm 50]
"""
import argparse
import glob
import json
import os
import re
import statistics

import cv2
import numpy as np

DICT_NAME = "DICT_4X4_50"
SCALE_DIR = os.path.join("data", "processed", "scale_calibration")


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def detect_markers(frame):
    d = _dictionary()
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, d)
    out = []
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out.append((int(i), np.asarray(c).reshape(4, 2)))
    return out


def side_stats(c):
    sides = [float(np.linalg.norm(c[k] - c[(k + 1) % 4])) for k in range(4)]
    mean_side = sum(sides) / 4.0
    skew = max(sides) / max(min(sides), 1e-9)
    return mean_side, skew


def make_marker(size_mm=50.0, marker_id=0, margin_mm=10.0, dpi=300,
                out_dir="assets"):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    total_mm = size_mm + 2 * margin_mm
    total_px = int(round(total_mm / 25.4 * dpi))
    marker_px = int(round(size_mm / 25.4 * dpi))
    d = _dictionary()
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(d, marker_id, marker_px)
    else:
        m = cv2.aruco.drawMarker(d, marker_id, marker_px)
    canvas = np.full((total_px, total_px), 255, np.uint8)
    off = (total_px - marker_px) // 2
    canvas[off:off + marker_px, off:off + marker_px] = m
    path = os.path.abspath(os.path.join(out_dir, "aruco_%s_id%d_%dmm.png" % (DICT_NAME, marker_id, int(size_mm))))
    Image.fromarray(canvas).save(path, dpi=(dpi, dpi))
    return path


def scale_from_video(video_path, marker_mm=50.0, marker_id=0, max_frames=450, step=5):
    cap = cv2.VideoCapture(video_path)
    sides, skews, used, seen = [], [], 0, 0
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            seen += 1
            for mid, c in detect_markers(frame):
                if mid == marker_id:
                    s, k = side_stats(c)
                    sides.append(s)
                    skews.append(k)
                    used += 1
                    break
        n += 1
    cap.release()
    if len(sides) < 5:
        return {"status": "MARKER_NOT_FOUND", "frames_checked": seen,
                "frames_with_marker": used, "video": os.path.abspath(video_path)}
    med = statistics.median(sides)
    spread = (statistics.pstdev(sides) / med) if med else None
    return {
        "status": "READY",
        "video": os.path.abspath(video_path),
        "marker_id": marker_id,
        "marker_size_mm": marker_mm,
        "marker_side_px_median": med,
        "px_per_mm": med / marker_mm,
        "mm_per_px": marker_mm / med,
        "side_spread_fraction": spread,
        "max_side_ratio_skew": max(skews),
        "frames_checked": seen,
        "frames_with_marker": used,
        "note": "Valid at the marker plane only. Objects nearer the camera appear larger.",
    }


def session_of(path):
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def _secs(sid):
    h = sid.split("_")[1]
    return int(h[:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:6])


def nearest_video(session_id, window_s=180):
    best = None
    for v in glob.glob(os.path.join("data", "raw", "video", "*_raw_camera.mp4")):
        vid = session_of(v)
        if vid.split("_")[0] != session_id.split("_")[0]:
            continue
        dt = _secs(vid) - _secs(session_id)
        if 0 <= dt <= window_s and (best is None or dt < best[0]):
            best = (dt, v)
    return best[1] if best else None


def save_scale(session_id, result):
    os.makedirs(SCALE_DIR, exist_ok=True)
    path = os.path.abspath(os.path.join(SCALE_DIR, "%s_scale.json" % session_id))
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def convert_px_to_mm(obj, mm_per_px):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = convert_px_to_mm(v, mm_per_px)
            if k.endswith("_px") and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k[:-3] + "_mm"] = v * mm_per_px
        return out
    if isinstance(obj, list):
        return [convert_px_to_mm(x, mm_per_px) for x in obj]
    return obj


def apply_to_trajectory(session_id, scale):
    src = os.path.join("data", "trajectory_results", "%s_trajectory_results.json" % session_id)
    if not os.path.exists(src):
        return None
    with open(src) as f:
        data = json.load(f)
    data = convert_px_to_mm(data, scale["mm_per_px"])
    data["scale_calibration"] = {k: scale[k] for k in ("marker_size_mm", "mm_per_px", "px_per_mm", "side_spread_fraction", "note")}
    dst = os.path.abspath(os.path.join(SCALE_DIR, "%s_trajectory_results_mm.json" % session_id))
    with open(dst, "w") as f:
        json.dump(data, f, indent=2)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make-marker")
    a.add_argument("--size-mm", type=float, default=50.0)
    a.add_argument("--id", type=int, default=0)
    b = sub.add_parser("auto")
    b.add_argument("--marker-mm", type=float, default=50.0)
    b.add_argument("--id", type=int, default=0)
    c = sub.add_parser("from-video")
    c.add_argument("video")
    c.add_argument("--session", default=None)
    c.add_argument("--marker-mm", type=float, default=50.0)
    c.add_argument("--id", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cmd == "make-marker":
        p = make_marker(args.size_mm, args.id)
        print("CREATED:", p)
        print("Print at 100%% scale (no fit-to-page). Measure the black square with a ruler;")
        print("if it is not %.0f mm, pass the measured value as --marker-mm." % args.size_mm)
        return 0

    if args.cmd == "from-video":
        sid = args.session or session_of(args.video)
        res = scale_from_video(args.video, args.marker_mm, args.id)
        print("RESULT:", json.dumps(res, indent=2))
        if res["status"] == "READY":
            print("CREATED:", save_scale(sid, res))
            out = apply_to_trajectory(sid, res)
            if out:
                print("CREATED:", out)
        return 0

    rows = []
    for tr in sorted(glob.glob(os.path.join("data", "trajectory_results", "*_trajectory_results.json"))):
        sid = os.path.basename(tr).replace("_trajectory_results.json", "")
        v = nearest_video(sid)
        if not v:
            rows.append((sid, "NO_VIDEO_FOUND", None, None))
            continue
        res = scale_from_video(v, args.marker_mm, args.id)
        if res["status"] == "READY":
            print("CREATED:", save_scale(sid, res))
            out = apply_to_trajectory(sid, res)
            if out:
                print("CREATED:", out)
            rows.append((sid, "READY", "%.4f" % res["mm_per_px"], "%.3f" % (res["side_spread_fraction"] or 0)))
        else:
            rows.append((sid, res["status"], None, None))
    print("\nsession | status | mm_per_px | side_spread")
    for r in rows:
        print(" | ".join(str(x) for x in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
