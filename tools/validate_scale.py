"""Check the pixel to millimetre scale against a known distance.

Print assets/aruco_scale_sheet_letter.pdf at 100%. Two 50 mm markers (id 0, id 1) sit 150 mm apart,
centre to centre. The scale comes from the marker sides and is used to measure that 150 mm gap.

  python tools/validate_scale.py --label flat --record 12     # record from the camera, then measure
  python tools/validate_scale.py --label flat --video FILE    # measure an existing video

Tolerance, fixed before any result: median error within 2% of the known distance.
"""
import argparse, json, os, sys, time
import cv2
sys.path.insert(0, os.getcwd())
from src.preprocessing.scale_validation import measure_pair, summarise


def _detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def _find(det, frame):
    corners, ids, _ = det.detectMarkers(frame)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = [tuple(map(float, p)) for p in c.reshape(4, 2)]
    return out, corners, ids


def record(a, det):
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened(): sys.exit("Camera did not open. Close other apps using it and try again.")
    ok, f = cap.read()
    if not ok: sys.exit("No frame from the camera.")
    h, w = f.shape[:2]
    print("camera resolution: %dx%d" % (w, h))
    os.makedirs(os.path.join("data", "raw", "video"), exist_ok=True)
    path = os.path.join("data", "raw", "video", "scale_check_%s_%s.mp4" % (a.label, time.strftime("%Y%m%d_%H%M%S")))
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    pairs, n = [], 0
    t0 = time.time()
    while time.time() - t0 < a.record:
        ok, frame = cap.read()
        if not ok: break
        wr.write(frame); n += 1
        found, corners, ids = _find(det, frame)
        both = 0 in found and 1 in found
        if both: pairs.append(measure_pair(found[0], found[1], a.marker_mm))
        try:
            view = frame.copy()
            if ids is not None: cv2.aruco.drawDetectedMarkers(view, corners, ids)
            view = cv2.flip(view, 1)
            msg = "BOTH MARKERS FOUND" if both else "looking for markers 0 and 1"
            cv2.putText(view, "%s  %.0fs left (Q stops)" % (msg, a.record - (time.time() - t0)), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("scale check", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")): break
        except Exception:
            pass
    cap.release(); wr.release()
    try: cv2.destroyAllWindows()
    except Exception: pass
    print("frames recorded: %d | frames with both markers: %d" % (n, len(pairs)))
    return path, pairs, (w, h)


def from_video(a, det):
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): sys.exit("Cannot open " + a.video)
    pairs, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        i += 1
        if i % 2: continue
        found, _, _ = _find(det, frame)
        if 0 in found and 1 in found: pairs.append(measure_pair(found[0], found[1], a.marker_mm))
    w, h = int(cap.get(3)), int(cap.get(4))
    print("frames read: %d | frames with both markers: %d" % (i, len(pairs)))
    return a.video, pairs, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="flat")
    ap.add_argument("--record", type=float, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--known-mm", type=float, default=150.0)
    ap.add_argument("--marker-mm", type=float, default=50.0)
    a = ap.parse_args()
    if not a.record and not a.video: sys.exit("Give --record SECONDS or --video FILE")
    det = _detector()
    path, pairs, size = record(a, det) if a.record else from_video(a, det)
    res = summarise(pairs, a.known_mm, 0.02)
    print("\n======== RESULT (%s) ========" % a.label)
    for k, v in res.items(): print("  %-26s %s" % (k, ("%.3f" % v) if isinstance(v, float) else v))
    print("  (tolerance was fixed beforehand at 2%% of %.0f mm)" % a.known_mm)
    ex = None
    try:
        from src.preprocessing.scale_calibration import scale_from_video
        ex = scale_from_video(path, a.marker_mm, 0)
        keep = {k: ex[k] for k in ("status", "mm_per_px", "px_per_mm", "side_spread_fraction", "note") if k in ex}
        print("\n======== EXISTING TOOL ON THE SAME VIDEO ========")
        for k, v in keep.items(): print("  %-26s %s" % (k, v))
        if keep.get("mm_per_px") and pairs:
            mine = res["mean_mm_per_px"]
            print("  difference between the two scale estimates: %.2f%%" % (100.0 * (keep["mm_per_px"] - mine) / mine))
    except Exception as e:
        print("\nexisting tool cross-check unavailable:", e)
    os.makedirs(os.path.join("data", "processed", "scale_calibration"), exist_ok=True)
    out = os.path.join("data", "processed", "scale_calibration", "validation_%s_%s.json" % (a.label, time.strftime("%Y%m%d_%H%M%S")))
    json.dump({"label": a.label, "video": os.path.abspath(path), "resolution": size, "result": res}, open(out, "w"), indent=2, default=str)
    print("\n======== FILE LOCATIONS ========")
    print("VIDEO   ", os.path.abspath(path))
    print("RESULT  ", os.path.abspath(out))


if __name__ == "__main__":
    main()
