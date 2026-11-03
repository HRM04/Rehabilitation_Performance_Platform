"""Record labelled grasps for accuracy testing.

The screen names a grasp and how to hold it. You get 5 seconds to get ready,
then 4 seconds are recorded, then 1 second of rest. Every grasp is repeated
for several rounds. Landmarks are saved with the true grasp name.

Run from the project folder:
    python tools/record_labelled_grasps.py [--reps 3] [--hand Right] [--camera 0]

Press q to quit early, s to skip the current grasp.
"""
import argparse, csv, os, sys, time

import cv2
import mediapipe as mp
import sys as _s, os as _o; _s.path.insert(0, _o.getcwd())
from src.preprocessing.handedness import final_label

GRASPS = [
    ("cylindrical", "Cylindrical", "Wrap all fingers around a bottle or can, thumb around the other side."),
    ("spherical", "Spherical", "Hold a ball or apple with fingers spread and curved around it."),
    ("large_palmar", "Large palmar", "Hold a wide, flat or large object (book, big box) with an open palm."),
    ("whole_hand_power", "Whole-hand power", "Close the whole hand firmly around a thick handle or bottle neck."),
    ("handle", "Handle", "Hold a cup or mug by its handle."),
    ("hook", "Hook", "Hook your fingers over a bag handle or bar. Thumb stays out."),
    ("pincer", "Pincer", "Pick up a small item between thumb and index finger, fingers curved."),
    ("tip_pinch", "Tip pinch", "Hold a very small item between the tips of thumb and index finger."),
    ("tripod", "Tripod", "Hold a pen like writing: thumb, index and middle finger."),
    ("lateral_pinch", "Lateral pinch", "Hold a key or card between thumb and the side of the index finger."),
    ("multidigit_precision", "Multi-digit precision", "Hold a spoon or small item with the thumb and several fingertips."),
]
PREP_S, REC_S, REST_S = 5.0, 4.0, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", default="Right", help="which hand you will use (for the on-screen reminder)")
    args = ap.parse_args()

    os.makedirs(os.path.join("data", "ground_truth"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join("data", "ground_truth", "grasp_labels_%s.csv" % ts))
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["session", "true_grasp", "rep", "frame_number", "timestamp_ms", "hand_label", "landmark_name", "x_norm", "y_norm", "z_rel"])

    mph = mp.solutions.hands
    hands = mph.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.isOpened():
        print("Could not open camera"); sys.exit(1)

    plan = [(rep, g) for rep in range(1, args.reps + 1) for g in GRASPS]
    t0 = time.perf_counter()
    frame_no = 0
    counts = {}
    quit_all = False
    for rep, (key, title, hint) in plan:
        if quit_all:
            break
        phase_start = time.perf_counter()
        total = PREP_S + REC_S + REST_S
        skip = False
        while True:
            el = time.perf_counter() - phase_start
            if el >= total or skip:
                break
            ok, frame = cam.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            frame_no += 1
            recording = PREP_S <= el < PREP_S + REC_S
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hl, mph.HAND_CONNECTIONS)
            if recording and res.multi_hand_landmarks:
                tms = int((time.perf_counter() - t0) * 1000)
                for hl, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    label = final_label(label)   # same swap as the main pipeline
                    for lm_id in mph.HandLandmark:
                        lm = hl.landmark[lm_id.value]
                        w.writerow([ts, key, rep, frame_no, tms, label, lm_id.name, lm.x, lm.y, lm.z])
                counts[key] = counts.get(key, 0) + 1
            if el < PREP_S:
                status = "GET READY  %.0f" % (PREP_S - el + 0.5)
                colour = (0, 200, 255)
            elif recording:
                status = "HOLD  (recording)"
                colour = (0, 0, 255)
            else:
                status = "rest"
                colour = (200, 200, 200)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
            cv2.putText(frame, "Round %d/%d   %s   (%s hand)" % (rep, args.reps, title, args.hand), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, hint[:95], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, status, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.imshow("Labelled grasp recording", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                quit_all = True
                break
            if k == ord("s"):
                skip = True
    f.close()
    cam.release()
    cv2.destroyAllWindows()
    print("\nCREATED:", out_path)
    print("frames recorded per grasp:", counts)
    print("Next: python tools/evaluate_grasp_labels.py")


if __name__ == "__main__":
    main()
