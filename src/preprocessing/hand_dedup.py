"""One detection per hand label per frame.

MediaPipe sometimes returns two detections under the same label (after the
mirror correction). Logs showed this in about 7% of hand frames: roughly two
thirds were the same hand found twice (centres under 100 px apart), about a
fifth were two real hands where one was mislabelled.

Rule, applied before anything else uses the result:
  - two detections under one label and close together: keep the stronger one
  - two detections under one label and far apart: they are two hands, so the
    weaker one takes the other label
Switch off with HAND_DEDUP=off.
"""
import os
from src.preprocessing.handedness import final_label, raw_label_for

CLOSE_PX = 100.0
FRAME_W, FRAME_H = 1280.0, 720.0
STATS = {"frames": 0, "dropped_same_hand": 0, "relabelled_second_hand": 0}


def enabled():
    return os.environ.get("HAND_DEDUP", "on").lower() != "off"


def describe():
    return {"enabled": enabled(), "close_px": CLOSE_PX}


def _final_label(raw_label):
    # same correction the pipeline applies after the mirror flip
    return final_label(raw_label)


def _centre(hand_landmarks):
    pts = hand_landmarks.landmark
    return (sum(p.x for p in pts) / len(pts) * FRAME_W, sum(p.y for p in pts) / len(pts) * FRAME_H)


def dedupe_hands(results):
    """Edit results.multi_hand_landmarks / multi_handedness in place. Returns results."""
    if not enabled() or not getattr(results, "multi_hand_landmarks", None):
        return results
    lms = results.multi_hand_landmarks
    hds = results.multi_handedness
    if len(lms) < 2 or len(lms) != len(hds):
        return results
    STATS["frames"] += 1
    keep = list(range(len(lms)))
    by_label = {}
    for i in keep:
        by_label.setdefault(_final_label(hds[i].classification[0].label), []).append(i)
    for label, idx in by_label.items():
        if len(idx) < 2:
            continue
        idx = sorted(idx, key=lambda i: -hds[i].classification[0].score)
        strong, weak = idx[0], idx[1]
        (ax, ay), (bx, by) = _centre(lms[strong]), _centre(lms[weak])
        if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < CLOSE_PX:
            keep.remove(weak)
            STATS["dropped_same_hand"] += 1
        else:
            other_final = "Left" if label == "Right" else "Right"
            if other_final not in by_label:
                # raw MediaPipe label is the opposite of the final label
                hds[weak].classification[0].label = raw_label_for(other_final)
                STATS["relabelled_second_hand"] += 1
            else:
                keep.remove(weak)
                STATS["dropped_same_hand"] += 1
    if len(keep) != len(lms):
        lms[:] = [lms[i] for i in keep]
        hds[:] = [hds[i] for i in keep]
    return results
