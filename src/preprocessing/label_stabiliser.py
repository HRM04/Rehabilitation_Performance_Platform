"""Keeps one hand's Left/Right label steady from frame to frame.

MediaPipe sometimes flips the label of a single hand for a few frames, usually at low confidence.
A label is changed only when ALL of these hold: its confidence is below CONF_MIN, the opposite
label was tracked within DIST (normalised image units) of this hand in the last MAX_AGE frames,
this label was not tracked there, and the opposite label is not already used by another hand in the
same frame. Switch off with LABEL_STABILISE=off.
"""
import atexit
import os
from src.preprocessing.handedness import final_label

CONF_MIN = 0.80
DIST = 0.15
MAX_AGE = 6


def enabled():
    return os.environ.get("LABEL_STABILISE", "on").lower() != "off"


class LabelStabiliser:
    def __init__(self):
        self._res = None
        self._labels = []
        self._seen = {}
        self._frame = 0
        self.flips = 0
        self.checked = 0

    def _hands(self, results):
        out = []
        lms = results.multi_hand_landmarks or []
        hds = results.multi_handedness or []
        for i in range(min(len(lms), len(hds))):
            c = hds[i].classification[0]
            pts = lms[i].landmark
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            out.append((final_label(c.label), float(c.score), cx, cy))
        return out

    def _near(self, label, cx, cy):
        s = self._seen.get(label)
        if s is None or self._frame - s[2] > MAX_AGE:
            return False
        return ((s[0] - cx) ** 2 + (s[1] - cy) ** 2) ** 0.5 <= DIST

    def update(self, results):
        self._frame += 1
        hands = self._hands(results)
        labels = [h[0] for h in hands]
        for i, (lab, conf, cx, cy) in enumerate(hands):
            if conf >= CONF_MIN:
                continue
            other = "Right" if lab == "Left" else "Left"
            self.checked += 1
            if self._near(other, cx, cy) and not self._near(lab, cx, cy) and other not in labels:
                labels[i] = other
                self.flips += 1
        for (lab, conf, cx, cy), new_lab in zip(hands, labels):
            self._seen[new_lab] = (cx, cy, self._frame)
        self._labels = labels
        self._res = results

    def label(self, results, index):
        if self._res is not results:
            self.update(results)
        return self._labels[index]


_STAB = LabelStabiliser()


def stable_hand_label(results, index):
    try:
        if enabled():
            return _STAB.label(results, index)
    except Exception:
        pass
    return final_label(results.multi_handedness[index].classification[0].label)


def describe():
    return {"enabled": enabled(), "conf_min": CONF_MIN, "dist": DIST, "max_age_frames": MAX_AGE}


def _report():
    if _STAB.checked or _STAB.flips:
        print("[LABEL STABILISER] corrected %d of %d low-confidence labels" % (_STAB.flips, _STAB.checked))


atexit.register(_report)
