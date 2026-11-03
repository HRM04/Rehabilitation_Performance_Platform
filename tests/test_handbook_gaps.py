"""Handbook gap tests for E09, E12, E15 and E16 (written 2 Nov 2026).

E15/E16 work at feature level: each detector receives finger angles and normalised distances,
the inputs the real pipeline computes from landmarks. A positive input is found by seeded random
search, then each input is swept to measure where the detector flips from True to False.
"""
import os
import random
from unittest import mock

import pytest

from src.grasp import grasp_classifier as gc
from src.object_detection import hand_object_interaction as hoi
from src.object_detection.duplicate_filter import calculate_iou

# ---------------------------------------------------------------- E09 geometry
def test_E09_iou_known_values():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert calculate_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert calculate_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(4 / 100)
    assert calculate_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_E09_distance_to_box():
    box = {"bounding_box": (100, 100, 200, 300)}
    f = hoi.distance_to_object_box
    assert f((150, 200), box) == 0
    assert f((100, 100), box) == 0
    assert f((90, 200), box) == pytest.approx(10)
    assert f((203, 304), box) == pytest.approx(5.0)


# ---------------------------------------------------------------- E12 association
NAMES = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP",
         "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP",
         "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
         "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"]
TIPS = ("THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP")


def hand(centre, tips_at=None):
    tips_at = tips_at or {}
    out = {}
    for n in NAMES:
        x, y = tips_at.get(n, centre)
        out[n] = {"x_px": x, "y_px": y, "x_scaled": x / 100.0, "y_scaled": y / 100.0, "x_norm": 0.5, "y_norm": 0.5, "z_rel": 0.0}
    return out


def obj(object_id, box, cls="cup"):
    x1, y1, x2, y2 = box
    return {"object_id": object_id, "track_id": object_id, "class_name": cls, "confidence": 0.9,
            "bounding_box": box, "center": ((x1 + x2) // 2, (y1 + y2) // 2)}


def test_E12_state_table():
    f = hoi.determine_interaction_state
    assert f(False, 0, False) == "NONE"
    assert f(True, 0, False) == "NEAR"
    assert f(True, 1, True) == "CONTACT"
    assert f(True, 2, False) == "CONTACT"
    assert f(True, 2, True) == "GRASPING"
    assert f(True, 5, True) == "GRASPING"


def test_E12_fingertip_margin_boundary():
    o = obj(1, (100, 100, 200, 300))
    inside = hand((150, 200), {"THUMB_TIP": (220, 200)})           # x2 + margin(20) exactly
    outside = hand((150, 200), {"THUMB_TIP": (221, 200)})
    far = (900, 900)
    h_in = {n: dict(v) for n, v in inside.items()}; h_out = {n: dict(v) for n, v in outside.items()}
    for n in NAMES:
        if n != "THUMB_TIP":
            h_in[n]["x_px"], h_in[n]["y_px"] = far
            h_out[n]["x_px"], h_out[n]["y_px"] = far
    assert hoi.count_fingertips_near_object(h_in, o, margin=20) == 1
    assert hoi.count_fingertips_near_object(h_out, o, margin=20) == 0


def test_E12_association_cases():
    f = hoi.select_hand_object_association
    a = obj(1, (100, 100, 200, 300))
    inside = hand((150, 200), {t: (150 + 5 * i, 200) for i, t in enumerate(TIPS)})
    o, status, diag = f((150, 200), inside, [a])
    assert status == "ASSOCIATED" and o["object_id"] == 1                       # near, fingertips on object
    o, status, _ = f((900, 600), hand((900, 600)), [a])
    assert status == "NONE" and o is None                                       # hand far away
    o, status, _ = f((150, 200), inside, [dict(a, object_id=None)])
    assert status == "NONE" and o is None                                       # no identity, not associated
    b = obj(2, (300, 100, 400, 300))
    o, status, _ = f((250, 200), hand((250, 200)), [a, b], max_distance=80)
    assert status == "AMBIGUOUS" and o is None                                  # equidistant, no fingertip evidence
    clash = [obj(7, (100, 100, 200, 300)), obj(7, (110, 110, 210, 310))]
    o, status, _ = f((150, 200), inside, clash)
    assert status != "ASSOCIATED"                                               # one id, two different boxes


# ---------------------------------------------------------------- E15/E16 detectors
ANG = ("index", "middle", "ring", "little")
ORDER = {
    "detect_cylindrical_grasp": ["angles", "ti", "tm"],
    "detect_spherical_grasp": ["angles", "ti", "tm", "spread"],
    "detect_large_aperture_palmar_grasp": ["angles", "spread", "ap", "ti", "tm"],
    "detect_whole_hand_power_grip": ["angles", "ti", "tm", "ap"],
    "detect_handle_grasp": ["angles", "ti", "tm", "ap", "spread"],
    "detect_hook_grasp": ["angles", "ti", "tm", "ap"],
    "detect_pincer_grasp": ["angles", "ti", "tm"],
    "detect_tip_pinch": ["angles", "ti", "tm"],
    "detect_tripod_grasp": ["angles", "ti", "tm"],
    "detect_lateral_pinch": ["angles", "ti", "side"],
    "detect_multidigit_precision_grasp": ["angles", "tfd"],
}
DETECTORS = list(ORDER)
RANGES = {"ti": (0.02, 2.6), "tm": (0.02, 2.6), "ap": (0.2, 2.6), "spread": (0.0, 0.7), "side": (0.02, 1.6)}
DOCUMENTED = {  # numeric thresholds listed in the 2 Nov source audit
    "detect_cylindrical_grasp": [1.5, 150, 120], "detect_spherical_grasp": [0.2, 1.7, 155],
    "detect_large_aperture_palmar_grasp": [0.18, 1.0, 2.0, 160], "detect_whole_hand_power_grip": [1.5, 1.6, 145, 110],
    "detect_handle_grasp": [0.35, 1.4, 1.7, 150], "detect_hook_grasp": [1.8, 0.8, 140, 110],
    "detect_pincer_grasp": [0.55, 0.65, 70, 120, 110], "detect_tip_pinch": [0.4, 120, 0.65, 110],
    "detect_tripod_grasp": [0.65, 165, 110], "detect_lateral_pinch": [0.75, 0.3, 115],
    "detect_multidigit_precision_grasp": [0.75, 165, 100],
}


def sample(rng):
    return {"angles": {k: rng.uniform(60, 180) for k in ANG}, "ti": rng.uniform(0.05, 2.5), "tm": rng.uniform(0.05, 2.5),
            "ap": rng.uniform(0.3, 2.5), "spread": rng.uniform(0.0, 0.6), "side": rng.uniform(0.05, 1.5),
            "tfd": {k: rng.uniform(0.05, 2.0) for k in ANG}}


def value(key, s, form):
    if key == "spread":
        v = s["spread"]
        return {"index_middle": v, "middle_ring": v, "ring_little": v} if form == "dict" else v
    if key == "angles":
        return dict(s["angles"])
    if key == "tfd":
        return dict(s["tfd"])
    return s[key]


def run(name, s):
    if name == "detect_cylindrical_grasp":
        with mock.patch.object(gc, "calculate_finger_angles", return_value=dict(s["angles"])), \
             mock.patch.object(gc, "calculate_thumb_opposition", return_value={"thumb_index": s["ti"], "thumb_middle": s["tm"]}):
            return gc.detect_cylindrical_grasp({})
    fn = getattr(gc, name)
    last = None
    for form in ("dict", "float"):
        try:
            return fn(*[value(k, s, form) for k in ORDER[name]])
        except (TypeError, AttributeError) as e:
            last = e
            if "spread" not in ORDER[name]:
                raise
    raise last


_FOUND = {}
_REPORT = "reports/detector_thresholds_measured.txt"
if os.path.exists(_REPORT):
    os.remove(_REPORT)


def best_positive(name):
    if name not in _FOUND:
        rng = random.Random(20261102 + DETECTORS.index(name))
        best = None
        for _ in range(60000):
            s = sample(rng)
            ok, conf = run(name, s)
            if ok and (best is None or conf > best[1]):
                best = (s, conf)
        _FOUND[name] = best
    return _FOUND[name]


def flips(name, base):
    out = []
    dims = [("angle:" + k, 40.0, 180.0) for k in ANG] + [("angle:all", 40.0, 180.0)]
    for k in ORDER[name]:
        if k in RANGES:
            dims.append((k,) + RANGES[k])
        if k == "tfd":
            dims += [("tfd:" + f, 0.02, 2.2) for f in ANG] + [("tfd:all", 0.02, 2.2)]
    for dim, lo, hi in dims:
        for direction, end in (("up", hi), ("down", lo)):
            if dim == "angle:all": cur = sum(base["angles"].values()) / 4
            elif dim == "tfd:all": cur = sum(base["tfd"].values()) / 4
            else: cur = base["angles"][dim[6:]] if dim.startswith("angle:") else (base["tfd"][dim[4:]] if dim.startswith("tfd:") else base[dim])
            for step in range(1, 401):
                v = cur + (end - cur) * step / 400
                s = {**base, "angles": dict(base["angles"]), "tfd": dict(base["tfd"])}
                if dim == "angle:all": s["angles"] = {k: v for k in ANG}
                elif dim == "tfd:all": s["tfd"] = {k: v for k in ANG}
                elif dim.startswith("angle:"): s["angles"][dim[6:]] = v
                elif dim.startswith("tfd:"): s["tfd"][dim[4:]] = v
                else: s[dim] = v
                if not run(name, s)[0]:
                    out.append((dim, direction, round(v, 3)))
                    break
    return out


@pytest.mark.parametrize("name", DETECTORS)
def test_E15_positive_case_exists(name):
    found = best_positive(name)
    assert found is not None, f"{name}: no positive input in 60000 seeded samples"
    assert found[1] > 0


def test_E15_open_hand_is_negative_for_all_detectors():
    s = {"angles": {k: 178.0 for k in ANG}, "ti": 1.8, "tm": 1.8, "ap": 2.0, "spread": 0.5, "side": 1.8,
         "tfd": {k: 1.8 for k in ANG}}
    positives = [n for n in DETECTORS if run(n, s)[0]]
    assert positives == [], f"open hand matched {positives}"


@pytest.mark.parametrize("name", DETECTORS)
def test_E16_boundaries_exist_and_are_reported(name):
    base = best_positive(name)
    assert base is not None
    f = flips(name, base[0])
    assert f, f"{name}: no input flips the positive case to negative"
    lines = [f"{name}  base confidence {base[1]:.1f}"]
    for dim, direction, v in f:
        rule = f"{dim} must be {'<' if direction == 'up' else '>'} {v}"
        near = [d for d in DOCUMENTED[name] if abs(d - v) <= (1.5 if dim.startswith("angle") else 0.03)]
        lines.append("    " + rule + (f"   (documented {near[0]})" if near else "   (score gate or combined rule)"))
    os.makedirs("reports", exist_ok=True)
    with open(_REPORT, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# exact boundary checks written from the source of three detectors
def yes(result):
    return bool(result[0])


def test_E16_tripod_thumb_index_boundary():
    a = {"index": 140.0, "middle": 140.0, "ring": 170.0, "little": 170.0}
    assert yes(gc.detect_tripod_grasp(a, 0.649, 0.30))
    assert not yes(gc.detect_tripod_grasp(a, 0.651, 0.30))


def test_E16_lateral_pinch_boundaries():
    a = {"index": 170.0, "middle": 170.0, "ring": 170.0, "little": 170.0}
    assert yes(gc.detect_lateral_pinch(a, 0.50, 0.74))
    assert not yes(gc.detect_lateral_pinch(a, 0.50, 0.76))
    assert not yes(gc.detect_lateral_pinch(a, 0.29, 0.50))
    assert not yes(gc.detect_lateral_pinch({**a, "index": 114.0}, 0.50, 0.50))


def test_E16_cylindrical_boundaries():
    def go(angles, ti, tm):
        with mock.patch.object(gc, "calculate_finger_angles", return_value=angles), \
             mock.patch.object(gc, "calculate_thumb_opposition", return_value={"thumb_index": ti, "thumb_middle": tm}):
            return gc.detect_cylindrical_grasp({})
    fist = {"index": 100.0, "middle": 100.0, "ring": 100.0, "little": 100.0}
    assert yes(go(fist, 1.0, 2.0))
    assert yes(go(fist, 1.49, 2.0)) and not yes(go(fist, 1.51, 2.0))
    three = {"index": 149.0, "middle": 149.0, "ring": 149.0, "little": 151.0}
    two = {"index": 149.0, "middle": 149.0, "ring": 151.0, "little": 151.0}
    assert yes(go(three, 1.0, 2.0)) and not yes(go(two, 1.0, 2.0))
    assert not yes(go({**fist, "little": None}, 1.0, 2.0))
