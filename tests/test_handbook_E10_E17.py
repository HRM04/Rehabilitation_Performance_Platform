"""Handbook E10 (duplicate threshold boundaries) and E17 (appropriateness decision table), 2 Nov 2026.

E10 builds boxes whose IoU, containment and mutual overlap sit just below, exactly at and just above the
thresholds in src/object_detection/duplicate_filter.py: suppress when IoU >= 0.90 or containment >= 0.95
(same class only), and treat a large box as a merged detection when it contains two distinct same-class
boxes whose mutual IoU is below 0.10.

E17 runs the real appropriateness branch. It is cut out of src/capture/camera_capture.py at test time, so it
tests the code that runs live, then checks the result against the task mapping and the logged reason text.
"""
import os
import re
import textwrap

import pytest

from src.object_detection.duplicate_filter import calculate_iou, filter_duplicate_objects
from src.grasp.grasp_family import FAMILY_OF, expected_families, family_assessment
from src.data.interaction_logger import _appropriateness_reason


def det(box, conf=0.9, cls="cup"):
    return {"class_name": cls, "bounding_box": box, "confidence": conf}


def kept_removed(objs, **kw):
    kept, removed = filter_duplicate_objects(objs, **kw)
    return len(kept), len(removed)


# ------------------------------------------------------------------ E10
# Two 19 x 5 boxes (area 95). Shifting one by dx gives IoU = (19 - dx) * 5 / (190 - (19 - dx) * 5).
def pair(dx):
    return [det((0, 0, 19, 5), 0.9), det((dx, 0, 19 + dx, 5), 0.8)]


def test_E10_iou_boundary_at_0_90():
    assert calculate_iou(pair(1)[0]["bounding_box"], pair(1)[1]["bounding_box"]) == pytest.approx(0.90)
    assert kept_removed(pair(1.1)) == (2, 0)      # IoU 0.8905, containment 0.942: both kept
    assert kept_removed(pair(1.0)) == (1, 1)      # IoU exactly 0.90: the lower-confidence box is suppressed
    assert kept_removed(pair(0.9)) == (1, 1)      # IoU 0.9095


def test_E10_iou_threshold_argument_is_honoured():
    a, b = det((0, 0, 10, 10), 0.9), det((2, 0, 12, 10), 0.8)     # IoU 0.667, containment 0.8
    assert kept_removed([a, b]) == (2, 0)
    assert kept_removed([a, b], iou_threshold=0.6) == (1, 1)
    assert kept_removed([a, b], iou_threshold=0.7) == (2, 0)


# A 100 x 100 box and a 20 x 10 box whose overlap with it covers a chosen share of the small box.
def nested(x0):
    return [det((0, 0, 100, 100), 0.6), det((x0, 0, x0 + 20, 10), 0.9)]


def test_E10_containment_boundary_at_0_95():
    assert kept_removed(nested(82)) == (2, 0)     # 90 percent inside: both kept
    assert kept_removed(nested(81)) == (1, 1)     # exactly 95 percent inside: suppressed
    assert kept_removed(nested(80)) == (1, 1)     # fully inside: suppressed


def test_E10_other_classes_are_never_suppressed():
    objs = [det((0, 0, 19, 5), 0.9, "cup"), det((0, 0, 19, 5), 0.8, "bottle")]
    assert kept_removed(objs) == (2, 0)


def test_E10_higher_confidence_box_is_the_one_kept():
    kept, removed = filter_duplicate_objects(pair(1.0))
    assert kept[0]["confidence"] == 0.9 and removed[0]["confidence"] == 0.8


# Large box over two smaller same-class boxes: flagged as a merged detection only when the two are distinct.
def merged(second_left):
    s2 = (second_left, 0, 20, 5)
    return [det((0, 0, 20, 5), 0.9), det((0, 0, 12, 5), 0.5), det(s2, 0.5)]


def test_E10_merged_detection_boundary_at_mutual_iou_0_10():
    # second box 10 to 20: mutual IoU with the first small box is exactly 0.10, so the large box is not flagged
    assert calculate_iou((0, 0, 12, 5), (10, 0, 20, 5)) == pytest.approx(0.10)
    kept, removed = filter_duplicate_objects(merged(10))
    assert [o["bounding_box"] for o in kept] == [(0, 0, 20, 5)]
    # second box 10.5 to 20: mutual IoU 0.075, so the large box is flagged and the two small boxes survive
    kept, removed = filter_duplicate_objects(merged(10.5))
    assert sorted(o["bounding_box"] for o in kept) == [(0, 0, 12, 5), (10.5, 0, 20, 5)]
    assert [o["bounding_box"] for o in removed] == [(0, 0, 20, 5)]
    # second box 9 to 20: mutual IoU 0.15, large box not flagged
    kept, removed = filter_duplicate_objects(merged(9))
    assert [o["bounding_box"] for o in kept] == [(0, 0, 20, 5)]


def test_E10_empty_and_single_inputs():
    assert filter_duplicate_objects([]) == ([], [])
    one = [det((0, 0, 10, 10))]
    assert filter_duplicate_objects(one) == (one, [])


# ------------------------------------------------------------------ E17
def live_branch(associated_object, hand_near_object, permitted_objects, grasp, candidates, expected, strict=False):
    """Run the appropriateness if/elif chain exactly as it stands in camera_capture.py."""
    text = open("src/capture/camera_capture.py", errors="ignore").read()
    assert "associated_object_class = (" in text and "# Grasp family (POWER or PRECISION)" in text, \
        "camera_capture.py layout changed: update the markers in live_branch"
    start = text.index("associated_object_class = (")
    start = text.rfind("\n", 0, start) + 1
    end = text.index("# Grasp family (POWER or PRECISION)")
    end = text.rfind("\n", 0, end) + 1
    block = textwrap.dedent(text[start:end])
    ns = {"associated_object": associated_object, "hand_near_object": hand_near_object,
          "permitted_objects": permitted_objects, "detected_grasp_type": grasp,
          "detected_grasp_types": candidates, "expected_grasps": expected, "_os": os}
    old = os.environ.get("AMBIGUOUS_APPROPRIATENESS")
    if strict:
        os.environ["AMBIGUOUS_APPROPRIATENESS"] = "strict"
    else:
        os.environ.pop("AMBIGUOUS_APPROPRIATENESS", None)
    try:
        exec(block, ns)
    finally:
        os.environ.pop("AMBIGUOUS_APPROPRIATENESS", None)
        if old is not None:
            os.environ["AMBIGUOUS_APPROPRIATENESS"] = old
    return ns["grasp_appropriateness"]


CUP = {"class_name": "cup"}
EXP = ["lateral_pinch", "tripod"]
PERM = ["cup"]


def test_E17_not_assessed_cases():
    assert live_branch(None, True, PERM, "tripod", [], EXP) == "NOT_ASSESSED"             # no object
    assert live_branch(CUP, False, PERM, "tripod", [], EXP) == "NOT_ASSESSED"             # hand not near
    assert live_branch({"class_name": "book"}, True, PERM, "tripod", [], EXP) == "NOT_ASSESSED"   # not the task object


def test_E17_single_label_cases():
    assert live_branch(CUP, True, PERM, "tripod", [], EXP) == "APPROPRIATE"
    assert live_branch(CUP, True, PERM, "lateral_pinch", [], EXP) == "APPROPRIATE"
    assert live_branch(CUP, True, PERM, "cylindrical", [], EXP) == "INAPPROPRIATE"
    assert live_branch(CUP, True, PERM, "UNKNOWN", [], EXP) == "UNCERTAIN"


def test_E17_ambiguous_cases():
    assert live_branch(CUP, True, PERM, "AMBIGUOUS", ["tripod", "lateral_pinch"], EXP) == "APPROPRIATE"      # all expected
    assert live_branch(CUP, True, PERM, "AMBIGUOUS", ["cylindrical", "handle"], EXP) == "INAPPROPRIATE"      # none expected
    assert live_branch(CUP, True, PERM, "AMBIGUOUS", ["tripod", "cylindrical"], EXP) == "UNCERTAIN"          # mixed
    assert live_branch(CUP, True, PERM, "AMBIGUOUS", ["tripod", "lateral_pinch"], EXP, strict=True) == "UNCERTAIN"


def test_E17_matches_the_real_task_mapping():
    from src.platform_config.session_config import SessionConfig
    for obj in ("bottle", "cup", "spoon"):
        cfg = SessionConfig(object_id=obj, trajectory="horizontal", patient_id="e17", session_purpose="testing",
                            volunteer_session=1, target_repetitions=1, trajectory_enabled=True,
                            orientation_enabled=False, segmentation_enabled=False)
        task = cfg.get_task_details()
        ek = [k for k in task if "expected" in k]
        pk = [k for k in task if "permitted" in k]
        assert ek and pk, "task details have no expected/permitted keys: %s" % sorted(task)
        expected, permitted = list(task[ek[0]]), list(task[pk[0]])
        assert expected, "%s has no expected grasps" % obj
        outside = [g for g in ("cylindrical", "spherical", "hook", "handle", "tip_pinch", "tripod", "lateral_pinch", "pincer")
                   if g not in expected]
        assert live_branch({"class_name": obj}, True, permitted, expected[0], [], expected) == "APPROPRIATE"
        if outside:
            assert live_branch({"class_name": obj}, True, permitted, outside[0], [], expected) == "INAPPROPRIATE"


def test_E17_family_level_table():
    power = next(g for g, f in FAMILY_OF.items() if f == "POWER")
    precision = next(g for g, f in FAMILY_OF.items() if f == "PRECISION")
    assert family_assessment("POWER", False, [power]) == "APPROPRIATE"
    assert family_assessment("PRECISION", False, [power]) == "INAPPROPRIATE"
    assert family_assessment("PRECISION", False, [power, precision]) == "APPROPRIATE"
    assert family_assessment(None, False, [power]) == "NOT_ASSESSED"
    assert family_assessment("POWER", True, [power]) == "NOT_ASSESSED"
    assert family_assessment("POWER", False, []) == "NOT_ASSESSED"
    assert expected_families(["not_a_grasp"]) == set()


def test_E17_logged_reason_text():
    f = _appropriateness_reason
    assert f({"grasp_appropriateness": "APPROPRIATE"}, "ASSOCIATED", True) == "GRASP_IN_EXPECTED_SET"
    assert f({"grasp_appropriateness": "INAPPROPRIATE"}, "ASSOCIATED", True) == "GRASP_NOT_IN_EXPECTED_SET"
    assert f({"grasp_appropriateness": "UNCERTAIN", "detected_grasp_type": "AMBIGUOUS"}, "ASSOCIATED", True) == "GRASP_AMBIGUOUS"
    assert f({"grasp_appropriateness": "NOT_ASSESSED"}, "AMBIGUOUS", True) == "NO_OBJECT_ASSOCIATED:AMBIGUOUS"
    assert f({"grasp_appropriateness": "NOT_ASSESSED"}, None, False) == "NO_OBJECT_ASSOCIATED"
    assert f({"grasp_appropriateness": "NOT_ASSESSED"}, "ASSOCIATED", True) == "OBJECT_NOT_NEAR_HAND_OR_NOT_TASK_OBJECT"
    assert f({}, None, False) is None
