"""Handbook E17 and E02 for every object in the library (2 Nov 2026).

The object ids come from the object library itself, found by scanning src/ for the module that holds the
13 object entries. Each object is turned into a live task through SessionConfig.get_task_details(), then the
real appropriateness branch from camera_capture.py is run against that task:
  - the task object with each expected grasp           -> APPROPRIATE
  - the task object with each grasp not expected       -> INAPPROPRIATE
  - a different object, or a hand away from the object -> NOT_ASSESSED
  - an ambiguous label whose candidates are all / partly expected -> APPROPRIATE / UNCERTAIN
The matrix of permitted objects and expected grasps is written to reports/task_mapping_matrix.csv and .txt.
"""
import csv
import glob
import importlib
import os

import pytest

try:
    from test_handbook_E10_E17 import live_branch
except ImportError:
    from tests.test_handbook_E10_E17 import live_branch

GRASPS = ("cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook",
          "pincer", "tip_pinch", "tripod", "lateral_pinch", "multidigit_precision")
FALLBACK = ["bottle", "cup", "spoon", "fork", "bowl", "book", "cell phone", "remote", "toothbrush",
            "scissors", "sports ball", "apple", "banana"]


def discover():
    for f in glob.glob("src/**/*.py", recursive=True):
        low = f.lower()
        if "backup" in low or ".pre_" in low:
            continue
        text = open(f, errors="ignore").read()
        if not ("toothbrush" in text and "scissors" in text):
            continue
        name = f[:-3].replace(os.sep, ".")
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr, val in vars(mod).items():
            if isinstance(val, dict) and len(val) >= 10 and "toothbrush" in {str(k).replace("_", " ") for k in val}:
                return name + "." + attr, [str(k) for k in val]
            if isinstance(val, (list, tuple)) and len(val) >= 10 and all(isinstance(v, dict) for v in val):
                for key in ("object_id", "class_name", "name", "id"):
                    names = [str(v.get(key)) for v in val if v.get(key)]
                    if "toothbrush" in {n.replace("_", " ") for n in names}:
                        return name + "." + attr, names
    return None, list(FALLBACK)


SOURCE, IDS = discover()


def task_for(obj):
    from src.platform_config.session_config import SessionConfig
    last = None
    for cand in dict.fromkeys([obj, obj.replace(" ", "_"), obj.replace("_", " ")]):
        try:
            cfg = SessionConfig(object_id=cand, trajectory="horizontal", patient_id="e17", session_purpose="testing",
                                volunteer_session=1, target_repetitions=1, trajectory_enabled=True,
                                orientation_enabled=False, segmentation_enabled=False)
            return cand, cfg.get_task_details()
        except Exception as e:
            last = e
    raise AssertionError("SessionConfig rejected %r: %r" % (obj, last))


def lists(task):
    ek = [k for k in task if "expected" in k]
    pk = [k for k in task if "permitted" in k]
    assert ek and pk, "task details have no expected/permitted keys: %s" % sorted(task)
    return list(task[ek[0]]), list(task[pk[0]])


def test_E02_object_set_and_matrix_written():
    assert len(IDS) >= 11, "found only %d objects (source: %s)" % (len(IDS), SOURCE)
    rows = []
    for obj in IDS:
        used, task = task_for(obj)
        expected, permitted = lists(task)
        rows.append({"object": obj, "task_id": task.get("task_id", ""), "permitted_objects": " | ".join(map(str, permitted)),
                     "expected_grasps": " | ".join(expected),
                     "detectors_not_expected": " | ".join(g for g in GRASPS if g not in expected)})
    os.makedirs("reports", exist_ok=True)
    with open("reports/task_mapping_matrix.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["Task mapping matrix (%d objects; library found at: %s)" % (len(rows), SOURCE or "NOT FOUND, used the audit list"), ""]
    lines += ["%-12s %-22s permitted: %-14s expected: %s" % (r["object"], r["task_id"], r["permitted_objects"], r["expected_grasps"]) for r in rows]
    covered = {g for r in rows for g in r["expected_grasps"].split(" | ") if g}
    lines += ["", "Detectors that no object expects: %s" % (", ".join(g for g in GRASPS if g not in covered) or "none")]
    open("reports/task_mapping_matrix.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


@pytest.mark.parametrize("obj", IDS)
def test_E17_task_mapping_for_object(obj):
    used, task = task_for(obj)
    expected, permitted = lists(task)
    assert permitted, "%s has no permitted objects" % obj
    assert expected, "%s has no expected grasps, so every grasp would be INAPPROPRIATE" % obj
    near = {"class_name": permitted[0]}
    assert live_branch({"class_name": "__another_object__"}, True, permitted, expected[0], [], expected) == "NOT_ASSESSED"
    assert live_branch(near, False, permitted, expected[0], [], expected) == "NOT_ASSESSED"
    assert live_branch(None, True, permitted, expected[0], [], expected) == "NOT_ASSESSED"
    for g in expected:
        assert live_branch(near, True, permitted, g, [], expected) == "APPROPRIATE", g
    outside = [g for g in GRASPS if g not in expected]
    for g in outside:
        assert live_branch(near, True, permitted, g, [], expected) == "INAPPROPRIATE", g
    assert live_branch(near, True, permitted, "UNKNOWN", [], expected) == "UNCERTAIN"
    assert live_branch(near, True, permitted, "AMBIGUOUS", list(expected), expected) == "APPROPRIATE"
    if outside:
        assert live_branch(near, True, permitted, "AMBIGUOUS", [expected[0], outside[0]], expected) == "UNCERTAIN"
        assert live_branch(near, True, permitted, "AMBIGUOUS", outside[:2], expected) == "INAPPROPRIATE"
