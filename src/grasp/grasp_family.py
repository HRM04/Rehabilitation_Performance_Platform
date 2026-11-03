"""Grasp family (POWER or PRECISION) from a small rule tree fitted on labelled recordings.

Fine grasp names could not be separated reliably from one RGB camera (see the labelled-grasp
evaluation). The two broad families can be, so task appropriateness is also scored at family level.
The rules live in family_model.json. If that file says enabled=false, or is missing, no family
is reported and family appropriateness stays NOT_ASSESSED.
"""
import json, os
from src.preprocessing.grasp_voting import GraspVoter

POWER = {"cylindrical", "spherical", "large_palmar", "whole_hand_power", "handle", "hook"}
PRECISION = {"pincer", "tip_pinch", "tripod", "lateral_pinch", "multidigit_precision"}
FAMILY_OF = {**{n: "POWER" for n in POWER}, **{n: "PRECISION" for n in PRECISION}}
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "family_model.json")
MODEL = None
_VOTER = GraspVoter()
CONTACT_MAX = 2  # at or below this many fingertips on the object: PRECISION


def contact_rule_on():
    return os.environ.get('FAMILY_CONTACT_RULE', 'on').lower() != 'off'


def set_model(model):
    global MODEL
    MODEL = model
    _VOTER.reset()


def load_model(path=_PATH):
    try:
        with open(path) as f:
            set_model(json.load(f))
    except Exception:
        set_model(None)


def describe():
    if not MODEL:
        return {"enabled": False, "reason": "no model file"}
    return {"enabled": bool(MODEL.get("enabled")), "depth": MODEL.get("depth"),
            "held_out_family_accuracy": MODEL.get("held_out_family_accuracy"),
            "contact_rule": contact_rule_on()}


def family_features(thumb_index, thumb_middle, aperture, side_dist, spread, angles, tfd):
    f = {"thumb_index": thumb_index, "thumb_middle": thumb_middle, "aperture": aperture,
         "side_dist": side_dist, "spread": spread}
    for k, v in (angles or {}).items():
        f["angle_" + str(k)] = v
    if isinstance(tfd, dict):
        for k, v in tfd.items():
            f["thumb_to_" + str(k)] = v
    return {k: float(v) for k, v in f.items() if isinstance(v, (int, float))}


def _walk(tree, f, feats, med):
    while "leaf" not in tree:
        name = tree["feat"]
        v = f.get(name)
        if v is None:
            v = med[feats.index(name)]
        tree = tree["le"] if v <= tree["thr"] else tree["gt"]
    return tree["leaf"]


def classify_family(hand, t_ms, features, fingertips=None):
    """Voted family for one hand, or None when no usable model."""
    if not MODEL or not MODEL.get("enabled"):
        return None
    raw = _walk(MODEL["tree"], features, MODEL["features"], MODEL["medians"])
    if contact_rule_on() and fingertips is not None and 1 <= fingertips <= CONTACT_MAX:
        raw = 'PRECISION'
    label, _ = _VOTER.vote(hand, t_ms, raw, [raw])
    return label


def expected_families(expected_grasps):
    return {FAMILY_OF[g] for g in (expected_grasps or []) if g in FAMILY_OF}


def family_assessment(family, not_assessable, expected_grasps):
    fams = expected_families(expected_grasps)
    if not_assessable or family is None or not fams:
        return "NOT_ASSESSED"
    return "APPROPRIATE" if family in fams else "INAPPROPRIATE"


load_model()
