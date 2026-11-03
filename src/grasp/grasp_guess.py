"""Best guess of the specific grasp, chosen inside the predicted family.

The family (POWER or PRECISION) comes from grasp_family. Inside it, the grasp whose typical
hand shape (from the labelled recordings) is closest to the current hand is the guess.
Held-out check: 84% right at family level, about 32% exact, one person's data. It is shown as
a guess and is not used for the appropriateness verdict.
"""
import json, os
from src.grasp.grasp_family import FAMILY_OF
from src.preprocessing.grasp_voting import GraspVoter

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_prototypes.json")
PROTO = None
_INFO = {}
_VOTER = GraspVoter()


def set_prototypes(data):
    global PROTO, _INFO
    data = data or {}
    PROTO = data.get("proto") if data.get("enabled") else None
    _INFO = {k: data.get(k) for k in ("sequences", "rounds")}
    _VOTER.reset()


def load(path=_PATH):
    try:
        with open(path) as f:
            set_prototypes(json.load(f))
    except Exception:
        set_prototypes(None)


def describe():
    return {"enabled": bool(PROTO), **_INFO}


def _dist(proto, f):
    ds = []
    for k, (m, s) in proto.items():
        v = f.get(k)
        if isinstance(v, (int, float)):
            ds.append(min(((v - m) / s) ** 2, 25.0))
    return sum(ds) / len(ds) if ds else 99.0


def best_guess(hand, t_ms, features, family):
    if not PROTO or not features or family not in ("POWER", "PRECISION"):
        return None
    cands = [g for g in PROTO if FAMILY_OF.get(g) == family]
    if not cands:
        return None
    raw = min(cands, key=lambda g: _dist(PROTO[g], features))
    label, _ = _VOTER.vote(hand, t_ms, raw, [raw])
    return label if label in PROTO else None


load()
