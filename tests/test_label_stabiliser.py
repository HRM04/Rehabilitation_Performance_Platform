from types import SimpleNamespace as NS
from src.preprocessing.label_stabiliser import LabelStabiliser


def _hand(label, conf, x, y):
    return NS(classification=[NS(label=label, score=conf)]), NS(landmark=[NS(x=x, y=y) for _ in range(21)])


def _res(*hands):
    hs = [_hand(*h) for h in hands]
    return NS(multi_handedness=[h[0] for h in hs], multi_hand_landmarks=[h[1] for h in hs])


def _run(st, *hands):
    r = _res(*hands)
    return [st.label(r, i) for i in range(len(hands))]


def _warm(st, n=5):
    for _ in range(n):
        assert _run(st, ("Right", 0.98, 0.7, 0.6)) == ["Right"]


def test_low_confidence_flip_next_to_tracked_hand_is_corrected():
    st = LabelStabiliser(); _warm(st)
    assert _run(st, ("Left", 0.65, 0.72, 0.6)) == ["Right"]
    assert st.flips == 1


def test_high_confidence_label_is_kept():
    st = LabelStabiliser(); _warm(st)
    assert _run(st, ("Left", 0.95, 0.72, 0.6)) == ["Left"]


def test_low_confidence_far_from_tracked_hand_is_kept():
    st = LabelStabiliser(); _warm(st)
    assert _run(st, ("Left", 0.65, 0.2, 0.6)) == ["Left"]


def test_no_flip_when_other_label_already_in_frame():
    st = LabelStabiliser(); _warm(st)
    assert _run(st, ("Left", 0.65, 0.72, 0.6), ("Right", 0.97, 0.3, 0.6)) == ["Left", "Right"]


def test_switch_off(monkeypatch):
    monkeypatch.setenv("LABEL_STABILISE", "off")
    from src.preprocessing import label_stabiliser as m
    assert m.stable_hand_label(_res(("Left", 0.65, 0.72, 0.6)), 0) == "Left"
