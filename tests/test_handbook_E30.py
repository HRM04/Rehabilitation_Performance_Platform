"""Handbook E30: regression replay. The replay tool must detect change and stay quiet when nothing changed."""
import glob, importlib.util, os, sys
import pytest

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("regression_replay", "tools/regression_replay.py")
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)


def test_E30_identical_documents_have_no_differences():
    d = {"a": 1, "b": [1.0, {"c": "x"}], "n": float("nan")}
    assert rr.diff(d, {"a": 1, "b": [1.0, {"c": "x"}], "n": float("nan")}) == []


def test_E30_changed_value_missing_key_and_length_are_reported():
    old = {"a": 1.0, "b": [1, 2], "c": "x"}
    new = {"a": 1.1, "b": [1, 2, 3], "d": 1}
    paths = {p for p, _, _ in rr.diff(old, new)}
    assert {"/a", "/b[length]", "/c", "/d"} <= paths


def test_E30_tiny_float_noise_is_ignored_but_real_change_is_not():
    assert rr.diff({"a": 1.0}, {"a": 1.0 + 1e-12}) == []
    assert rr.diff({"a": 1.0}, {"a": 1.001}) != []


def test_E30_newest_recorded_session_replays_to_its_stored_export():
    sid = rr.newest_session()
    stored = "data/processed/unified_sessions/%s_unified_assessment_v02.json" % sid
    if not os.path.exists(stored):
        pytest.skip("no stored export for %s" % sid)
    import json
    old = json.load(open(stored))
    new = rr.rebuild(sid)
    d = rr.diff({k: v for k, v in old.items() if k != "posture_calibration"},
                {k: v for k, v in new.items() if k != "posture_calibration"})
    assert d == [], "replay of %s differs from its export: %s" % (sid, d[:10])
