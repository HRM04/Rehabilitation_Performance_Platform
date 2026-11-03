"""Handbook E32: do the numbers on the dashboards match the exported JSON? (2 Nov 2026)

Each dashboard is run headless with Streamlit's AppTest. The numbers it shows are compared with the files they
come from: the unified assessment JSON for the clinician session page and the patient page, and the trajectory
results JSON for the longitudinal page. Every quality warning stored for an attempt must be shown, and the
"no warnings" message must appear only when none are stored.
"""
import glob
import json
import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASH = Path.cwd() / "src" / "dashboard"
UNIFIED = sorted(glob.glob("data/processed/unified_sessions/*_unified_assessment_v02.json"))
TRAJ = sorted(glob.glob("data/trajectory_results/*_trajectory_results.json"))


def run(page):
    logging.disable(logging.CRITICAL)
    at = AppTest.from_file(str(DASH / page), default_timeout=120).run()
    assert not at.exception, "%s raised: %s" % (page, [e.value for e in at.exception])
    return at


def shown(at):
    return {m.label: m.value for m in at.metric}


def newest_export():
    assert UNIFIED, "no unified assessment files found in data/processed/unified_sessions"
    return json.load(open(sorted(UNIFIED, reverse=True)[0]))


def test_E32_clinician_session_numbers_match_the_export():
    d = newest_export()
    summary, task, attempts = d.get("session_summary", {}), d.get("task", {}), d.get("attempts", [])
    expected = {"Completed Repetitions": summary.get("completed_repetitions", 0),
                "Total Attempts": summary.get("total_attempts", len(attempts)),
                "Interrupted Attempts": summary.get("interrupted_attempts", 0),
                "Target Repetitions": task.get("target_repetitions", "N/A")}
    got = shown(run("clinician_session.py"))
    for label, value in expected.items():
        assert label in got, "page shows no '%s' (it shows %s)" % (label, sorted(got))
        assert got[label] == str(value), "%s: page %r, export %r (session %s)" % (label, got[label], value, d.get("session_id"))


def test_E32_clinician_session_calibration_label_matches_the_export():
    d = newest_export()
    got = shown(run("clinician_session.py"))
    assert "Calibration Status" in got
    norm = lambda s: str(s).replace("_", " ").strip().lower()
    assert norm(got["Calibration Status"]) == norm(d["posture_calibration"]["status"]), \
        "page %r, export %r" % (got["Calibration Status"], d["posture_calibration"]["status"])


def test_E32_every_stored_quality_warning_is_shown():
    d = newest_export()
    stored = [w for a in d.get("attempts", []) for w in (a.get("quality_warnings") or [])]
    at = run("clinician_session.py")
    texts = [str(w.value) for w in at.warning] + [str(i.value) for i in at.info]
    missing = [w for w in stored if not any(str(w) in t for t in texts)]
    assert not missing, "stored warnings not shown on the page: %s" % missing
    success = [str(s.value) for s in at.success]
    if not stored:
        assert any("No quality warnings" in s for s in success), "no stored warnings, but the page does not say so"
    else:
        assert not any("No quality warnings" in s for s in success), "page says there are no warnings, but %d are stored" % len(stored)


def test_E32_patient_page_matches_an_export():
    assert UNIFIED, "no unified assessment files found"
    triples = {}
    for p in UNIFIED:
        d = json.load(open(p))
        s, t, a = d.get("session_summary", {}), d.get("task", {}), d.get("attempts", [])
        triples[(str(s.get("completed_repetitions", 0)), str(s.get("total_attempts", len(a))), str(t.get("target_repetitions", "—")))] = d.get("session_id")
    got = shown(run("patient_dashboard.py"))
    page = (got.get("Completed repetitions"), got.get("Total attempts"), got.get("Target repetitions"))
    assert page in triples, "page shows %s, no export has these values: %s" % (page, triples)
    print("patient page matches session", triples[page])


def test_E32_longitudinal_totals_match_the_trajectory_files_per_patient():
    """The page shows one patient at a time (first selectbox). Pick each patient and compare with the files."""
    assert TRAJ, "no trajectory results found"
    sessions = [json.load(open(p)) for p in TRAJ]
    patients = sorted({s.get("patient_id") for s in sessions if s.get("patient_id") is not None})
    logging.disable(logging.CRITICAL)
    at = AppTest.from_file(str(DASH / "clinician_longitudinal.py"), default_timeout=120).run()
    assert not at.exception
    bad = {}
    for pat in patients:
        at.selectbox[0].select(pat).run()
        assert not at.exception, "page raised for patient %s" % pat
        mine = [s for s in sessions if s.get("patient_id") == pat]
        att = [a for s in mine for a in (s.get("attempts") or [])]
        expected = {"Recorded Sessions": len(mine),
                    "Completed Attempts": sum(a.get("outcome") == "COMPLETED" for a in att),
                    "Interrupted Attempts": sum(a.get("outcome") == "INTERRUPTED" for a in att),
                    "Total Attempts": len(att)}
        got = shown(at)
        wrong = {k: (got.get(k), str(v)) for k, v in expected.items() if got.get(k) != str(v)}
        if wrong:
            bad[pat] = wrong
    assert not bad, "page vs files (page, files) by patient: %s" % bad
