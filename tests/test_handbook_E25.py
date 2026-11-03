"""Handbook E25: task completion. Start dwell, progress and target dwell (2 Nov 2026).

Rules under test, from src/assessment/game_trajectory.py:
  start     object within 100 px of the start point for 1.5 s with the same id
  progress  follows the path, never goes down, only counts within max(40, target radius) px of the path
  finish    object inside the target radius with progress >= 0.95 for 1.5 s; losing tracking or leaving
            the target restarts the hold
The clock is replaced, so every time value is exact and the tests take no real time.
"""
from unittest import mock

import pytest

from src.assessment import game_trajectory as gt

START, TARGET = (200, 400), (800, 400)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    c = Clock()
    with mock.patch.object(gt.time, "monotonic", c):
        yield c


def new():
    return gt.GameTrajectory(START, TARGET, target_radius=50, waypoints=[], object_name="cup")


def started(clock, g=None):
    g = g or new()
    g.start_attempt(object_id=1, object_class="cup")
    return g


def test_E25_start_needs_the_full_1_5_seconds(clock):
    g = new()
    assert g.consider_start(1, START) is False
    clock.t = 100.75
    assert g.consider_start(1, START) is False
    assert 0.45 < g.start_confirmation_progress < 0.55
    clock.t = 101.25
    assert g.consider_start(1, START) is False
    clock.t = 101.5
    assert g.consider_start(1, START) is True
    assert g.attempt_active and g.outcome == "ACTIVE" and g.active_object_id == 1


def test_E25_start_tolerance_boundary_is_100_px(clock):
    g = new()
    g.consider_start(1, (300, 400))
    assert g.start_hold_since is not None
    g.consider_start(1, (301, 400))
    assert g.start_hold_since is None


def test_E25_leaving_the_start_zone_restarts_the_dwell(clock):
    g = new()
    g.consider_start(1, START)
    clock.t = 101.0
    assert g.consider_start(1, (400, 400)) is False
    clock.t = 101.25
    g.consider_start(1, START)
    clock.t = 102.5
    assert g.consider_start(1, START) is False
    clock.t = 102.75
    assert g.consider_start(1, START) is True


def test_E25_a_different_object_id_restarts_the_dwell(clock):
    g = new()
    g.consider_start(1, START)
    clock.t = 101.0
    g.consider_start(2, START)
    clock.t = 102.25
    assert g.consider_start(2, START) is False
    clock.t = 102.5
    assert g.consider_start(2, START) is True and g.active_object_id == 2


def test_E25_no_start_without_an_object_or_while_active(clock):
    g = new()
    assert g.consider_start(1, None) is False
    g = started(clock)
    assert g.consider_start(1, START) is False


def test_E25_progress_follows_the_path_and_never_decreases(clock):
    g = started(clock)
    g.update((500, 400), object_id=1, timestamp_ms=1000)
    assert 0.45 <= g.progress_fraction <= 0.55
    before = g.progress_fraction
    g.update((300, 400), object_id=1, timestamp_ms=1200)
    assert g.progress_fraction == before


def test_E25_progress_ignores_positions_far_from_the_path(clock):
    g = started(clock)
    g.update((500, 600), object_id=1, timestamp_ms=1000)
    assert g.progress_fraction == 0.0
    g.update((500, 440), object_id=1, timestamp_ms=1200)
    assert g.progress_fraction > 0.4


def test_E25_other_object_ids_are_ignored(clock):
    g = started(clock)
    g.update((700, 400), object_id=2, timestamp_ms=1000)
    assert g.progress_fraction == 0.0 and g.actual_path == []


def at_target(clock):
    g = started(clock)
    g.update((500, 400), object_id=1, timestamp_ms=1000)
    g.update((800, 400), object_id=1, timestamp_ms=2000)
    assert g.progress_fraction >= 0.95 and g.target_reached()
    return g


def test_E25_no_success_before_the_target_hold_is_complete(clock):
    g = at_target(clock)
    clock.t = 110.0
    assert g.update_attempt_state(object_visible=True) is False
    clock.t = 111.25
    assert g.update_attempt_state(object_visible=True) is False
    assert g.attempt_active and g.outcome == "ACTIVE"
    clock.t = 111.5
    assert g.update_attempt_state(object_visible=True) is True
    assert g.outcome == "COMPLETED" and g.attempt_completed and not g.attempt_active


def test_E25_leaving_the_target_restarts_the_hold(clock):
    g = at_target(clock)
    clock.t = 110.0
    g.update_attempt_state(object_visible=True)
    clock.t = 110.75
    g.update((600, 400), object_id=1, timestamp_ms=3000)
    assert g.update_attempt_state(object_visible=True) is False
    assert g.target_hold_since is None
    clock.t = 111.0
    g.update((800, 400), object_id=1, timestamp_ms=4000)
    g.update_attempt_state(object_visible=True)
    clock.t = 112.25
    assert g.update_attempt_state(object_visible=True) is False
    clock.t = 112.5
    assert g.update_attempt_state(object_visible=True) is True


def test_E25_losing_tracking_restarts_the_hold(clock):
    g = at_target(clock)
    clock.t = 110.0
    g.update_attempt_state(object_visible=True)
    clock.t = 110.5
    assert g.update_attempt_state(object_visible=False) is False
    assert g.outcome == "TRACKING LOST" and g.target_hold_since is None and g.attempt_active
    clock.t = 110.75
    g.update_attempt_state(object_visible=True)
    clock.t = 112.0
    assert g.update_attempt_state(object_visible=True) is False
    clock.t = 112.25
    assert g.update_attempt_state(object_visible=True) is True


def test_E25_incomplete_progress_never_completes(clock):
    g = at_target(clock)
    g.progress_fraction = 0.9
    clock.t = 110.0
    g.update_attempt_state(object_visible=True)
    clock.t = 130.0
    assert g.update_attempt_state(object_visible=True) is False
    assert g.attempt_active and g.outcome == "ACTIVE"


def test_E25_manual_end_outcomes(clock):
    g = started(clock)
    g.update((500, 400), object_id=1, timestamp_ms=1000)
    g.end_attempt()
    assert g.outcome == "INCOMPLETE" and g.attempt_completed is False
    g = at_target(clock)
    g.end_attempt()
    assert g.outcome == "COMPLETED" and g.attempt_completed is True
