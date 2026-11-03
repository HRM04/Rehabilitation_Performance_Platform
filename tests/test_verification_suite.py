"""Verification suite with known-answer inputs (register items A1 to A8).

Run from the project folder:  python -m pytest -q tests/test_verification_suite.py
A failure here is a finding to report, not necessarily a code defect.
"""
import csv, math, os, random, tempfile, unittest

import numpy as np

from src.assessment.trajectory_assessment import TrajectoryAssessment
from src.object_detection.interaction_event_tracker import InteractionEventTracker
from src.object_detection.duplicate_filter import filter_duplicate_objects
from src.assessment.joint_quality_control import assess_joint_quality
from src.assessment.baseline_comparison import calculate_relative_change
from src.assessment.feedback_generator import generate_feedback
from src.assessment.repetition_analysis import segment_repetitions
from src.preprocessing.landmark_smoothing import HandLandmarkSmoother


class Lm:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class HandLms:
    def __init__(self, pts):
        self.landmark = [Lm(*p) for p in pts]


def traj(start, target, radius=50):
    t = TrajectoryAssessment(start, target, radius)
    t.start_attempt()
    return t


class TestTrajectoryMetrics(unittest.TestCase):
    def test_straight_path(self):
        t = traj((0, 0), (100, 0))
        for x in range(0, 101, 10):
            t.update((x, 0))
        m = t.get_metrics()
        self.assertAlmostEqual(m["path_length_px"], 100.0)
        self.assertAlmostEqual(m["direct_displacement_px"], 100.0)
        self.assertAlmostEqual(m["movement_efficiency"], 1.0)
        self.assertAlmostEqual(m["mean_trajectory_deviation_px"], 0.0)
        self.assertAlmostEqual(m["final_target_error_px"], 0.0)
        self.assertTrue(m["target_reached"])

    def test_l_shaped_path_against_diagonal(self):
        t = traj((0, 0), (100, 100))
        for p in [(0, 0), (100, 0), (100, 100)]:
            t.update(p)
        m = t.get_metrics()
        self.assertAlmostEqual(m["path_length_px"], 200.0)
        self.assertAlmostEqual(m["direct_displacement_px"], 141.4214, places=3)
        self.assertAlmostEqual(m["movement_efficiency"], 0.70711, places=4)
        self.assertAlmostEqual(m["mean_trajectory_deviation_px"], 70.7107 / 3, places=3)

    def test_target_boundary(self):
        for dx, expected in ((50, True), (51, False)):
            t = traj((0, 0), (200, 0), radius=50)
            t.update((200 - dx, 0))
            self.assertEqual(t.target_reached(), expected, dx)

    def test_other_object_id_ignored(self):
        t = TrajectoryAssessment((0, 0), (100, 0))
        t.start_attempt(object_id=7)
        t.update((10, 0), object_id=7)
        t.update((500, 500), object_id=9)
        self.assertEqual(len(t.actual_path), 1)

    def test_no_data_gives_none_not_zero(self):
        t = traj((0, 0), (100, 0))
        m = t.get_metrics()
        self.assertIsNone(m["movement_efficiency"])
        self.assertIsNone(m["final_target_error_px"])
        self.assertFalse(m["target_reached"])

    def test_all_five_presets(self):
        for name in ("horizontal", "vertical", "diagonal", "curved", "s_shaped"):
            t = TrajectoryAssessment((0, 0), (1, 1))
            t.set_preset(name)
            self.assertGreaterEqual(len(t.desired_path), 2, name)
            for x, y in t.desired_path:
                self.assertTrue(0 <= x <= 1280 and 0 <= y <= 720, (name, x, y))
        with self.assertRaises(ValueError):
            TrajectoryAssessment((0, 0), (1, 1)).set_preset("zigzag")

    def test_attempt_ends_after_grace(self):
        t = traj((0, 0), (1, 1))
        ended = [t.update_attempt_state(False, grace_frames=3) for _ in range(3)]
        self.assertEqual(ended, [False, False, True])


class TestInteractionEvents(unittest.TestCase):
    def feed(self, tr, seq, hand="Right", oid=1):
        out = []
        for i, (state, mv) in enumerate(seq):
            out.append(tr.update(hand, oid, state, mv, timestamp_ms=i * 100))
        return out

    def test_grasp_release_regrasp(self):
        tr = InteractionEventTracker()
        ev = self.feed(tr, [("NONE", 0), ("NEAR", 0), ("GRASPING", 0), ("CONTACT", 0), ("GRASPING", 0)])
        self.assertEqual(ev, [None, None, "GRASP", "RELEASE", "REGRASP"])

    def test_movement_categories(self):
        tr = InteractionEventTracker()
        ev = self.feed(tr, [("GRASPING", 0), ("GRASPING", 0), ("GRASPING", 5), ("GRASPING", 12), ("GRASPING", 12), ("GRASPING", 1.9)])
        self.assertEqual(ev, ["GRASP", "HOLDING", "MANIPULATION", "TRANSPORT", None, "HOLDING"])

    def test_movement_thresholds_exact(self):
        tr = InteractionEventTracker()
        ev = self.feed(tr, [("GRASPING", 0), ("GRASPING", 2.0)])
        self.assertEqual(ev[1], "MANIPULATION")
        tr = InteractionEventTracker()
        ev = self.feed(tr, [("GRASPING", 0), ("GRASPING", 10.0)])
        self.assertEqual(ev[1], "TRANSPORT")

    def test_release_duration(self):
        tr = InteractionEventTracker()
        self.feed(tr, [("GRASPING", 0), ("GRASPING", 0), ("GRASPING", 0), ("NEAR", 0)])
        self.assertEqual(tr.pair_history[("Right", 1)]["last_grasp_duration_ms"], 300)

    def test_missing_frames_do_not_release(self):
        tr = InteractionEventTracker()
        self.feed(tr, [("GRASPING", 0)])
        # hand not visible: no update calls at all, then visible again
        ev = tr.update("Right", 1, "GRASPING", 0, timestamp_ms=900)
        self.assertNotIn(ev, ("RELEASE", "REGRASP"))

    def test_prolonged_loss_resets_history(self):
        tr = InteractionEventTracker()
        self.feed(tr, [("GRASPING", 0)])
        tr.mark_unobserved("Right")
        ev = tr.update("Right", 1, "GRASPING", 0, timestamp_ms=5000)
        self.assertEqual(ev, "GRASP")

    def test_hands_are_independent(self):
        tr = InteractionEventTracker()
        self.feed(tr, [("GRASPING", 0)], hand="Left", oid=1)
        ev = self.feed(tr, [("GRASPING", 0)], hand="Right", oid=2)
        self.assertEqual(ev, ["GRASP"])


def det(cls, box, conf):
    return {"class_name": cls, "bounding_box": box, "confidence": conf}


class TestDuplicateFilter(unittest.TestCase):
    def test_identical_boxes_collapse_keep_best(self):
        kept, removed = filter_duplicate_objects([det("cup", (0, 0, 100, 100), .6), det("cup", (0, 0, 100, 100), .9)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["confidence"], .9)
        self.assertEqual(len(removed), 1)

    def test_two_separate_same_class_objects_kept(self):
        kept, _ = filter_duplicate_objects([det("cup", (0, 0, 100, 100), .8), det("cup", (300, 0, 400, 100), .8)])
        self.assertEqual(len(kept), 2)

    def test_adjacent_same_class_objects_kept(self):
        kept, _ = filter_duplicate_objects([det("cup", (0, 0, 100, 100), .8), det("cup", (90, 0, 190, 100), .7)])
        self.assertEqual(len(kept), 2)

    def test_different_classes_same_box_kept(self):
        kept, _ = filter_duplicate_objects([det("cup", (0, 0, 100, 100), .8), det("bottle", (0, 0, 100, 100), .7)])
        self.assertEqual(len(kept), 2)

    def test_nested_small_box_removed(self):
        kept, removed = filter_duplicate_objects([det("cup", (0, 0, 100, 100), .9), det("cup", (20, 20, 60, 60), .5)])
        self.assertEqual(len(kept), 1)

    def test_large_box_covering_two_objects_flagged(self):
        kept, removed = filter_duplicate_objects([
            det("cup", (0, 0, 400, 100), .9), det("cup", (0, 0, 100, 100), .8), det("cup", (300, 0, 400, 100), .8)])
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(removed), 1)


def frames(angles, dt=100, key="left_elbow"):
    return [{"timestamp_ms": i * dt, "angles": {key: a}} for i, a in enumerate(angles)]


class TestJointScreening(unittest.TestCase):
    def test_smooth_signal_sufficient(self):
        r = assess_joint_quality(frames([90 + i for i in range(20)]), "left_elbow")
        self.assertEqual(r["status"], "SUFFICIENT")

    def test_too_few_observations(self):
        r = assess_joint_quality(frames([90 + i for i in range(9)]), "left_elbow")
        self.assertEqual(r["status"], "REVIEW")
        self.assertFalse(r["clinician_display_allowed"])

    def test_exactly_ten_observations(self):
        r = assess_joint_quality(frames([90 + i for i in range(10)]), "left_elbow")
        self.assertEqual(r["status"], "SUFFICIENT")

    def test_jump_flagged(self):
        a = [90.0] * 10 + [135.0] + [135.0] * 10
        r = assess_joint_quality(frames(a, dt=1000), "left_elbow")
        self.assertEqual(r["flagged_transition_count"], 1)
        self.assertEqual(r["status"], "REVIEW")

    def test_speed_flag_without_big_jump(self):
        a = [90.0, 90.0] + [90 + 20 * i for i in range(1, 4)] + [150.0] * 8
        r = assess_joint_quality(frames(a, dt=50), "left_elbow")  # 20 deg in 50 ms = 400 deg/s
        self.assertGreaterEqual(r["flagged_transition_count"], 1)

    def test_coverage_threshold(self):
        a = [90 + i if i % 2 == 0 else None for i in range(40)]
        r = assess_joint_quality(frames(a), "left_elbow")
        self.assertEqual(r["coverage_percent"], 50.0)
        self.assertEqual(r["status"], "REVIEW")

    def test_nan_not_counted(self):
        a = [90 + i for i in range(14)] + [float("nan")] * 6
        r = assess_joint_quality(frames(a), "left_elbow")
        self.assertEqual(r["valid_observations"], 14)
        self.assertEqual(r["coverage_percent"], 70.0)


class TestBaselineAndFeedback(unittest.TestCase):
    def test_relative_change(self):
        self.assertAlmostEqual(calculate_relative_change(100, 120), 20.0)
        self.assertAlmostEqual(calculate_relative_change(100, 80), -20.0)
        self.assertAlmostEqual(calculate_relative_change(-50, -25), 50.0)
        self.assertIsNone(calculate_relative_change(0, 5))
        self.assertIsNone(calculate_relative_change(None, 5))
        self.assertIsNone(calculate_relative_change(5, None))

    def cats(self, comp):
        return [f["category"] for f in generate_feedback(comp)]

    def test_no_comparison_gives_no_feedback(self):
        self.assertEqual(generate_feedback(None), [])

    def test_missing_values_trigger_nothing(self):
        comp = {"duration_change_percent": None, "joints": {"left_elbow": {"rom_deg": None}}, "compensation": {}}
        self.assertEqual(self.cats(comp), ["no_flagged_change"])

    def test_rom_boundary(self):
        self.assertIn("range_of_motion", self.cats({"joints": {"left_elbow": {"rom_deg": -15.0}}}))
        self.assertNotIn("range_of_motion", self.cats({"joints": {"left_elbow": {"rom_deg": -14.9}}}))

    def test_duration_boundary(self):
        self.assertIn("task_duration", self.cats({"duration_change_percent": 20.0}))
        self.assertNotIn("task_duration", self.cats({"duration_change_percent": 19.9}))

    def test_speed_trunk_reversal(self):
        c = self.cats({"joints": {"left_elbow": {"mean_angular_speed_deg_s": -20.0, "velocity_reversals": 25.0}},
                       "compensation": {"trunk_displacement_change_percent": 25.0}})
        for k in ("movement_speed", "movement_control", "trunk_movement"):
            self.assertIn(k, c)

    def test_improvement_not_flagged(self):
        c = self.cats({"duration_change_percent": -30.0, "joints": {"left_elbow": {"rom_deg": 40.0}}})
        self.assertEqual(c, ["no_flagged_change"])


def row(frame, state, mv=0.0, cls="bottle", hand="Right", oid=None):
    return {"frame_number": frame, "timestamp_ms": frame * 100, "interaction_state": state,
            "object_class": cls, "object_movement": mv, "handedness": hand, "object_id": oid}


def seq(start, n, state, mv=0.0, **kw):
    return [row(f, state, mv, **kw) for f in range(start, start + n)]


class TestRepetitions(unittest.TestCase):
    def valid(self, **kw):
        return seq(0, 3, "GRASPING", 5.0, **kw) + seq(3, 7, "TRANSPORT", 10.0, **kw) + seq(10, 5, "NONE", **kw)

    def test_every_task_object(self):
        for cls in ("bottle", "cup", "spoon", "apple", "cell phone"):
            reps = segment_repetitions(self.valid(cls=cls))
            self.assertEqual(len(reps), 1, cls)
            self.assertEqual(reps[0]["duration_ms"], 900)
            self.assertEqual(reps[0]["start_frame"], 0)

    def test_person_is_not_an_object(self):
        self.assertEqual(len(segment_repetitions(self.valid(cls="person"))), 0)

    def test_too_short_rejected(self):
        rows = seq(0, 3, "GRASPING", 50.0) + seq(3, 5, "NONE")
        self.assertEqual(len(segment_repetitions(rows)), 0)

    def test_too_little_movement_rejected(self):
        rows = seq(0, 3, "GRASPING", 1.0) + seq(3, 7, "HOLDING", 1.0) + seq(10, 5, "NONE")
        self.assertEqual(len(segment_repetitions(rows)), 0)

    def test_start_needs_three_frames(self):
        rows = seq(0, 2, "GRASPING", 50.0) + seq(2, 10, "NONE")
        self.assertEqual(len(segment_repetitions(rows)), 0)

    def test_short_dropout_tolerated(self):
        rows = seq(0, 3, "GRASPING", 5.0) + seq(3, 3, "TRANSPORT", 10.0) + seq(6, 3, "NONE") + seq(9, 4, "TRANSPORT", 10.0) + seq(13, 5, "NONE")
        reps = segment_repetitions(rows)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["end_frame"], 12)

    def test_long_dropout_splits(self):
        rows = self.valid() + seq(15, 3, "GRASPING", 5.0) + seq(18, 7, "TRANSPORT", 10.0) + seq(25, 5, "NONE")
        self.assertEqual(len(segment_repetitions(rows)), 2)

    def test_incomplete_repetition_identifiable(self):
        rows = seq(0, 3, "GRASPING", 5.0) + seq(3, 7, "TRANSPORT", 10.0)
        reps = segment_repetitions(rows)
        self.assertEqual(len(reps), 1)
        self.assertTrue(reps[0].get("ended_with_recording"))

    def test_hands_segmented_separately(self):
        rows = self.valid(hand="Left") + self.valid(hand="Right")
        reps = segment_repetitions(rows)
        self.assertEqual(sorted(r["handedness"] for r in reps), ["Left", "Right"])

    def test_object_switch_not_merged(self):
        rows = seq(0, 3, "GRASPING", 5.0, oid=1) + seq(3, 7, "TRANSPORT", 10.0, oid=1) + seq(10, 5, "NONE") + seq(15, 3, "GRASPING", 5.0, oid=2) + seq(18, 7, "TRANSPORT", 10.0, oid=2) + seq(25, 5, "NONE")
        reps = segment_repetitions(rows)
        self.assertEqual([r["object_id"] for r in reps], [1, 2])


class TestSmoothing(unittest.TestCase):
    def test_constant_is_unchanged(self):
        s = HandLandmarkSmoother(enabled=True)
        for i in range(10):
            h = HandLms([(0.5, 0.5, 0.1)] * 21)
            s.smooth("Left", h, i * 200)
        self.assertAlmostEqual(h.landmark[5].x, 0.5, places=9)

    def test_reduces_noise(self):
        rnd = random.Random(1)
        s = HandLandmarkSmoother(enabled=True)
        raw, out = [], []
        for i in range(200):
            x = 0.5 + rnd.gauss(0, 0.004)
            h = HandLms([(x, 0.5, 0.0)] * 21)
            s.smooth("Left", h, i * 150)
            raw.append(x); out.append(h.landmark[0].x)
        self.assertLess(np.std(out[20:]), 0.8 * np.std(raw[20:]))

    def test_follows_step_within_a_few_frames(self):
        s = HandLandmarkSmoother(enabled=True)
        v = None
        for i in range(30):
            x = 0.3 if i < 10 else 0.7
            h = HandLms([(x, 0.5, 0.0)] * 21)
            s.smooth("Left", h, i * 200)
            if i == 14: v = h.landmark[0].x
        self.assertGreater(v, 0.65)

    def test_gap_resets_filter(self):
        s = HandLandmarkSmoother(enabled=True)
        s.smooth("Left", HandLms([(0.2, 0.2, 0)] * 21), 0)
        h = HandLms([(0.8, 0.8, 0)] * 21)
        s.smooth("Left", h, 5000)
        self.assertAlmostEqual(h.landmark[0].x, 0.8)

    def test_hands_independent_and_disabled_passthrough(self):
        s = HandLandmarkSmoother(enabled=True)
        s.smooth("Left", HandLms([(0.2, 0.2, 0)] * 21), 0)
        h = HandLms([(0.9, 0.9, 0)] * 21)
        s.smooth("Right", h, 100)
        self.assertAlmostEqual(h.landmark[0].x, 0.9)
        off = HandLandmarkSmoother(enabled=False)
        h2 = HandLms([(0.4, 0.4, 0)] * 21)
        off.smooth("Left", h2, 0)
        self.assertEqual(h2.landmark[0].x, 0.4)


class TestScaleCalibration(unittest.TestCase):
    def test_marker_scale_recovered(self):
        try:
            import cv2
            from src.preprocessing import scale_calibration as sc
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            m = cv2.aruco.generateImageMarker(d, 0, 200) if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(d, 0, 200)
        except Exception as e:
            self.skipTest("aruco unavailable: %s" % e)
        img = np.full((720, 1280), 255, np.uint8)
        img[300:500, 400:600] = m
        found = sc.detect_markers(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        self.assertEqual([i for i, _ in found], [0])
        side, skew = sc.side_stats(found[0][1])
        self.assertAlmostEqual(side, 200, delta=3)
        self.assertLess(skew, 1.02)
        out = sc.convert_px_to_mm({"a_px": 100, "n": {"b_px": 10}, "flag": True}, 0.25)
        self.assertEqual(out["a_mm"], 25.0)
        self.assertEqual(out["n"]["b_mm"], 2.5)
        self.assertNotIn("flag_mm", out)


class TestInteractionLog(unittest.TestCase):
    def test_reason_column_and_no_hand_row(self):
        from src.data.interaction_logger import create_interaction_logger, log_no_hand_interaction
        path = os.path.join(tempfile.mkdtemp(), "i.csv")
        f, w = create_interaction_logger(path)
        log_no_hand_interaction(w, 1, 10, "s", "t", 1, "a", "n", "o")
        f.close()
        rows = list(csv.DictReader(open(path)))
        self.assertIn("appropriateness_reason", rows[0])
        self.assertEqual(rows[0]["appropriateness_reason"], "NO_HAND_DETECTED")
        self.assertEqual(rows[0]["detected_grasp_type"], "")


if __name__ == "__main__":
    unittest.main()
