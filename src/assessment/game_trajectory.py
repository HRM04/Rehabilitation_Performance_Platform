"""
Game visualization for the existing trajectory assessment.

Keeps the original assessment calculations while adding:
- Blue dotted remaining path
- Green solid completed path
- Red actual movement path
- Object-specific start/target silhouettes
- Time-based interaction-loss tolerance
"""

import math
import time

import cv2
import numpy as np

from src.assessment.trajectory_assessment import TrajectoryAssessment


class GameTrajectory(TrajectoryAssessment):

    def __init__(
        self,
        *args,
        object_name="bottle",
        grace_seconds=3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.object_name = object_name
        self.grace_seconds = grace_seconds
        self.last_interaction_time = None
        self.progress_fraction = 0.0
        self.start_hold_seconds = 1.5
        self.start_tolerance_px = 100
        self.start_hold_since = None
        self.start_candidate_id = None
        self.start_confirmation_progress = 0.0
        self.outcome = "WAITING"
        self.last_seen_position = None
        self.last_detection_time = None
        self.recovery_distance_px = 120
        self.completion_progress_threshold = 0.95
        self.target_hold_seconds = 1.5
        self.target_hold_since = None
        self.go_display_until = 0.0
        self.retry_ready_at = 0.0
        self.completion_display_until = 0.0
        self.tracking_restored_until = 0.0

    def reset(self):
        super().reset()
        self.last_interaction_time = None
        self.progress_fraction = 0.0
        self.clear_start_confirmation()
        self.outcome = "WAITING"
        self.completion_display_until = 0.0
        self.tracking_restored_until = 0.0
        self.last_seen_position = None
        self.last_detection_time = None
        self.target_hold_since = None

    def start_attempt(self, object_id=None):
        super().start_attempt(object_id=object_id)
        self.last_interaction_time = time.monotonic()
        self.progress_fraction = 0.0
        self.outcome = "ACTIVE"
        self.tracking_restored_until = 0.0
        self.go_display_until = time.monotonic() + 1.2
        self.last_seen_position = tuple(self.start_point)
        self.last_detection_time = None
        self.target_hold_since = None
        self.clear_start_confirmation()

    def end_attempt(self, outcome=None):
        if not self.attempt_active:
            return

        if outcome is None:
            outcome = (
                "COMPLETED"
                if self.target_reached()
                and self.progress_fraction >= self.completion_progress_threshold
                else "INCOMPLETE"
            )

        super().end_attempt()
        self.outcome = outcome
        self.attempt_completed = outcome == "COMPLETED"
        if outcome == "COMPLETED":
            self.completion_display_until = time.monotonic() + 2.0
        self.last_interaction_time = None

        if outcome == "INTERRUPTED":
            self.retry_ready_at = time.monotonic() + 1.0

    def update_attempt_state(
        self,
        interaction_active=False,
        grace_frames=10,
        object_visible=None,
    ):
        if not self.attempt_active:
            return False

        now = time.monotonic()

        if object_visible is None:
            object_visible = interaction_active

        if not object_visible:
            if self.last_detection_time is None:
                self.last_detection_time = now

            # Preserve attempt and measured path.
            # Missing observations do not count as target hold.
            self.outcome = "TRACKING LOST"
            self.target_hold_since = None
            return False

        self.last_detection_time = now
        if self.outcome == "TRACKING LOST":
            self.tracking_restored_until = now + 1.0
        self.outcome = "ACTIVE"

        qualified = (
            self.target_reached()
            and self.progress_fraction
            >= self.completion_progress_threshold
        )

        if not qualified:
            self.target_hold_since = None
            return False

        if self.target_hold_since is None:
            self.target_hold_since = now
            return False

        if now - self.target_hold_since >= self.target_hold_seconds:
            self.end_attempt("COMPLETED")
            return True

        return False

    def recover_object(self, detections):
        if not self.attempt_active:
            return None

        if self.last_seen_position is None:
            return None

        candidates = []

        for obj in detections:
            center = obj.get("center")
            if center is None:
                continue

            distance = math.dist(
                center, self.last_seen_position
            )

            if distance <= self.recovery_distance_px:
                candidates.append(obj)

        if len(candidates) != 1:
            return None

        candidate = candidates[0]

        self.active_object_id = candidate.get("object_id")
        return candidate

    def prepare_retry(self):
        if self.attempt_active:
            return

        if self.outcome != "INTERRUPTED":
            return

        if time.monotonic() < self.retry_ready_at:
            return

        self.reset()

    def clear_start_confirmation(self):
        self.start_hold_since = None
        self.start_candidate_id = None
        self.start_confirmation_progress = 0.0
        self.start_detection_count = 0

    def consider_start(self, object_id, object_center):
        if self.attempt_active or object_center is None:
            self.clear_start_confirmation()
            return False

        if math.dist(object_center, self.start_point) > self.start_tolerance_px:
            self.clear_start_confirmation()
            return False

        now = time.monotonic()

        if self.start_hold_since is None:
            self.start_hold_since = now

        self.start_candidate_id = object_id
        elapsed = max(0.0, now - self.start_hold_since)
        self.start_confirmation_progress = min(
            1.0,
            elapsed / max(self.start_hold_seconds, 0.001)
        )

        if elapsed >= self.start_hold_seconds:
            self.start_attempt(object_id=object_id)
            return True

        return False



    def _draw_start_loading(self, frame):
        if self.attempt_active:
            return

        progress = self.start_confirmation_progress

        if progress <= 0:
            return

        self._draw_silhouette_fill(
            frame, self.start_point, 2.5, progress
        )

    def _draw_silhouette_fill(self, frame, center, scale, progress):
        """Transparent green interior, filled bottom-to-top."""
        outline = np.zeros_like(frame)

        self._draw_object_icon(
            outline, center, (255, 255, 255), scale=scale
        )

        gray = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.drawContours(
            mask, contours, -1, 255, thickness=-1
        )

        ys, xs = np.where(mask > 0)

        if len(ys) == 0:
            return

        top = int(ys.min())
        bottom = int(ys.max())

        progress = max(0.0, min(1.0, float(progress)))

        boundary = bottom - int(
            progress * (bottom - top + 1)
        )

        filled = (
            (mask > 0)
            & (np.arange(frame.shape[0])[:, None] >= boundary)
        )

        alpha = 0.20
        green = np.array([0, 255, 0], dtype=np.float32)

        frame[filled] = (
            frame[filled].astype(np.float32) * (1.0 - alpha)
            + green * alpha
        ).astype(np.uint8)

        cv2.drawContours(
            frame, contours, -1, (0, 255, 0), 2
        )

    def _draw_silhouette_fill(self, frame, center, scale, progress):
        """Transparent green interior, filled bottom-to-top."""
        outline = np.zeros_like(frame)

        self._draw_object_icon(
            outline, center, (255, 255, 255), scale=scale
        )

        gray = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.drawContours(
            mask, contours, -1, 255, thickness=-1
        )

        ys, xs = np.where(mask > 0)

        if len(ys) == 0:
            return

        top = int(ys.min())
        bottom = int(ys.max())

        progress = max(0.0, min(1.0, float(progress)))

        boundary = bottom - int(
            progress * (bottom - top + 1)
        )

        filled = (
            (mask > 0)
            & (np.arange(frame.shape[0])[:, None] >= boundary)
        )

        alpha = 0.20
        green = np.array([0, 255, 0], dtype=np.float32)

        frame[filled] = (
            frame[filled].astype(np.float32) * (1.0 - alpha)
            + green * alpha
        ).astype(np.uint8)

        cv2.drawContours(
            frame, contours, -1, (0, 255, 0), 2
        )



    def _draw_target_loading(self, frame):
        if not self.attempt_active:
            return

        if self.target_hold_since is None:
            return

        if self.outcome == "TRACKING LOST":
            return

        progress = (
            time.monotonic() - self.target_hold_since
        ) / max(self.target_hold_seconds, 0.001)

        self._draw_silhouette_fill(
            frame, self.target_center, 3.5, progress
        )

    def _instruction(self):
        now = time.monotonic()

        if self.outcome == "COMPLETED":
            return "REPETITION COMPLETE - RETURN TO START", (0, 255, 0)

        if self.outcome == "TRACKING LOST":
            return "TRACKING LOST - HOLD POSITION", (0, 165, 255)

        if self.attempt_active:
            if now < self.go_display_until:
                return "GO!", (0, 255, 0)

            if self.target_hold_since is not None:
                progress = min(100, int(
                    100 * (now - self.target_hold_since)
                    / max(self.target_hold_seconds, 0.001)
                ))
                return f"HOLD AT TARGET - {progress}%", (0, 191, 255)

            if now < self.tracking_restored_until:
                return "TRACKING RESTORED - CONTINUE", (0, 255, 0)

            if self.progress_fraction >= self.completion_progress_threshold:
                return "MOVE OBJECT INTO TARGET", (0, 191, 255)

            return "MOVE TOWARD TARGET", (255, 255, 255)

        if self.start_confirmation_progress > 0:
            return (
                f"HOLD AT START - "
                f"{int(self.start_confirmation_progress * 100)}%",
                (0, 255, 0)
            )

        return "MOVE OBJECT TO START", (255, 255, 255)

    def _draw_instruction(self, frame):
        message, color = self._instruction()

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(0.9, max(0.42, frame.shape[1] / 1500.0))
        thickness = 2

        (width, height), _ = cv2.getTextSize(
            message, font, scale, thickness
        )

        x = max(8, (frame.shape[1] - width) // 2)
        y = min(frame.shape[0] - 12, max(115, height + 80))

        cv2.putText(
            frame, message, (x, y), font, scale,
            (0, 0, 0), thickness + 4, cv2.LINE_AA
        )
        cv2.putText(
            frame, message, (x, y), font, scale,
            color, thickness, cv2.LINE_AA
        )

    def _project_to_path(self, point):
        """Return closest path position, measured in pixels."""
        path = self.desired_path

        if len(path) < 2:
            return 0.0, 0.0

        cumulative = 0.0
        best_distance = float("inf")
        best_position = 0.0
        total_length = 0.0

        for a, b in zip(path[:-1], path[1:]):
            total_length += math.dist(a, b)

        for a, b in zip(path[:-1], path[1:]):
            ax, ay = a
            bx, by = b
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                continue

            t = max(
                0.0,
                min(
                    1.0,
                    (
                        (point[0] - ax) * dx
                        + (point[1] - ay) * dy
                    ) / length_sq,
                ),
            )

            projection = (ax + t * dx, ay + t * dy)
            distance = math.dist(point, projection)
            segment_length = math.sqrt(length_sq)

            if distance < best_distance:
                best_distance = distance
                best_position = cumulative + t * segment_length

            cumulative += segment_length

        fraction = (
            best_position / total_length
            if total_length > 0
            else 0.0
        )
        return fraction, best_distance

    def update(self, object_center, object_id=None, timestamp_ms=None):
        before = len(self.actual_path)

        super().update(
            object_center,
            object_id=object_id,
            timestamp_ms=timestamp_ms,
        )

        if len(self.actual_path) == before:
            return

        self.last_seen_position = self.actual_path[-1]
        self.last_detection_time = time.monotonic()

        fraction, distance = self._project_to_path(
            self.actual_path[-1]
        )

        # Only advance progress when the object is close
        # enough to the desired path.
        tolerance_px = max(40, self.target_radius)

        if distance <= tolerance_px:
            self.progress_fraction = max(
                self.progress_fraction,
                fraction,
            )

    def _draw_path(self, frame):
        path = self.desired_path

        if len(path) < 2:
            return

        lengths = [
            math.dist(a, b)
            for a, b in zip(path[:-1], path[1:])
        ]
        total = sum(lengths)

        if total <= 0:
            return

        completed_distance = self.progress_fraction * total
        travelled = 0.0

        for (a, b), length in zip(
            zip(path[:-1], path[1:]),
            lengths,
        ):
            if length <= 0:
                continue

            steps = max(1, math.ceil(length / 5))

            for step in range(steps):
                t0 = step / steps
                t1 = (step + 1) / steps

                p0 = (
                    int(a[0] + (b[0] - a[0]) * t0),
                    int(a[1] + (b[1] - a[1]) * t0),
                )
                p1 = (
                    int(a[0] + (b[0] - a[0]) * t1),
                    int(a[1] + (b[1] - a[1]) * t1),
                )

                midpoint_distance = (
                    travelled + (t0 + t1) * 0.5 * length
                )

                if midpoint_distance <= completed_distance:
                    # Green completed segment.
                    cv2.line(
                        frame, p0, p1, (0, 210, 0), 4
                    )
                else:
                    # Blue dotted remaining segment.
                    # Never draw blue underneath green.
                    if (int(midpoint_distance / 10) % 2) == 0:
                        cv2.line(
                            frame, p0, p1, (255, 150, 0), 3
                        )

            travelled += length

    def _draw_object_icon(self, frame, center, color, scale=1.0):
        x, y = map(int, center)
        name = self.object_name.lower()

        def scaled(value):
            return max(1, int(round(value * scale)))

        if name in ("sports ball", "apple"):
            cv2.circle(frame, (x, y), scaled(28), color, 2)

        elif name in ("cell phone", "remote", "book"):
            w, h = (
                (30, 56) if name == "cell phone"
                else (24, 70) if name == "remote"
                else (58, 44)
            )
            w, h = scaled(w), scaled(h)
            cv2.rectangle(
                frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                color,
                2,
            )

        elif name == "bottle":
            points = np.array([
                [x - 10, y - 35],
                [x + 10, y - 35],
                [x + 10, y - 22],
                [x + 20, y - 15],
                [x + 20, y + 35],
                [x - 20, y + 35],
                [x - 20, y - 15],
                [x - 10, y - 22],
            ], dtype=np.int32)
            cv2.polylines(frame, [points], True, color, 2)

        elif name in ("cup", "bowl"):
            w = 38 if name == "cup" else 55
            h = 42 if name == "cup" else 25
            cv2.rectangle(
                frame,
                (x - w // 2, y - h // 2),
                (x + w // 2, y + h // 2),
                color,
                2,
            )
            if name == "cup":
                cv2.ellipse(
                    frame,
                    (x + w // 2 + 7, y),
                    (9, 13),
                    0, -90, 90, color, 2,
                )

        elif name in ("spoon", "fork", "toothbrush"):
            cv2.line(
                frame, (x, y - 30), (x, y + 35),
                color, 3,
            )
            cv2.ellipse(
                frame,
                (x, y - 24),
                (9, 15),
                0, 0, 360, color, 2,
            )

        else:
            # Provisional generic silhouette for objects
            # without a dedicated icon yet.
            cv2.rectangle(
                frame,
                (x - 22, y - 28),
                (x + 22, y + 28),
                color,
                2,
            )

        cv2.circle(frame, (x, y), 3, color, -1)

    def draw(self, frame):
        # Draw desired path with mutually exclusive colors.
        self._draw_path(frame)

        # The starting silhouette follows accepted object
        # observations during an active repetition.
        # The prescribed target remains stationary.
        moving_center = self.start_point
        moving_color = (255, 255, 255)

        if self.attempt_active and self.actual_path:
            moving_center = self.actual_path[-1]

            if self.outcome == "TRACKING LOST":
                moving_color = (0, 165, 255)

        self._draw_object_icon(
            frame, moving_center,
            moving_color, scale=2.5,
        )
        self._draw_object_icon(
            frame, self.target_center,
            (0, 220, 0), scale=3.5,
        )
        self._draw_start_loading(frame)
        self._draw_target_loading(frame)

        # Show the exact desired centre and the
        # acceptable starting region.
        if not self.attempt_active:
            sx, sy = map(int, self.start_point)

            overlay = frame.copy()
            cv2.circle(
                overlay, (sx, sy),
                int(self.start_tolerance_px),
                (0, 215, 255), -1,
            )
            cv2.addWeighted(
                overlay, 0.12, frame, 0.88, 0, frame
            )

            cv2.circle(
                frame, (sx, sy),
                int(self.start_tolerance_px),
                (0, 215, 255), 2,
            )

            cv2.line(
                frame, (sx - 18, sy),
                (sx + 18, sy),
                (0, 255, 255), 3,
            )
            cv2.line(
                frame, (sx, sy - 18),
                (sx, sy + 18),
                (0, 255, 255), 3,
            )
            cv2.circle(
                frame, (sx, sy), 5,
                (0, 255, 255), -1,
            )

        cv2.putText(
            frame, "START",
            (self.start_point[0] - 25,
             self.start_point[1] - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, "TARGET",
            (self.target_center[0] - 30,
             self.target_center[1] - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 220, 0), 2,
        )

        # Red: actual movement trace.
        if len(self.actual_path) >= 2:
            points = np.asarray(
                self.actual_path, dtype=np.int32
            )
            cv2.polylines(
                frame, [points], False, (0, 0, 255), 3
            )

        if self.actual_path:
            cv2.circle(
                frame, self.actual_path[-1],
                7, (0, 0, 255), -1,
            )

        status = self.outcome

        if not self.attempt_active and self.outcome == "WAITING":
            status = (
                "HOLD AT START"
                if self.start_candidate_id is not None
                else "MOVE TO START"
            )

        if (
            self.attempt_active
            and time.monotonic() < self.go_display_until
        ):
            go_text = "GO!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = min(3.0, max(1.5, frame.shape[1] / 430.0))
            thickness = max(3, int(scale * 2))
            (text_width, text_height), _ = cv2.getTextSize(
                go_text, font, scale, thickness
            )
            position = (
                max(0, (frame.shape[1] - text_width) // 2),
                (frame.shape[0] + text_height) // 2,
            )
            cv2.putText(
                frame, go_text, position,
                font, scale, (0, 255, 0),
                thickness, cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"{self.object_name.upper()} | {status} | "
            f"Progress: {self.progress_fraction * 100:.0f}%",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2,
        )

        self._draw_instruction(frame)
        return frame
