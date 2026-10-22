import cv2
import numpy as np


class TrajectoryAssessment:
    """
    Tracks an object's observed 2D trajectory against a predefined
    desired path and target region.

    All measurements are currently image-plane measurements in pixels.
    They are preliminary engineering metrics, not clinical measures.
    """

    def __init__(
        self,
        start_point,
        target_center,
        target_radius=50,
        waypoints=None,
    ):
        self.start_point = start_point
        self.target_center = target_center
        self.target_radius = target_radius
        self.waypoints = waypoints or []

        self.desired_path = (
            [tuple(self.start_point)]
            + [tuple(point) for point in self.waypoints]
            + [tuple(self.target_center)]
        )

       

        # Stores the observed object-center positions.
        self.actual_path = []

        # Timestamped trajectory observations.
        # These allow trajectory measurements to be aligned
        # with detected repetition windows.
        self.trajectory_samples = []

        # Tracks whether a trajectory attempt is currently active.
        self.attempt_active = False

        # Object ID being followed during the current attempt.
        self.active_object_id = None

        # Number of consecutive frames for which the meaningful
        # interaction has been absent.
        self.inactive_frame_count = 0

        self.attempt_completed = False

    def set_preset(self, preset_name, frame_width=1280, frame_height=720):
        """
        Configure a predefined trajectory using frame-relative coordinates.

        Presets:
            horizontal
            vertical
            diagonal
            curved
            s_shaped
        """
        preset_name = preset_name.lower()

        # Keep trajectories inside the camera frame.
        def point(x, y):
            return (
                int(x * frame_width),
                int(y * frame_height),
            )

        if preset_name == "horizontal":
            start = point(0.20, 0.55)
            target = point(0.80, 0.55)
            waypoints = []

        elif preset_name == "vertical":
            start = point(0.50, 0.80)
            target = point(0.50, 0.20)
            waypoints = []

        elif preset_name == "diagonal":
            start = point(0.20, 0.80)
            target = point(0.80, 0.20)
            waypoints = []

        elif preset_name == "curved":
            start = point(0.20, 0.70)
            target = point(0.80, 0.70)
            waypoints = [
                point(0.30, 0.48),
                point(0.40, 0.35),
                point(0.50, 0.30),
                point(0.60, 0.35),
                point(0.70, 0.48),
            ]

        elif preset_name == "s_shaped":
            start = point(0.20, 0.75)
            target = point(0.80, 0.25)
            waypoints = [
                point(0.35, 0.75),
                point(0.40, 0.50),
                point(0.50, 0.50),
                point(0.60, 0.50),
                point(0.65, 0.25),
            ]

        else:
            raise ValueError(f"Unknown trajectory preset: {preset_name}")

        self.start_point = start
        self.target_center = target
        self.waypoints = waypoints

        self.desired_path = (
            [tuple(start)]
            + [tuple(p) for p in waypoints]
            + [tuple(target)]
        )

        self.preset_name = preset_name

        # Only allow switching between attempts.
        self.reset()

        print(f"Trajectory preset selected: {preset_name}")

    def reset(self):
        """
        Fully reset trajectory assessment for a new session.
        """

        self.actual_path = []
        self.trajectory_samples = []

        self.attempt_active = False
        self.attempt_completed = False
        self.active_object_id = None
        self.inactive_frame_count = 0

    def start_attempt(self, object_id=None):
        """
        Start a new trajectory attempt.
        """

        self.attempt_completed = False
        self.actual_path = []
        self.attempt_active = True
        self.active_object_id = object_id
        self.inactive_frame_count = 0

    def end_attempt(self):
        """
        End the current trajectory attempt while preserving
        the completed path for review.
        """

        self.attempt_active = False
        self.attempt_completed = True
        self.inactive_frame_count = 0

    def update_attempt_state(
        self,
        interaction_active,
        grace_frames=10,
    ):
        """
        Update the trajectory attempt lifecycle.

        Brief interruptions in interaction evidence are
        tolerated before an attempt is ended.

        Returns True only when the attempt has just ended.
        """

        if not self.attempt_active:
            return False

        if interaction_active:
            self.inactive_frame_count = 0
            return False

        self.inactive_frame_count += 1

        if self.inactive_frame_count >= grace_frames:
            self.end_attempt()
            return True

        return False

    def update(self, object_center, object_id=None, timestamp_ms=None,):
        """
        Add the current tracked-object center to the trajectory.
        """

        if not self.attempt_active:
            return

        if (
            self.active_object_id is not None
            and object_id != self.active_object_id
        ):
            return

        if object_center is None:
            return

        point = (
            int(object_center[0]),
            int(object_center[1]),
        )

        self.actual_path.append(point)

        self.trajectory_samples.append({
            "timestamp_ms": timestamp_ms,
            "object_id": object_id,
            "center": point,
        })


    def get_path_for_window(
        self,
        start_timestamp_ms,
        end_timestamp_ms,
        object_id=None,
    ):
        """
        Return observed object positions belonging to a specific
        repetition time window.
        """

        points = []

        for sample in self.trajectory_samples:

            timestamp_ms = sample.get(
                "timestamp_ms"
            )

            if timestamp_ms is None:
                continue

            if not (
                start_timestamp_ms
                <= timestamp_ms
                <= end_timestamp_ms
            ):
                continue

            if (
                object_id is not None
                and sample.get("object_id") != object_id
            ):
                continue

            points.append(
                sample["center"]
            )

        return points


    def calculate_metrics_for_window(
        self,
        start_timestamp_ms,
        end_timestamp_ms,
        object_id=None,
    ):
        """
        Calculate trajectory metrics for a specific repetition
        time window.

        The live trajectory is not modified.

        All distance measurements are image-plane pixel
        measurements.
        """

        path = self.get_path_for_window(
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            object_id=object_id,
        )

        if not path:
            return None

        # ---------------------------------------------
        # Path length
        # ---------------------------------------------

        path_length = 0.0

        for index in range(1, len(path)):

            previous_point = np.asarray(
                path[index - 1],
                dtype=float,
            )

            current_point = np.asarray(
                path[index],
                dtype=float,
            )

            path_length += np.linalg.norm(
                current_point - previous_point
            )

        # ---------------------------------------------
        # Direct displacement
        # ---------------------------------------------

        if len(path) >= 2:

            start = np.asarray(
                path[0],
                dtype=float,
            )

            end = np.asarray(
                path[-1],
                dtype=float,
            )

            direct_displacement = float(
                np.linalg.norm(end - start)
            )

        else:
            direct_displacement = 0.0

        # ---------------------------------------------
        # Movement efficiency
        # ---------------------------------------------

        if path_length > 0:

            movement_efficiency = (
                direct_displacement
                / float(path_length)
            )

        else:
            movement_efficiency = None

        # ---------------------------------------------
        # Trajectory deviation
        # ---------------------------------------------

        deviations = []

        for actual_point in path:

            segment_distances = []

            for index in range(
                len(self.desired_path) - 1
            ):

                distance = (
                    self._point_to_segment_distance(
                        actual_point,
                        self.desired_path[index],
                        self.desired_path[index + 1],
                    )
                )

                segment_distances.append(
                    distance
                )

            if segment_distances:
                deviations.append(
                    min(segment_distances)
                )

        mean_deviation = (
            float(np.mean(deviations))
            if deviations
            else None
        )

        # ---------------------------------------------
        # Final target error
        # ---------------------------------------------

        final_position = np.asarray(
            path[-1],
            dtype=float,
        )

        target = np.asarray(
            self.target_center,
            dtype=float,
        )

        final_target_error = float(
            np.linalg.norm(
                final_position - target
            )
        )

        target_reached = (
            final_target_error
            <= self.target_radius
        )

        # ---------------------------------------------
        # Return repetition-level trajectory results
        # ---------------------------------------------

        return {
            "sample_count": len(path),

            "path_length_px": float(
                path_length
            ),

            "direct_displacement_px":
                direct_displacement,

            "movement_efficiency":
                movement_efficiency,

            "mean_trajectory_deviation_px":
                mean_deviation,

            "final_target_error_px":
                final_target_error,

            "target_reached":
                target_reached,

            "start_position_px":
                path[0],

            "final_position_px":
                path[-1],
        }

    def calculate_path_length(self):
        """
        Total distance travelled along the observed trajectory.
        """

        if len(self.actual_path) < 2:
            return 0.0

        total_distance = 0.0

        for index in range(
            1,
            len(self.actual_path),
        ):
            previous_point = np.asarray(
                self.actual_path[index - 1],
                dtype=float,
            )

            current_point = np.asarray(
                self.actual_path[index],
                dtype=float,
            )

            total_distance += np.linalg.norm(
                current_point - previous_point
            )

        return float(total_distance)

    def calculate_direct_displacement(self):
        """
        Straight-line displacement between the first and latest
        observed object positions.
        """

        if len(self.actual_path) < 2:
            return 0.0

        start = np.asarray(
            self.actual_path[0],
            dtype=float,
        )

        end = np.asarray(
            self.actual_path[-1],
            dtype=float,
        )

        return float(
            np.linalg.norm(end - start)
        )

    def calculate_movement_efficiency(self):
        """
        Preliminary path efficiency:

            direct displacement / actual path length

        Values closer to 1 indicate a more direct observed path.
        """

        path_length = self.calculate_path_length()

        if path_length <= 0:
            return None

        direct_displacement = (
            self.calculate_direct_displacement()
        )

        return float(
            direct_displacement / path_length
        )

    def calculate_final_target_error(self):
        """
        Distance between the current object position
        and the center of the target region.
        """

        if not self.actual_path:
            return None

        current_position = np.asarray(
            self.actual_path[-1],
            dtype=float,
        )

        target = np.asarray(
            self.target_center,
            dtype=float,
        )

        return float(
            np.linalg.norm(
                current_position - target
            )
        )

    def target_reached(self):
        """
        Determine whether the current object center lies
        inside the target region.
        """

        target_error = (
            self.calculate_final_target_error()
        )

        if target_error is None:
            return False

        return (
            target_error <= self.target_radius
        )

    def _point_to_segment_distance(
        self,
        point,
        segment_start,
        segment_end,
    ):
        """
        Calculate the shortest distance between a point
        and one line segment of the desired trajectory.
        """

        point = np.asarray(point, dtype=float)
        start = np.asarray(
            segment_start,
            dtype=float,
        )
        end = np.asarray(
            segment_end,
            dtype=float,
        )

        segment = end - start
        segment_length_squared = np.dot(
            segment,
            segment,
        )

        if segment_length_squared == 0:
            return float(
                np.linalg.norm(point - start)
            )

        projection = np.dot(
            point - start,
            segment,
        ) / segment_length_squared

        projection = np.clip(
            projection,
            0.0,
            1.0,
        )

        nearest_point = (
            start + projection * segment
        )

        return float(
            np.linalg.norm(
                point - nearest_point
            )
        )

    def calculate_trajectory_deviation(self):
        """
        Calculate the mean distance of observed object positions
        from the desired trajectory.

        Returns pixels.
        """

        if not self.actual_path:
            return None

        deviations = []

        for actual_point in self.actual_path:

            segment_distances = []

            for index in range(
                len(self.desired_path) - 1
            ):
                distance = (
                    self._point_to_segment_distance(
                        actual_point,
                        self.desired_path[index],
                        self.desired_path[index + 1],
                    )
                )

                segment_distances.append(
                    distance
                )

            if segment_distances:
                deviations.append(
                    min(segment_distances)
                )

        if not deviations:
            return None

        return float(
            np.mean(deviations)
        )

    def get_metrics(self):
        """
        Return the current trajectory-assessment results.
        """

        return {
            "path_length_px":
                self.calculate_path_length(),

            "direct_displacement_px":
                self.calculate_direct_displacement(),

            "movement_efficiency":
                self.calculate_movement_efficiency(),

            "mean_trajectory_deviation_px":
                self.calculate_trajectory_deviation(),

            "final_target_error_px":
                self.calculate_final_target_error(),

            "target_reached":
                self.target_reached(),
        }

    def draw(self, frame):
        """
        Draw the desired trajectory, target region,
        actual trajectory, and current object position.
        """

        # ---------------------------------------------
        # Desired trajectory
        # ---------------------------------------------

        for index in range(
            len(self.desired_path) - 1
        ):
            cv2.line(
                frame,
                self.desired_path[index],
                self.desired_path[index + 1],
                (255, 255, 0),
                3,
            )

        # -------------------------------------------------
        # Draw start position
        # -------------------------------------------------

        start_x, start_y = self.start_point

        cv2.circle(
            frame,
            (start_x, start_y),
            12,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "START",
            (start_x - 30, start_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # -------------------------------------------------
        # Draw desired waypoints
        # -------------------------------------------------

        for index, waypoint in enumerate(
            self.waypoints,
            start=1,
        ):
            waypoint_x, waypoint_y = waypoint

            cv2.circle(
                frame,
                (waypoint_x, waypoint_y),
                8,
                (255, 255, 255),
                -1,
            )

            cv2.putText(
                frame,
                f"W{index}",
                (waypoint_x + 10, waypoint_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # ---------------------------------------------
        # Target / landing region
        # ---------------------------------------------

        cv2.circle(
            frame,
            self.target_center,
            self.target_radius,
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            "TARGET",
            (
                self.target_center[0] - 35,
                self.target_center[1] - self.target_radius - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # ---------------------------------------------
        # Actual observed trajectory
        # ---------------------------------------------

        if len(self.actual_path) >= 2:

            trajectory_points = np.asarray(
                self.actual_path,
                dtype=np.int32,
            )

            cv2.polylines(
                frame,
                [trajectory_points],
                isClosed=False,
                color=(0, 0, 255),
                thickness=3,
            )

        # ---------------------------------------------
        # Current object position
        # ---------------------------------------------

        if self.actual_path:

            current_position = (
                self.actual_path[-1]
            )

            cv2.circle(
                frame,
                current_position,
                8,
                (0, 165, 255),
                -1,
            )


        # -------------------------------------------------
        # Testing-mode status panel
        # -------------------------------------------------

        panel_x = 20
        panel_y = 30

        cv2.putText(
            frame,
            "TRAJECTORY TESTING MODE",
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.attempt_active:
            status_text = "ACTIVE"

        elif self.attempt_completed:
            status_text = "ATTEMPT COMPLETE"

        else:
            status_text = "WAITING"

        cv2.putText(
            frame,
            f"Attempt: {status_text}",
            (panel_x, panel_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if self.active_object_id is not None:

            cv2.putText(
                frame,
                f"Object ID: {self.active_object_id}",
                (panel_x, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )


        if self.actual_path:

            metrics = self.get_metrics()

            efficiency = metrics.get(
                "movement_efficiency"
            )

            deviation = metrics.get(
                "mean_trajectory_deviation_px"
            )

            target_error = metrics.get(
                "final_target_error_px"
            )

            target_reached = metrics.get(
                "target_reached"
            )

            if efficiency is not None:
                efficiency_text = (
                    f"{efficiency:.2f}"
                )
            else:
                efficiency_text = "N/A"

            if deviation is not None:
                deviation_text = (
                    f"{deviation:.1f} px"
                )
            else:
                deviation_text = "N/A"

            if target_error is not None:
                target_error_text = (
                    f"{target_error:.1f} px"
                )
            else:
                target_error_text = "N/A"

            cv2.putText(
                frame,
                f"Efficiency: {efficiency_text}",
                (panel_x, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Path deviation: {deviation_text}",
                (panel_x, panel_y + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Target error: {target_error_text}",
                (panel_x, panel_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            target_text = (
                "YES"
                if target_reached
                else "NO"
            )

            cv2.putText(
                frame,
                f"Target reached: {target_text}",
                (panel_x, panel_y + 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return frame

    