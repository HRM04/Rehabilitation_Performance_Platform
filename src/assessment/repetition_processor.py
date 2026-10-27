from src.assessment.repetition_metrics import (
    calculate_repetition_metrics,
)

from src.assessment.functional_performance import (
    assess_functional_performance,
)

from src.assessment.repetition_assessment import (
    create_repetition_assessment,
)


def process_repetition(
    repetition,
    pose_frames,
    trajectory_assessment=None,
    grasp_assessment=None,
    reference_posture=None,
):
    """
    Process one completed rehabilitation repetition.

    The repetition's timestamp window is used to align
    available assessment modalities.

    Combines:
        - repetition timing/object information
        - pose-derived movement metrics
        - trajectory metrics
        - functional performance
        - grasp assessment

    Missing modalities remain None rather than being
    interpreted as zero or normal performance.
    """

    if repetition is None:
        return None

    # -------------------------------------------------
    # 1. Obtain repetition time window
    # -------------------------------------------------

    start_timestamp_ms = repetition.get(
        "start_timestamp_ms"
    )

    end_timestamp_ms = repetition.get(
        "end_timestamp_ms"
    )

    object_id = repetition.get("object_id")

    # -------------------------------------------------
    # 2. Calculate pose-derived movement metrics
    # -------------------------------------------------

    movement_metrics = None

    if (
        pose_frames is not None
        and start_timestamp_ms is not None
        and end_timestamp_ms is not None
    ):
        movement_metrics = calculate_repetition_metrics(
            pose_frames=pose_frames,
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            reference_posture=reference_posture,
        )

    # -------------------------------------------------
    # 3. Calculate trajectory metrics
    # -------------------------------------------------

    trajectory_metrics = None

    if (
        trajectory_assessment is not None
        and start_timestamp_ms is not None
        and end_timestamp_ms is not None
    ):
        samples_in_window = [
            sample
            for sample in trajectory_assessment.trajectory_samples
            if (
                sample.get("timestamp_ms") is not None
                and start_timestamp_ms
                <= sample["timestamp_ms"]
                <= end_timestamp_ms
            )
        ]

        observed_object_ids = {
            sample.get("object_id")
            for sample in samples_in_window
            if sample.get("object_id") is not None
        }

        safe_object_id = object_id

        if object_id is None and len(observed_object_ids) == 1:
            safe_object_id = next(iter(observed_object_ids))

        if object_id is not None or len(observed_object_ids) == 1:
            trajectory_metrics = (
                trajectory_assessment.calculate_metrics_for_window(
                    start_timestamp_ms=start_timestamp_ms,
                    end_timestamp_ms=end_timestamp_ms,
                    object_id=safe_object_id,
                )
            )

    # -------------------------------------------------
    # 4. Assess functional performance
    # -------------------------------------------------

    functional_performance = (
        assess_functional_performance(
            repetition=repetition,
            trajectory_metrics=trajectory_metrics,
        )
    )

    # -------------------------------------------------
    # 5. Create unified repetition assessment
    # -------------------------------------------------

    assessment = create_repetition_assessment(
        repetition=repetition,
        movement_metrics=movement_metrics,
        functional_performance=functional_performance,
        grasp_assessment=grasp_assessment,
        trajectory_metrics=trajectory_metrics,
    )

    return assessment