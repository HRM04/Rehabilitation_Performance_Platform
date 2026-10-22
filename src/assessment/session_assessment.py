from src.assessment.repetition_analysis import (
    segment_repetitions,
)

from src.assessment.repetition_processor import (
    process_repetition,
)

from src.assessment.consistency_analysis import (
    calculate_repetition_consistency,
)

from src.assessment.performance_deterioration import (
    calculate_performance_deterioration,
)

from src.assessment.baseline_comparison import (
    compare_repetition_to_baseline,
)

from src.assessment.feedback_generator import (
    generate_feedback,
)

def assess_session(
    interaction_rows,
    pose_frames,
    trajectory_assessment=None,
    grasp_assessments=None,
    baseline=None,
):
    """
    Process all detected repetitions from one rehabilitation session.

    Returns:
        repetitions:
            Raw detected repetition records.

        repetition_assessments:
            Unified assessment record for each repetition.

        movement_consistency:
            Preliminary comparison of pose-derived movement
            metrics across repetitions.
    """

    # -------------------------------------------------
    # 1. Detect repetitions
    # -------------------------------------------------

    repetitions = segment_repetitions(
        interaction_rows
    )

    repetition_assessments = []
    movement_results = []

    # -------------------------------------------------
    # 2. Process each detected repetition
    # -------------------------------------------------

    for repetition in repetitions:

        repetition_number = repetition.get(
            "repetition_number"
        )

        # ---------------------------------------------
        # Find matching grasp assessment, if available
        # ---------------------------------------------

        grasp_assessment = None

        if grasp_assessments is not None:
            grasp_assessment = (
                grasp_assessments.get(
                    repetition_number
                )
            )

        # ---------------------------------------------
        # Create unified repetition assessment
        # ---------------------------------------------

        assessment = process_repetition(
            repetition=repetition,
            pose_frames=pose_frames,
            trajectory_assessment=trajectory_assessment,
            grasp_assessment=grasp_assessment,
        )

        if assessment is None:
            continue

        repetition_assessments.append(
            assessment
        )

        # ---------------------------------------------
        # Compare this repetition with the patient's
        # baseline and generate preliminary feedback.
        # ---------------------------------------------

        baseline_comparison = None
        feedback = []

        if baseline is not None:
            baseline_comparison = (
                compare_repetition_to_baseline(
                    repetition_assessment=assessment,
                    baseline=baseline,
                )
            )

            feedback = generate_feedback(
                baseline_comparison
            )

        assessment["baseline_comparison"] = (
            baseline_comparison
        )

        assessment["feedback"] = feedback

        # Keep movement results separately because the
        # existing consistency module currently expects
        # movement-metric dictionaries rather than full
        # repetition assessments.
        movement_metrics = assessment.get(
            "movement_quality"
        )

        if movement_metrics is not None:
            movement_results.append(
                movement_metrics
            )

    # -------------------------------------------------
    # 3. Compare movement across repetitions
    # -------------------------------------------------

    movement_consistency = None

    if len(movement_results) >= 2:
        movement_consistency = (
            calculate_repetition_consistency(
                movement_results
            )
        )



    # -------------------------------------------------
    # 4. Assess change across repetitions
    # -------------------------------------------------

    performance_deterioration = None

    if len(movement_results) >= 2:
        performance_deterioration = (
            calculate_performance_deterioration(
                movement_results
            )
        )

    # -------------------------------------------------
    # 5. Return session assessment
    # -------------------------------------------------

    return {
        "repetitions": repetitions,

        "repetition_assessments":
            repetition_assessments,

        "movement_consistency":
            movement_consistency,

        "performance_deterioration":
            performance_deterioration,

        "baseline_used":
            baseline,
    }