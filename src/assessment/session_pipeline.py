import json
from pathlib import Path

from src.assessment.clinician_summary import (
    create_clinician_summary,
)
from src.assessment.rehab_state import (
    create_rehab_state,
    update_rehab_state,
)
from src.assessment.session_assessment import (
    assess_session,
)
from src.assessment.personalization import (
    create_patient_baseline,
)

def build_session_state(
    session_id,
    task_id,
    repetition_assessments,
    baseline=None,
    consistency=None,
    deterioration=None,
    feedback=None,
    patient_id=None,
    glove_record=None,
):
    """
    Build the integrated rehabilitation state for one session.
    """

    clinician_summary = create_clinician_summary(
        session_id=session_id,
        task_id=task_id,
        repetition_assessments=repetition_assessments,
        consistency_results=consistency,
        deterioration_results=deterioration,
        feedback=feedback,
    )

    state = create_rehab_state(
        patient_id=patient_id
    )

    for repetition in repetition_assessments:

        if repetition is None:
            continue

        state = update_rehab_state(
            state,
            repetition_assessment=repetition,
        )

    state = update_rehab_state(
        state,
        session_summary=clinician_summary,
        baseline=baseline,
        consistency=consistency,
        deterioration=deterioration,
        feedback=feedback,
        glove_record=glove_record,
    )

    return state

def process_session(
    session_id,
    task_id,
    interaction_rows,
    pose_frames,
    trajectory_assessment=None,
    grasp_assessments=None,
    baseline=None,
    feedback=None,
    patient_id=None,
    glove_record=None,
):
    """
    Run the complete assessment pipeline for one
    rehabilitation session.

    This function:
        1. Detects repetitions.
        2. Creates repetition-level assessments.
        3. Calculates across-repetition consistency.
        4. Calculates observed performance change.
        5. Builds the higher-level rehabilitation state.

    Missing modalities remain None.
    """

    # -------------------------------------------------
    # 1. Assess the recorded session
    # -------------------------------------------------

    session_assessment = assess_session(
        interaction_rows=interaction_rows,
        pose_frames=pose_frames,
        trajectory_assessment=trajectory_assessment,
        grasp_assessments=grasp_assessments,
        baseline=baseline,
    )

    # -------------------------------------------------
    # Establish a candidate baseline when no previous
    # patient-specific baseline is available.
    # -------------------------------------------------

    candidate_baseline = None

    if baseline is None:

        repetition_assessments = (
            session_assessment.get(
                "repetition_assessments",
                []
            )
        )

        if repetition_assessments:
            candidate_baseline = (
                create_patient_baseline(
                    repetition_assessments
                )
            )

    # -------------------------------------------------
    # 2. Extract assessment results
    # -------------------------------------------------

    repetition_assessments = (
        session_assessment.get(
            "repetition_assessments",
            [],
        )
    )

    # -------------------------------------------------
    # Collect feedback generated for each repetition
    # -------------------------------------------------

    session_feedback = []

    for repetition in repetition_assessments:

        if repetition is None:
            continue

        repetition_feedback = repetition.get(
            "feedback"
        )

        if not repetition_feedback:
            continue

        session_feedback.append({
            "repetition_number":
                repetition.get(
                    "repetition_number"
                ),

            "feedback":
                repetition_feedback,
        })

    if feedback is not None:
        session_feedback.append({
            "repetition_number": None,
            "feedback": feedback,
        })

    consistency = session_assessment.get(
        "movement_consistency"
    )

    deterioration = session_assessment.get(
        "performance_deterioration"
    )

    # -------------------------------------------------
    # 3. Build integrated rehabilitation state
    # -------------------------------------------------

    state = build_session_state(
        session_id=session_id,
        task_id=task_id,
        repetition_assessments=
            repetition_assessments,
        baseline=baseline,
        consistency=consistency,
        deterioration=deterioration,
        feedback=session_feedback,
        patient_id=patient_id,
        glove_record=glove_record,
    )

    # -------------------------------------------------
    # 4. Return both levels
    # -------------------------------------------------

    return {
        "session_assessment":
            session_assessment,

        "rehab_state":
            state,

        "candidate_baseline":
            candidate_baseline,
    }

def save_session_state(
    state,
    session_id,
    output_dir="data/processed/session_state",
):
    """
    Save the integrated rehabilitation state as JSON.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{session_id}_rehab_state.json"
    )

    with output_path.open("w") as file:
        json.dump(
            state,
            file,
            indent=2,
        )

    return output_path