import json
from pathlib import Path

from src.assessment.clinician_summary import (
    create_clinician_summary,
)
from src.assessment.rehab_state import (
    create_rehab_state,
    update_rehab_state,
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