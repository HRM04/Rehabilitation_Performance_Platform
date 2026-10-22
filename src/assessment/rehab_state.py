from copy import deepcopy


def create_rehab_state(
    patient_id=None,
):
    """
    Create the persistent rehabilitation-state structure.

    The state stores the latest known information about
    rehabilitation performance without assuming that all
    sensing modalities are always available.
    """

    return {
        "patient_id": patient_id,

        "current_session": None,

        "latest_repetition": None,

        "latest_grasp_assessment": None,

        "latest_movement_quality": None,

        "latest_functional_performance": None,

        "latest_trajectory": None,

        "latest_compensation": None,

        "consistency": None,

        "performance_deterioration": None,

        "baseline": None,

        "feedback": [],

        "glove": {
            "available": False,
            "latest_record": None,
        },

        "history": {
            "sessions": [],
            "repetitions": [],
        },
    }


def update_rehab_state(
    state,
    repetition_assessment=None,
    session_summary=None,
    baseline=None,
    consistency=None,
    deterioration=None,
    feedback=None,
    glove_record=None,
):
    """
    Update the rehabilitation state with newly available data.

    Missing inputs do not overwrite existing information.
    """

    updated_state = deepcopy(state)

    if repetition_assessment is not None:

        updated_state["latest_repetition"] = (
            repetition_assessment
        )

        updated_state["latest_grasp_assessment"] = (
            repetition_assessment.get(
                "grasp_assessment"
            )
        )

        updated_state["latest_movement_quality"] = (
            repetition_assessment.get(
                "movement_quality"
            )
        )

        updated_state["latest_functional_performance"] = (
            repetition_assessment.get(
                "functional_performance"
            )
        )

        updated_state["latest_trajectory"] = (
            repetition_assessment.get(
                "trajectory"
            )
        )

        updated_state["latest_compensation"] = (
            repetition_assessment.get(
                "compensation"
            )
        )

        updated_state["history"][
            "repetitions"
        ].append(repetition_assessment)

    if session_summary is not None:

        updated_state["current_session"] = (
            session_summary
        )

        updated_state["history"][
            "sessions"
        ].append(session_summary)

    if baseline is not None:
        updated_state["baseline"] = baseline

    if consistency is not None:
        updated_state["consistency"] = consistency

    if deterioration is not None:
        updated_state[
            "performance_deterioration"
        ] = deterioration

    if feedback is not None:
        updated_state["feedback"] = feedback

    if glove_record is not None:

        updated_state["glove"][
            "available"
        ] = True

        updated_state["glove"][
            "latest_record"
        ] = glove_record

    return updated_state