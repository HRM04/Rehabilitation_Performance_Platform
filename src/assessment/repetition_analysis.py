def create_repetition_record(
    repetition_number,
    start_timestamp_ms,
    end_timestamp_ms,
    joint_metrics,
    trunk_metrics=None,
):
    """
    Store the measurements for one exercise repetition.

    joint_metrics contains the movement measurements
    calculated for each joint during this repetition.
    """

    return {
        "repetition_number": repetition_number,
        "start_timestamp_ms": start_timestamp_ms,
        "end_timestamp_ms": end_timestamp_ms,
        "duration_ms": end_timestamp_ms - start_timestamp_ms,
        "joint_metrics": joint_metrics,
        "trunk_metrics": trunk_metrics,
    }