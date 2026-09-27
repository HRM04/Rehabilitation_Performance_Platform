def normalize_glove_record(
    raw_record,
    timestamp_ms=None,
    hand=None,
):
    """
    Convert a raw glove reading into the standard glove format
    used by the rehabilitation platform.

    The exact HERO Glove fields will be mapped here once the
    hardware interface and sensor channels are confirmed.
    """

    if raw_record is None:
        return None

    return {
        "timestamp_ms": timestamp_ms,

        "hand": hand,

        "force": {
            "thumb": raw_record.get("thumb_force"),
            "index": raw_record.get("index_force"),
            "middle": raw_record.get("middle_force"),
        },

        "finger_bend": {
            "thumb": raw_record.get("thumb_bend"),
            "index": raw_record.get("index_bend"),
            "middle": raw_record.get("middle_bend"),
            "ring": raw_record.get("ring_bend"),
            "little": raw_record.get("little_bend"),
        },

        "imu": {
            "acceleration": raw_record.get("acceleration"),
            "angular_velocity": raw_record.get("angular_velocity"),
            "orientation": raw_record.get("orientation"),
        },

        "status": {
            "connected": raw_record.get(
                "connected",
                False,
            ),
            "calibrated": raw_record.get(
                "calibrated",
                False,
            ),
        },

        "source": {
            "device": raw_record.get("device"),
            "interface": raw_record.get("interface"),
        },
    }