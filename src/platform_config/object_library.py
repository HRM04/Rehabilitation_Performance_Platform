"""
Object library for the unified rehabilitation platform.

Object labels correspond to YOLO11n categories.

Expected grasp mappings are provisional engineering
configurations, not clinically validated classifications.
"""

OBJECT_LIBRARY = {
    "bottle": {
        "display_name": "Bottle",
        "expected_grasps": [
            "cylindrical",
            "whole_hand_power",
        ],
        "status": "configured",
    },
    "cup": {
        "display_name": "Cup",
        "expected_grasps": [
            "handle",
            "cylindrical", "lateral_pinch",
        ],
        "status": "configured",
    },
    "spoon": {
        "display_name": "Spoon",
        "expected_grasps": [
            "handle",
            "multidigit_precision",
            "tripod", "lateral_pinch", "tip_pinch",
        ],
        "status": "configured",
    },
    "fork": {
        "display_name": "Fork",
        "expected_grasps": [
            "tripod",
            "multidigit_precision",
        ],
        "status": "experimental",
    },
    "bowl": {
        "display_name": "Bowl",
        "expected_grasps": [
            "large_palmar",
            "spherical",
        ],
        "status": "experimental",
    },
    "book": {
        "display_name": "Book",
        "expected_grasps": [
            "large_palmar",
            "whole_hand_power",
        ],
        "status": "experimental",
    },
    "cell phone": {
        "display_name": "Cell Phone",
        "expected_grasps": [
            "large_palmar",
            "whole_hand_power",
        ],
        "status": "experimental",
    },
    "remote": {
        "display_name": "Remote Control",
        "expected_grasps": [
            "cylindrical",
            "whole_hand_power",
        ],
        "status": "experimental",
    },
    "toothbrush": {
        "display_name": "Toothbrush",
        "expected_grasps": [
            "tripod",
            "multidigit_precision",
        ],
        "status": "experimental",
    },
    "scissors": {
        "display_name": "Scissors",
        "expected_grasps": [
            "handle",
        ],
        "status": "experimental",
    },
    "sports ball": {
        "display_name": "Sports Ball",
        "expected_grasps": [
            "spherical",
            "large_palmar",
        ],
        "status": "experimental",
    },
    "apple": {
        "display_name": "Apple",
        "expected_grasps": [
            "spherical",
            "large_palmar",
        ],
        "status": "experimental",
    },
    "banana": {
        "display_name": "Banana",
        "expected_grasps": [
            "cylindrical",
            "whole_hand_power",
        ],
        "status": "experimental",
    },
}

def get_object(object_id):
    if object_id not in OBJECT_LIBRARY:
        raise ValueError(
            f"Unsupported object: {object_id}"
        )
    return OBJECT_LIBRARY[object_id].copy()


def list_objects():
    return OBJECT_LIBRARY.copy()
