TASKS = {
    "bottle_transport": {
        "task_name": "Bottle Transport",

        "permitted_objects": [
            "bottle"
        ],

        "expected_grasps": [
            "cylindrical",
            "whole_hand_power"
        ],

        "description": (
            "Reach for the bottle, grasp it, transport it, "
            "and release it at the target location."
        )
    },

    "cup_manipulation": {
        "task_name": "Cup Manipulation",

        "permitted_objects": [
            "cup"
        ],

        "expected_grasps": [
            "handle",
            "cylindrical"
        ],

        "description": (
            "Reach for the cup, grasp it, move it, "
            "and release it."
        )
    },

    "spoon_manipulation": {
        "task_name": "Spoon Manipulation",

        "permitted_objects": [
            "spoon"
        ],

        "expected_grasps": [
            "handle",
            "multidigit_precision",
            "tripod"
        ],

        "description": (
            "Reach for the spoon, grasp it, manipulate it, "
            "and release it."
        )
    }
}



def get_task(task_id):

    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task: {task_id}"
        )

    return TASKS[task_id]