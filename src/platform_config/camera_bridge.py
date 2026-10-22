"""
Compatibility bridge between the unified platform
and the existing camera capture pipeline.
"""

import json
import os

from src.platform_config.session_config import SessionConfig


def load_camera_session():
    config_path = os.environ.get(
        "REHAB_SESSION_CONFIG"
    )

    if config_path:
        with open(config_path, "r") as file:
            data = json.load(file)

        config = SessionConfig(**data)
    else:
        config = SessionConfig()

    config.validate()

    return config, config.get_task_details()
