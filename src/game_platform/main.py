"""Standalone launcher for the Delivery Dash game prototype."""

from src.game_platform.game_setup import choose_game_config
from src.game_platform.delivery_dash import DeliveryDash


def main():
    config = choose_game_config()

    if config is None:
        print("Game setup cancelled.")
        return

    print("\nStarting Delivery Dash")
    print(f"Object: {config.object_name}")
    print(f"Trajectory: {config.trajectory}")
    print(f"Repetitions: {config.repetitions}")

    game = DeliveryDash(
        target_repetitions=config.repetitions,
        object_name=config.object_name,
        trajectory_preset=config.trajectory,
    )

    game.run()


if __name__ == "__main__":
    main()
