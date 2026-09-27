import csv
from pathlib import Path

from src.assessment.repetition_analysis import segment_repetitions


interaction_path = Path(
    "data/raw/interactions/"
    "20260925_203859_interaction_data.csv"
)

with interaction_path.open(newline="") as file:
    rows = list(csv.DictReader(file))

repetitions = segment_repetitions(
    rows,
    minimum_duration_ms=1000,
)
print(f"Recording: {interaction_path.name}")
print(f"Detected repetitions: {len(repetitions)}")

for repetition in repetitions:
    print(
        f"Rep {repetition['repetition_number']} | "
        f"Frames {repetition['start_frame']}–{repetition['end_frame']} | "
        f"Duration {repetition['duration_ms']} ms | "
        f"Movement {repetition['total_object_movement']:.2f}"
    )