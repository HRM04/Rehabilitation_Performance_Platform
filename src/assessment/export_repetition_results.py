import csv
import json
from pathlib import Path


def export_repetition_results(
    output_dir,
    session_id,
    repetition_results,
    consistency_results=None,
    deterioration_results=None,
):
    """
    Save repetition-level metrics plus summary analyses.

    Creates:
        1. repetition_metrics.csv
        2. consistency_summary.json
        3. deterioration_summary.json
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Repetition-level CSV
    # -----------------------------------------

    csv_path = output_dir / f"{session_id}_repetition_metrics.csv"

    rows = []

    for rep_index, rep in enumerate(
        repetition_results,
        start=1,
    ):
        if rep is None:
            continue

        base = {
            "repetition_number": rep_index,
            "duration_ms": rep.get("duration_ms"),
            "maximum_trunk_displacement_norm": (
                rep.get("trunk", {})
                .get("maximum_displacement_norm")
            ),
        }

        for joint_name, metrics in rep.get(
            "joints", {}
        ).items():

            for metric_name, value in metrics.items():
                base[
                    f"{joint_name}_{metric_name}"
                ] = value

        rows.append(base)

    if rows:
        fieldnames = sorted({
            key
            for row in rows
            for key in row.keys()
        })

        with csv_path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    # -----------------------------------------
    # Consistency summary
    # -----------------------------------------

    consistency_path = (
        output_dir
        / f"{session_id}_consistency_summary.json"
    )

    with consistency_path.open("w") as file:
        json.dump(
            consistency_results,
            file,
            indent=2,
        )

    # -----------------------------------------
    # Performance-deterioration summary
    # -----------------------------------------

    deterioration_path = (
        output_dir
        / f"{session_id}_deterioration_summary.json"
    )

    with deterioration_path.open("w") as file:
        json.dump(
            deterioration_results,
            file,
            indent=2,
        )

    return {
        "repetition_csv": csv_path,
        "consistency_json": consistency_path,
        "deterioration_json": deterioration_path,
    }