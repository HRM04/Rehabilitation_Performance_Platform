"""Read-only longitudinal dashboard for rehabilitation sessions."""

import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from src.assessment.combined_assessment_summary import load_combined_session

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/trajectory_results"
UNIFIED_DIR = PROJECT_ROOT / "data/processed/unified_sessions"

METRICS = {
    "movement_efficiency": "Movement Efficiency",
    "mean_trajectory_deviation_px": "Mean Trajectory Deviation (px)",
    "final_target_error_px": "Final Target Error (px)",
    "path_length_px": "Path Length (px)",
    "direct_displacement_px": "Direct Displacement (px)",
}


def load_sessions():
    sessions = []

    for path in sorted(DATA_DIR.glob("*_trajectory_results.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                session = json.load(file)

            if isinstance(session, dict) and isinstance(
                session.get("attempts"), list
            ):
                sessions.append(session)

        except (OSError, ValueError, TypeError):
            st.warning(f"Could not read {path.name}.")

    return sessions


def session_datetime(session_id):
    try:
        return datetime.strptime(str(session_id), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def numeric_metric(attempt, metric):
    value = (attempt.get("metrics") or {}).get(metric)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    return None


def summarize_session(session):
    attempts = session.get("attempts", [])

    completed = [
        attempt for attempt in attempts
        if attempt.get("outcome") == "COMPLETED"
    ]

    interrupted = [
        attempt for attempt in attempts
        if attempt.get("outcome") == "INTERRUPTED"
    ]

    row = {
        "Session ID": session.get("session_id"),
        "Unified Assessment": (
            "Export available"
            if (
                UNIFIED_DIR
                / f"{session.get('session_id')}_unified_assessment_v02.json"
            ).is_file()
            else "Raw trajectory only"
        ),
        "Date": session_datetime(session.get("session_id")),
        "Patient": session.get("patient_id"),
        "Object": session.get("object_id"),
        "Trajectory": session.get("trajectory"),
        "Total Attempts": len(attempts),
        "Completed": len(completed),
        "Interrupted": len(interrupted),
        "Target Repetitions": session.get("target_repetitions"),
    }

    for metric in METRICS:
        values = [
            numeric_metric(attempt, metric)
            for attempt in completed
        ]

        values = [value for value in values if value is not None]

        row[metric] = mean(values) if values else None
        row[f"{metric}_n"] = len(values)

    durations = [
        attempt.get("duration_ms")
        for attempt in completed
        if isinstance(attempt.get("duration_ms"), (int, float))
        and not isinstance(attempt.get("duration_ms"), bool)
    ]

    row["Mean Duration (s)"] = (
        mean(durations) / 1000 if durations else None
    )

    return row


st.set_page_config(
    page_title="Clinician Longitudinal Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Clinician Longitudinal Dashboard")
st.caption(
    "Upper-limb rehabilitation | Between-session performance review"
)

sessions = load_sessions()

if not sessions:
    st.warning("No trajectory session recordings were found.")
    st.stop()

patients = sorted({
    session.get("patient_id")
    for session in sessions
    if session.get("patient_id") is not None
})

if not patients:
    st.warning("No patient identifiers were found.")
    st.stop()

selected_patient = st.selectbox("Select patient", patients)

patient_sessions = [
    session for session in sessions
    if session.get("patient_id") == selected_patient
]

objects = sorted({
    session.get("object_id")
    for session in patient_sessions
    if session.get("object_id") is not None
})

trajectories = sorted({
    session.get("trajectory")
    for session in patient_sessions
    if session.get("trajectory") is not None
})

object_col, trajectory_col = st.columns(2)

with object_col:
    selected_object = st.selectbox(
        "Assessment object",
        ["All Objects"] + objects,
    )

with trajectory_col:
    selected_trajectory = st.selectbox(
        "Trajectory approach",
        ["All Trajectories"] + trajectories,
    )

filtered_sessions = [
    session for session in patient_sessions
    if (
        selected_object == "All Objects"
        or session.get("object_id") == selected_object
    )
    and (
        selected_trajectory == "All Trajectories"
        or session.get("trajectory") == selected_trajectory
    )
]

if not filtered_sessions:
    st.info("No sessions match the selected filters.")
    st.stop()

rows = [summarize_session(session) for session in filtered_sessions]

df = pd.DataFrame(rows).sort_values(
    ["Date", "Session ID"],
    na_position="last",
).reset_index(drop=True)

st.subheader("Session Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Recorded Sessions", len(df))
col2.metric("Completed Attempts", int(df["Completed"].sum()))
col3.metric("Interrupted Attempts", int(df["Interrupted"].sum()))
col4.metric("Total Attempts", int(df["Total Attempts"].sum()))

if len(df) < 2:
    st.info(
        "At least two comparable sessions are needed to display "
        "between-session trends."
    )

st.divider()

st.subheader("Session History")

history_columns = [
    "Session ID",
    "Date",
    "Completed",
    "Interrupted",
    "Total Attempts",
    "Target Repetitions",
    "Mean Duration (s)",
]

st.dataframe(
    df[history_columns],
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Session Performance Comparison")

comparison_columns = {
    "Session ID": "Session",
    "movement_efficiency": "Movement Efficiency",
    "mean_trajectory_deviation_px": "Mean Deviation (px)",
    "final_target_error_px": "Final Target Error (px)",
    "path_length_px": "Path Length (px)",
    "Mean Duration (s)": "Mean Duration (s)",
}

comparison_df = df[
    list(comparison_columns.keys())
].rename(columns=comparison_columns)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Movement Efficiency": st.column_config.NumberColumn(
            format="%.3f"
        ),
        "Mean Deviation (px)": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Final Target Error (px)": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Path Length (px)": st.column_config.NumberColumn(
            format="%.2f"
        ),
        "Mean Duration (s)": st.column_config.NumberColumn(
            format="%.2f"
        ),
    },
)

st.caption(
    "Values are averages across completed attempts. "
    "Missing measurements are left blank."
)

st.divider()

st.subheader("Observed Performance Changes")

st.caption(
    "Changes are calculated between the earliest and latest "
    "recorded sessions for the selected patient, object, and "
    "trajectory. They describe task performance, not clinical recovery."
)

change_metrics = {
    "movement_efficiency": "Movement Efficiency",
    "mean_trajectory_deviation_px": "Mean Deviation (px)",
    "final_target_error_px": "Final Target Error (px)",
    "path_length_px": "Path Length (px)",
    "direct_displacement_px": "Direct Displacement (px)",
    "Mean Duration (s)": "Mean Duration (s)",
}

if len(df) < 2:
    st.info(
        "At least two sessions are needed to calculate "
        "between-session changes."
    )
else:
    earlier = df.iloc[0]
    later = df.iloc[-1]

    st.write(
        f"**Earlier session:** {earlier['Session ID']}  |  "
        f"**Later session:** {later['Session ID']}"
    )

    change_rows = []

    for metric, label in change_metrics.items():
        first_value = earlier.get(metric)
        last_value = later.get(metric)

        if pd.notna(first_value) and pd.notna(last_value):
            absolute_change = last_value - first_value

            percentage_change = (
                absolute_change / abs(first_value) * 100
                if first_value != 0
                else None
            )
        else:
            absolute_change = None
            percentage_change = None

        change_rows.append({
            "Measurement": label,
            "Earlier": first_value,
            "Later": last_value,
            "Absolute Change": absolute_change,
            "Change (%)": percentage_change,
        })

    changes_df = pd.DataFrame(change_rows)

    st.dataframe(
        changes_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Earlier": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Later": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Absolute Change": st.column_config.NumberColumn(
                format="%+.3f"
            ),
            "Change (%)": st.column_config.NumberColumn(
                format="%+.1f%%"
            ),
        },
    )

    st.caption(
        "Positive and negative changes are not automatically "
        "classified as improvement or deterioration. "
        "Missing measurements remain unavailable. "
        "Historical reference-path geometry and tracking "
        "conditions have not been fully verified."
    )

st.divider()

st.subheader("Performance Trends")

st.caption(
    "Each point represents the mean across completed attempts "
    "within one session. Interrupted attempts are counted separately."
)

trend_metrics = {
    **METRICS,
    "Mean Duration (s)": "Mean Duration (s)",
}

chart_columns = st.columns(2)

for index, (metric, label) in enumerate(trend_metrics.items()):
    with chart_columns[index % 2]:
        st.markdown(f"**{label}**")

        plot_df = df[
            ["Session ID", "Date", metric]
        ].dropna(subset=[metric]).copy()

        if plot_df.empty:
            st.info("No measurements available.")
            continue

        # Preserve original session positions when values are missing.
        plot_df["Session Order"] = [
            df.index.get_loc(row_index) + 1
            for row_index in plot_df.index
        ]

        fig = px.line(
            plot_df,
            x="Session Order",
            y=metric,
            markers=True,
            hover_data={
                "Session ID": True,
                "Date": True,
                "Session Order": False,
            },
            labels={
                metric: label,
                "Session Order": "Session",
            },
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(df) + 1)),
            ticktext=df["Session ID"].tolist(),
            tickangle=-30,
        )

        fig.update_layout(
            height=320,
            margin=dict(l=15, r=15, t=20, b=25),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        if len(plot_df) == 1:
            st.caption(
                "Only one session has this measurement; "
                "no between-session trend is available."
            )

st.divider()


# -------------------------------------------------------
# Longitudinal grasp assessment
# -------------------------------------------------------

st.subheader("Longitudinal Grasp Assessment")

grasp_rows = []
grasp_event_rows = []
grasp_hand_rows = []
grasp_warnings = []

for recorded_session in filtered_sessions:
    sid = str(recorded_session.get("session_id") or "")

    try:
        combined = load_combined_session(sid)
    except (FileNotFoundError, ValueError, OSError) as error:
        grasp_warnings.append(f"{sid}: {error}")
        continue

    grasp = combined["session_grasp_summary"]
    counts = grasp["appropriateness_counts"]

    grasp_rows.append({
        "Session": sid,
        "Date": session_datetime(sid),
        "Object": recorded_session.get("object_id"),
        "Trajectory": recorded_session.get("trajectory"),
        "Hand Observations": grasp["hand_observations"],
        "Appropriate": counts.get("APPROPRIATE", 0),
        "Inappropriate": counts.get("INAPPROPRIATE", 0),
        "Uncertain": counts.get("UNCERTAIN", 0),
        "Not Assessed": counts.get("NOT_ASSESSED", 0),
    })

    for hand, hand_data in grasp["by_hand"].items():
        hand_counts = hand_data["appropriateness_counts"]
        grasp_hand_rows.append({
            "Session": sid,
            "Detected Hand": hand,
            "Observations": hand_data["hand_observations"],
            "Appropriate": hand_counts.get("APPROPRIATE", 0),
            "Inappropriate": hand_counts.get("INAPPROPRIATE", 0),
            "Uncertain": hand_counts.get("UNCERTAIN", 0),
            "Not Assessed": hand_counts.get("NOT_ASSESSED", 0),
        })

    for event, count in grasp["interaction_event_counts"].items():
        grasp_event_rows.append({
            "Session": sid,
            "Event": event,
            "Observations": count,
        })

for warning in grasp_warnings:
    st.warning(f"Grasp data unavailable — {warning}")

if grasp_rows:
    grasp_df = pd.DataFrame(grasp_rows).sort_values(
        ["Date", "Session"],
        na_position="last",
    )

    st.markdown("**Grasp Observations by Session**")
    st.dataframe(
        grasp_df.drop(columns=["Date"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Grasp Appropriateness Across Sessions**")

    appropriateness_long = grasp_df.melt(
        id_vars=["Session", "Object", "Trajectory"],
        value_vars=[
            "Appropriate",
            "Inappropriate",
            "Uncertain",
            "Not Assessed",
        ],
        var_name="Classification",
        value_name="Observations",
    )

    fig = px.bar(
        appropriateness_long,
        x="Session",
        y="Observations",
        color="Classification",
        barmode="stack",
        hover_data=["Object", "Trajectory"],
    )

    st.plotly_chart(fig, use_container_width=True)

    if grasp_hand_rows:
        st.markdown("**Hand-Specific Observations**")
        st.dataframe(
            pd.DataFrame(grasp_hand_rows),
            use_container_width=True,
            hide_index=True,
        )

    if grasp_event_rows:
        st.markdown("**Recorded Interaction Events**")
        event_df = pd.DataFrame(grasp_event_rows)
        fig = px.bar(
            event_df,
            x="Session",
            y="Observations",
            color="Event",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These are preliminary frame-level grasp and interaction "
        "observations, not independent grasp attempts or validated "
        "clinical recovery scores. Observation counts depend on "
        "recording duration, hand visibility, and tracking quality. "
        "Detected handedness does not establish the affected limb. "
        "Differences across objects or trajectory configurations "
        "must not be interpreted as equivalent rehabilitation tasks."
    )
else:
    st.info("No matching grasp recordings were available.")

st.divider()

st.subheader("Metric Availability")

availability = df[["Session ID"]].copy()

for metric, label in METRICS.items():
    availability[label] = df[f"{metric}_n"].apply(
        lambda count: f"{int(count)} attempts"
    )

availability["Movement Duration"] = df[
    "Mean Duration (s)"
].apply(
    lambda value: "Available" if pd.notna(value) else "Unavailable"
)

st.dataframe(
    availability,
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("Data Quality and Comparability")

quality_rows = []

for session in filtered_sessions:
    session_id = session.get("session_id")
    attempts = session.get("attempts", [])

    completed = [
        attempt for attempt in attempts
        if attempt.get("outcome") == "COMPLETED"
    ]

    trajectory_points = sum(
        len(attempt.get("trajectory_points") or [])
        for attempt in completed
    )

    timestamped_samples = sum(
        len(attempt.get("trajectory_samples") or [])
        for attempt in completed
    )

    durations_available = sum(
        1 for attempt in completed
        if isinstance(attempt.get("duration_ms"), (int, float))
        and not isinstance(attempt.get("duration_ms"), bool)
    )

    pose_path = (
        PROJECT_ROOT
        / "data/raw/pose"
        / f"{session_id}_pose_landmarks.csv"
    )

    configurations = [
        attempt.get("trajectory_config")
        for attempt in completed
    ]

    exact_paths_available = (
        bool(completed)
        and all(
            isinstance(config, dict)
            and isinstance(config.get("desired_path"), list)
            and len(config["desired_path"]) >= 2
            for config in configurations
        )
    )

    session_config = session.get("trajectory_config")

    if not exact_paths_available:
        exact_paths_available = (
            isinstance(session_config, dict)
            and isinstance(session_config.get("desired_path"), list)
            and len(session_config["desired_path"]) >= 2
        )

    quality_rows.append({
        "Session": session_id,
        "Completed Attempts": len(completed),
        "Trajectory Points": trajectory_points,
        "Timestamped Samples": timestamped_samples,
        "Duration Available": (
            f"{durations_available}/{len(completed)}"
        ),
        "Pose CSV": "Available" if pose_path.exists()
        else "Unavailable",
        "Recorded Reference Path": (
            "Available" if exact_paths_available
            else "Unavailable"
        ),
    })

quality_df = pd.DataFrame(quality_rows).sort_values("Session")

st.dataframe(
    quality_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Availability does not establish measurement accuracy. "
    "Pose CSV presence does not mean joint measurements passed "
    "quality screening. Reference-path availability indicates "
    "whether prescribed coordinates were saved, not whether "
    "the task configuration was independently verified."
)

if (
    len(quality_df) >= 2
    and quality_df["Timestamped Samples"].eq(0).any()
    and quality_df["Timestamped Samples"].gt(0).any()
):
    st.warning(
        "These sessions have different tracking-data availability. "
        "Tracking gaps and object ID changes cannot be compared "
        "consistently across all recordings."
    )

if quality_df["Recorded Reference Path"].eq("Unavailable").any():
    st.info(
        "At least one session does not contain its exact prescribed "
        "trajectory coordinates. Comparisons of trajectory deviation "
        "and target error should be interpreted cautiously."
    )

st.caption(
    "Comparisons are descriptive engineering measurements, not "
    "validated clinical recovery scores. Differences may reflect "
    "task execution, tracking quality, or session conditions. "
    "Sessions are grouped by patient, object, and recorded "
    "trajectory label; exact historical task equivalence "
    "has not been established."
)
