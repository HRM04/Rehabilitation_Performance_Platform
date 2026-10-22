"""Trajectory visualizations for the clinician dashboard."""

import plotly.graph_objects as go
import streamlit as st


def get_reference_path(preset, width, height):
    """Reconstruct a reference path using the current game presets."""

    def point(x, y):
        return [int(x * width), int(y * height)]

    presets = {
        "horizontal": [
            point(0.20, 0.55),
            point(0.80, 0.55),
        ],
        "vertical": [
            point(0.50, 0.80),
            point(0.50, 0.20),
        ],
        "diagonal": [
            point(0.20, 0.80),
            point(0.80, 0.20),
        ],
        "curved": [
            point(0.20, 0.70),
            point(0.30, 0.48),
            point(0.40, 0.35),
            point(0.50, 0.30),
            point(0.60, 0.35),
            point(0.70, 0.48),
            point(0.80, 0.70),
        ],
        "s_shaped": [
            point(0.20, 0.75),
            point(0.35, 0.75),
            point(0.40, 0.50),
            point(0.50, 0.50),
            point(0.60, 0.50),
            point(0.65, 0.25),
            point(0.80, 0.25),
        ],
    }

    return presets.get(str(preset).lower())


def render_trajectory_visualization(
    attempt,
    preset,
    frame_width=1280,
    frame_height=720,
    session_trajectory_config=None,
):
    samples = attempt.get("trajectory_samples") or []

    points = [
        sample["center"]
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("center"), (list, tuple))
        and len(sample["center"]) == 2
    ]

    if not points:
        points = attempt.get("trajectory_points") or []

    if not points:
        st.info("No recorded trajectory points are available.")
        return

    # Prefer the exact prescribed path recorded during the attempt.
    attempt_config = attempt.get("trajectory_config") or {}
    session_config = session_trajectory_config or {}

    reference = None
    reference_source = None

    for config, source_name in (
        (attempt_config, "Recorded attempt configuration"),
        (session_config, "Recorded session configuration"),
    ):
        candidate = config.get("desired_path")

        if (
            isinstance(candidate, list)
            and len(candidate) >= 2
            and all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
                for point in candidate
            )
        ):
            reference = candidate
            reference_source = source_name
            break

    if reference is None:
        reference = get_reference_path(
            preset, frame_width, frame_height
        )
        reference_source = (
            "Reconstructed from preset using assumed frame dimensions"
        )

    figure = go.Figure()

    if reference:
        figure.add_trace(go.Scatter(
            x=[point[0] for point in reference],
            y=[point[1] for point in reference],
            mode="lines+markers",
            name="Reconstructed reference",
            line=dict(color="royalblue", width=3, dash="dot"),
        ))

    # Preliminary tracking-quality review.
    gap_threshold_ms = 300
    valid_tracking_samples = [
        sample for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("timestamp_ms"), (int, float))
        and isinstance(sample.get("center"), (list, tuple))
        and len(sample["center"]) == 2
    ]

    gap_events = []
    id_change_events = []
    plotted_x = []
    plotted_y = []
    hover_details = []

    if valid_tracking_samples:
        first_timestamp = valid_tracking_samples[0]["timestamp_ms"]

        for index, sample in enumerate(valid_tracking_samples):
            timestamp = sample["timestamp_ms"]
            center = sample["center"]
            object_id = sample.get("object_id")

            if index > 0:
                previous = valid_tracking_samples[index - 1]
                interval = timestamp - previous["timestamp_ms"]

                if interval > gap_threshold_ms:
                    gap_events.append((sample, interval))

                if object_id != previous.get("object_id"):
                    id_change_events.append((
                        sample,
                        previous.get("object_id"),
                        object_id,
                    ))

                if (
                    interval > gap_threshold_ms
                    or object_id != previous.get("object_id")
                ):
                    plotted_x.append(None)
                    plotted_y.append(None)
                    hover_details.append(None)

            plotted_x.append(center[0])
            plotted_y.append(center[1])
            hover_details.append(
                f"Time: {(timestamp - first_timestamp) / 1000:.2f} s"
                f"<br>Timestamp: {timestamp} ms"
                f"<br>Object ID: {object_id}"
            )

        figure.add_trace(go.Scatter(
            x=plotted_x,
            y=plotted_y,
            mode="lines+markers",
            name="Recorded movement",
            line=dict(color="crimson", width=3),
            marker=dict(size=5),
            text=hover_details,
            hovertemplate=(
                "X: %{x:.0f} px"
                "<br>Y: %{y:.0f} px"
                "<br>%{text}<extra></extra>"
            ),
            connectgaps=False,
        ))

        if gap_events:
            figure.add_trace(go.Scatter(
                x=[event[0]["center"][0] for event in gap_events],
                y=[event[0]["center"][1] for event in gap_events],
                mode="markers",
                name="Observation gap",
                marker=dict(
                    color="darkorange",
                    size=13,
                    symbol="diamond",
                ),
                text=[
                    f"Gap: {interval:.0f} ms"
                    for _, interval in gap_events
                ],
                hovertemplate="%{text}<extra></extra>",
            ))

        if id_change_events:
            figure.add_trace(go.Scatter(
                x=[event[0]["center"][0] for event in id_change_events],
                y=[event[0]["center"][1] for event in id_change_events],
                mode="markers",
                name="Object ID change",
                marker=dict(
                    color="purple",
                    size=15,
                    symbol="x",
                ),
                text=[
                    f"ID: {old_id} → {new_id}"
                    for _, old_id, new_id in id_change_events
                ],
                hovertemplate="%{text}<extra></extra>",
            ))

    else:
        figure.add_trace(go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            name="Recorded movement",
            line=dict(color="crimson", width=3),
        ))

    figure.add_trace(go.Scatter(
        x=[points[0][0]],
        y=[points[0][1]],
        mode="markers",
        name="Recorded start",
        marker=dict(color="orange", size=13),
    ))

    figure.add_trace(go.Scatter(
        x=[points[-1][0]],
        y=[points[-1][1]],
        mode="markers",
        name="Recorded end",
        marker=dict(color="darkred", size=13),
    ))

    if reference:
        figure.add_trace(go.Scatter(
            x=[reference[-1][0]],
            y=[reference[-1][1]],
            mode="markers",
            name="Reference target",
            marker=dict(
                color="green",
                size=17,
                symbol="circle-open",
                line=dict(width=3),
            ),
        ))

    figure.update_layout(
        title="Recorded Movement vs Reference Trajectory",
        xaxis_title="Horizontal position (px)",
        yaxis_title="Vertical position (px)",
        height=520,
        legend=dict(orientation="h", y=-0.2),
    )

    figure.update_xaxes(
        range=[0, frame_width],
        constrain="domain",
    )
    figure.update_yaxes(
        range=[frame_height, 0],
        scaleanchor="x",
        scaleratio=1,
    )

    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        f"Reference path source: {reference_source}. "
        "All displayed coordinates are image-plane pixels. "
        "Measurements are preliminary engineering metrics."
    )

    if valid_tracking_samples:
        st.caption(
            f"Tracking review: {len(valid_tracking_samples)} observations | "
            f"{len(gap_events)} intervals over {gap_threshold_ms} ms | "
            f"{len(id_change_events)} object ID changes. "
            "Flagged intervals indicate missing observations or tracking "
            "identity changes, not confirmed patient movement errors."
        )

    valid_samples = [
        sample for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("timestamp_ms"), (int, float))
        and isinstance(sample.get("center"), (list, tuple))
        and len(sample["center"]) == 2
    ]

    if len(valid_samples) >= 2:
        start_time = valid_samples[0]["timestamp_ms"]

        timeline = go.Figure()

        # Break plotted lines across observation gaps and
        # object ID transitions. Do not interpolate missing data.
        time_values = []
        horizontal_values = []
        vertical_values = []

        for index, sample in enumerate(valid_samples):
            time_s = (
                sample["timestamp_ms"] - start_time
            ) / 1000

            if index > 0:
                previous = valid_samples[index - 1]
                interval = (
                    sample["timestamp_ms"]
                    - previous["timestamp_ms"]
                )

                if (
                    interval > gap_threshold_ms
                    or sample.get("object_id")
                    != previous.get("object_id")
                ):
                    time_values.append(None)
                    horizontal_values.append(None)
                    vertical_values.append(None)

            time_values.append(time_s)
            horizontal_values.append(sample["center"][0])
            vertical_values.append(sample["center"][1])

        timeline.add_trace(go.Scatter(
            x=time_values,
            y=horizontal_values,
            mode="lines+markers",
            name="Horizontal position",
            connectgaps=False,
        ))

        timeline.add_trace(go.Scatter(
            x=time_values,
            y=vertical_values,
            mode="lines+markers",
            name="Vertical position",
            connectgaps=False,
        ))

        # Highlight the first observation after each long gap.
        if gap_events:
            timeline.add_trace(go.Scatter(
                x=[
                    (sample["timestamp_ms"] - start_time) / 1000
                    for sample, _ in gap_events
                ],
                y=[
                    sample["center"][0]
                    for sample, _ in gap_events
                ],
                mode="markers",
                name="Observation gap",
                marker=dict(
                    color="darkorange",
                    size=12,
                    symbol="diamond",
                ),
                text=[
                    f"Observation interval: {interval:.0f} ms"
                    for _, interval in gap_events
                ],
                hovertemplate=(
                    "Time: %{x:.2f} s"
                    "<br>%{text}<extra></extra>"
                ),
            ))

        # Highlight changes in tracked object identity.
        if id_change_events:
            timeline.add_trace(go.Scatter(
                x=[
                    (sample["timestamp_ms"] - start_time) / 1000
                    for sample, _, _ in id_change_events
                ],
                y=[
                    sample["center"][0]
                    for sample, _, _ in id_change_events
                ],
                mode="markers",
                name="Object ID change",
                marker=dict(
                    color="purple",
                    size=14,
                    symbol="x",
                ),
                text=[
                    f"Object ID: {old_id} → {new_id}"
                    for _, old_id, new_id in id_change_events
                ],
                hovertemplate=(
                    "Time: %{x:.2f} s"
                    "<br>%{text}<extra></extra>"
                ),
            ))

        timeline.update_layout(
            title="Object Position Over Time",
            xaxis_title="Time since first observation (s)",
            yaxis_title="Position (px)",
            height=350,
        )

        st.plotly_chart(timeline, use_container_width=True)
