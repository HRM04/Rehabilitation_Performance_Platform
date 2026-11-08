import csv

def create_landmark_logger(landmark_output_path):
    csv_file = open(
        landmark_output_path,
        "w",
        newline=""
    )

    csv_writer = csv.writer(csv_file)



    # Write column headings
    csv_writer.writerow([
        "session_name",
        "frame_number",
        "timestamp_ms",
        "detected_hand_count",
        "hand_label",
        "hand_confidence",
        "landmark_number",
        "landmark_name",
        "x_norm",
        "y_norm",
        "z_rel",
        "x_px",
        "y_px",
        "x_rel",
        "y_rel",
        "x_scaled",
        "y_scaled",
        "thumb_index_distance",
        "predicted_grasp",
        "grasp_confidence"
    ])

    return csv_file, csv_writer


def log_hand_landmarks(
    csv_writer,
    session_name,
    frame_number,
    timestamp_ms,
    num_hands,
    hand_label,
    hand_confidence,
    hand_data,
    hand_landmark_enum,
    thumb_index_distance,
    predicted_grasp,
    grasp_confidence
    
):

    for landmark_number, landmark_name in enumerate(
        hand_landmark_enum
    ):

        coordinates = hand_data[landmark_name.name]

        csv_writer.writerow([
            session_name,
            frame_number,
            timestamp_ms,
            num_hands,
            hand_label,
            hand_confidence,
            landmark_number,
            landmark_name.name,
            coordinates["x_norm"],
            coordinates["y_norm"],
            coordinates["z_rel"],
            coordinates["x_px"],
            coordinates["y_px"],
            coordinates["x_rel"],
            coordinates["y_rel"],
            coordinates["x_scaled"],
            coordinates["y_scaled"],
            thumb_index_distance,
            predicted_grasp,
            grasp_confidence
        ])




def log_no_hand_frame(
    csv_writer,
    session_name,
    frame_number,
    timestamp_ms
):
    """
    Record a frame where no hands were detected.

    Landmark measurements remain empty because
    no valid hand landmarks are available.
    """

    csv_writer.writerow([
        session_name,
        frame_number,
        timestamp_ms,
        0,       # detected_hand_count
        None,    # hand_label
        None,    # hand_confidence
        None,    # landmark_number
        None,    # landmark_name
        None,    # x_norm
        None,    # y_norm
        None,    # z_rel
        None,    # x_px
        None,    # y_px
        None,    # x_rel
        None,    # y_rel
        None,    # x_scaled
        None,    # y_scaled
        None,    # thumb_index_distance
        None,    # predicted_grasp
        None     # grasp_confidence
    ])