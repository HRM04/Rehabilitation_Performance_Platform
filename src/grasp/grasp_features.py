def calculate_thumb_index_distance(hand_data):
    thumb_tip = hand_data["THUMB_TIP"]
    index_tip = hand_data["INDEX_FINGER_TIP"]

    x_distance = (
        thumb_tip["x_scaled"] - index_tip["x_scaled"]
    )

    y_distance = (
        thumb_tip["y_scaled"] - index_tip["y_scaled"]
    )

    thumb_index_distance = (
        x_distance ** 2 + y_distance ** 2
    ) ** 0.5

    return thumb_index_distance