def calculate_iou(box_a, box_b):
    """
    Calculate intersection over union (IoU) for two bounding boxes.

    IoU = overlapping area / combined area.
    Returns a value between 0 and 1.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(
        0,
        min(ax2, bx2) - max(ax1, bx1)
    )
    intersection_height = max(
        0,
        min(ay2, by2) - max(ay1, by1)
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def filter_duplicate_objects(
    detected_objects,
    iou_threshold=0.90
):
    """
    Keep the higher-confidence detection when two detections
    of the same class have almost identical bounding boxes.

    Returns:
        filtered_objects: detections retained
        removed_objects: detections flagged as probable duplicates
    """
    sorted_objects = sorted(
        detected_objects,
        key=lambda obj: obj["confidence"],
        reverse=True
    )

    filtered_objects = []
    removed_objects = []

    for candidate in sorted_objects:
        duplicate_of = None

        for retained in filtered_objects:
            if candidate["class_name"] != retained["class_name"]:
                continue

            overlap = calculate_iou(
                candidate["bounding_box"],
                retained["bounding_box"]
            )

            if overlap >= iou_threshold:
                duplicate_of = retained
                break

        if duplicate_of is None:
            filtered_objects.append(candidate)
        else:
            removed_objects.append(candidate)

    return filtered_objects, removed_objects