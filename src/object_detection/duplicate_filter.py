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
    # Identify large detections that may combine multiple objects.
    # Require two distinct same-class boxes, each strongly contained
    # within the larger detection.
    ambiguous_large_indices = set()

    def _area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _containment(smaller, larger):
        sx1, sy1, sx2, sy2 = smaller
        lx1, ly1, lx2, ly2 = larger

        intersection = (
            max(0, min(sx2, lx2) - max(sx1, lx1))
            * max(0, min(sy2, ly2) - max(sy1, ly1))
        )

        smaller_area = _area(smaller)
        return intersection / smaller_area if smaller_area else 0.0

    for large_index, large in enumerate(detected_objects):
        large_box = large["bounding_box"]
        large_area = _area(large_box)

        contained = []

        for small_index, small in enumerate(detected_objects):
            if small_index == large_index:
                continue

            if small["class_name"] != large["class_name"]:
                continue

            small_box = small["bounding_box"]

            if _area(small_box) >= large_area:
                continue

            if _containment(small_box, large_box) >= 0.95:
                contained.append(small_index)

        # Two spatially distinct contained boxes suggest that the
        # larger box may be an ambiguous merged detection.
        for i, first_index in enumerate(contained):
            for second_index in contained[i + 1:]:
                first_box = detected_objects[first_index]["bounding_box"]
                second_box = detected_objects[second_index]["bounding_box"]

                if calculate_iou(first_box, second_box) < 0.10:
                    ambiguous_large_indices.add(large_index)
                    break

            if large_index in ambiguous_large_indices:
                break

    eligible_objects = [
        obj for index, obj in enumerate(detected_objects)
        if index not in ambiguous_large_indices
    ]

    sorted_objects = sorted(
        eligible_objects,
        key=lambda obj: obj["confidence"],
        reverse=True
    )

    filtered_objects = []
    removed_objects = [
        obj for index, obj in enumerate(detected_objects)
        if index in ambiguous_large_indices
    ]

    for candidate in sorted_objects:
        duplicate_of = None

        for retained in filtered_objects:
            if candidate["class_name"] != retained["class_name"]:
                continue

            overlap = calculate_iou(
                candidate["bounding_box"],
                retained["bounding_box"]
            )

            # Calculate how much of the smaller box is covered.
            # This detects nested boxes that may have modest IoU.
            box_a = candidate["bounding_box"]
            box_b = retained["bounding_box"]

            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b

            intersection_width = max(
                0, min(ax2, bx2) - max(ax1, bx1)
            )
            intersection_height = max(
                0, min(ay2, by2) - max(ay1, by1)
            )

            intersection_area = (
                intersection_width * intersection_height
            )

            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

            smaller_area = min(area_a, area_b)

            containment = (
                intersection_area / smaller_area
                if smaller_area > 0
                else 0.0
            )

            if overlap >= iou_threshold or containment >= 0.95:
                duplicate_of = retained
                break

        if duplicate_of is None:
            filtered_objects.append(candidate)
        else:
            removed_objects.append(candidate)

    return filtered_objects, removed_objects