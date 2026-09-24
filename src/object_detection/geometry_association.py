from src.object_detection.duplicate_filter import calculate_iou
from src.object_detection.object_geometry import calculate_object_geometry

def associate_object_geometry(
    tracked_objects,
    segmented_objects,
    min_iou=0.30,
):
    """
    Associate tracked-object detections with segmentation results.

    Matching requires:
      - the same object class;
      - sufficient bounding-box overlap;
      - a one-to-one assignment between detections.

    A geometric match does not independently confirm
    physical-object identity.
    """

    candidates = []

    for tracked_index, tracked in enumerate(tracked_objects):
        for segment_index, segmented in enumerate(segmented_objects):
            if tracked["class_name"] != segmented["class_name"]:
                continue

            overlap = calculate_iou(
                tracked["bounding_box"],
                segmented["bounding_box"],
            )

            if overlap >= min_iou:
                candidates.append(
                    (overlap, tracked_index, segment_index)
                )

    # Evaluate the strongest overlaps first.
    candidates.sort(key=lambda item: item[0], reverse=True)

    assigned_tracked = set()
    assigned_segments = set()
    associations = []

    for overlap, tracked_index, segment_index in candidates:
        if tracked_index in assigned_tracked:
            continue

        if segment_index in assigned_segments:
            continue

        associations.append({
            "object_id": tracked_objects[tracked_index].get("object_id"),
            "tracked_object": tracked_objects[tracked_index],
            "segmented_object": segmented_objects[segment_index],
            "association_iou": overlap,
        })

        assigned_tracked.add(tracked_index)
        assigned_segments.add(segment_index)

    return associations



def attach_object_geometry(
    tracked_objects,
    segmented_objects,
    min_iou=0.30,
):
    """
    Attach segmentation-derived geometry to matched tracked objects.

    Returns copies of the tracked-object dictionaries.
    Objects without a reliable segmentation match retain
    a geometry value of None.
    """

    updated_objects = [
        {
            **obj,
            "geometry": None,
        }
        for obj in tracked_objects
    ]

    associations = associate_object_geometry(
        tracked_objects,
        segmented_objects,
        min_iou=min_iou,
    )

    for association in associations:
        tracked_object = association["tracked_object"]
        segmented_object = association["segmented_object"]

        geometry = calculate_object_geometry(
            segmented_object["mask"]
        )

        if geometry is None:
            continue

        # Find the corresponding tracked object using its
        # application-level ID.
        object_id = tracked_object.get("object_id")

        for updated_object in updated_objects:
            if updated_object.get("object_id") == object_id:
                updated_object["geometry"] = {
                    "orientation_deg": geometry["orientation_deg"],
                    "orientation_reliable": geometry[
                        "orientation_reliable"
                    ],
                    "major_axis_px": geometry["major_axis_px"],
                    "minor_axis_px": geometry["minor_axis_px"],
                    "aspect_ratio": geometry["aspect_ratio"],
                    "association_iou": association["association_iou"],
                }

                break

    return updated_objects