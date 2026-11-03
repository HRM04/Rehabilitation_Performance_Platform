"""Preliminary grasp observations and reliability assessment.

This module does not produce a clinically validated quality score.
"""

from collections import Counter
import math


UNKNOWN_TYPES = {"", "UNKNOWN", "AMBIGUOUS", "NONE", "NOT_ASSESSED"}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def assess_grasp_quality(
    observations,
    start_timestamp_ms=None,
    end_timestamp_ms=None,
    min_coverage=0.70,
    interruption_threshold_ms=200,
    min_classified_observations=3,
    min_classification_coverage=0.70,
    require_object_association=False,
    max_gap_ms=500,
):
    """Describe grasp observations without claiming clinical validity.

    A reliable association requires explicit association_reliable=True
    when require_object_association is enabled.

    The legacy quality_score field is retained for compatibility but
    is intentionally None until a validated scoring method exists.
    """

    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1")
    if not 0 <= min_classification_coverage <= 1:
        raise ValueError("min_classification_coverage must be between 0 and 1")
    if interruption_threshold_ms < 0:
        raise ValueError("interruption_threshold_ms must be nonnegative")
    if min_classified_observations < 1:
        raise ValueError("min_classified_observations must be positive")

    cleaned = []

    for observation in observations:
        try:
            timestamp = float(observation["timestamp_ms"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if not math.isfinite(timestamp) or timestamp < 0:
            continue

        if start_timestamp_ms is not None and timestamp < start_timestamp_ms:
            continue
        if end_timestamp_ms is not None and timestamp > end_timestamp_ms:
            continue

        detected = _parse_bool(observation.get("grasp_detected"))

        associated = _parse_bool(
            observation.get("association_reliable")
        )

        # An explicitly unreliable association invalidates the observation.
        # Missing association metadata is only permitted for general
        # hand-posture analysis, not object-specific assessment.
        if associated is False:
            detected = None
        elif require_object_association and associated is not True:
            detected = None

        grasp_type = str(
            observation.get("detected_grasp_type") or ""
        ).strip()

        if grasp_type.upper() in UNKNOWN_TYPES:
            grasp_type = None

        if detected is not True:
            grasp_type = None

        cleaned.append({
            "timestamp_ms": timestamp,
            "detected": detected,
            "grasp_type": grasp_type,
            "association_reliable": associated,
        })

    cleaned.sort(key=lambda row: row["timestamp_ms"])

    # Preserve the last recorded observation at duplicate timestamps.
    deduplicated = {
        row["timestamp_ms"]: row
        for row in cleaned
    }
    samples = list(deduplicated.values())

    total = len(samples)
    known = [s for s in samples if s["detected"] is not None]
    grasped = [s for s in known if s["detected"] is True]
    classified = [s for s in grasped if s["grasp_type"] is not None]

    coverage = len(known) / total if total else None
    maintenance = len(grasped) / len(known) if known else None
    classification_coverage = (
        len(classified) / len(grasped)
        if grasped else None
    )

    counts = Counter(s["grasp_type"] for s in classified)
    dominant = counts.most_common(1)[0][0] if counts else None
    consistency = (
        counts[dominant] / len(classified)
        if dominant is not None else None
    )

    # Time-weighted versions: each observation counts for the time until the next one,
    # capped at max_gap_ms so a long gap in the log is not counted as observed time.
    weights = []
    for index, sample in enumerate(samples):
        if index + 1 < total:
            weights.append(min(samples[index + 1]["timestamp_ms"] - sample["timestamp_ms"], max_gap_ms))
        else:
            weights.append(0.0)
    positive = sorted(w for w in weights[:-1] if w > 0)
    if total > 1 and positive:
        weights[-1] = positive[len(positive) // 2]
    total_time = sum(weights)
    known_time = sum(w for s, w in zip(samples, weights) if s["detected"] is not None)
    grasped_time = sum(w for s, w in zip(samples, weights) if s["detected"] is True)
    classified_time = sum(
        w for s, w in zip(samples, weights) if s["detected"] is True and s["grasp_type"] is not None
    )
    time_by_type = {}
    for s, w in zip(samples, weights):
        if s["detected"] is True and s["grasp_type"] is not None:
            time_by_type[s["grasp_type"]] = time_by_type.get(s["grasp_type"], 0.0) + w

    def _pct(numerator, denominator):
        return round(100 * numerator / denominator, 1) if denominator and denominator > 0 else None

    dominant_time_type = max(time_by_type, key=time_by_type.get) if time_by_type else None
    consistency_tw = _pct(time_by_type[dominant_time_type], classified_time) if dominant_time_type else None

    # Conservative observed release/regrasp counting.
    # Missing data breaks continuity; it is not interpreted as release.
    seen_grasp = False
    release_start = None
    regrasp_count = 0
    interruption_count = 0

    for sample in samples:
        state = sample["detected"]
        timestamp = sample["timestamp_ms"]

        if state is None:
            seen_grasp = False
            release_start = None
            continue

        if state is True:
            if release_start is not None and seen_grasp:
                regrasp_count += 1
                if timestamp - release_start >= interruption_threshold_ms:
                    interruption_count += 1
            seen_grasp = True
            release_start = None

        elif seen_grasp and release_start is None:
            release_start = timestamp

    reasons = []

    if total == 0:
        reasons.append("No usable timestamped observations")
    elif len(known) < 3 or coverage < min_coverage:
        reasons.append("Insufficient reliable grasp-presence observations")

    if len(classified) < min_classified_observations:
        reasons.append("Insufficient confidently classified grasp observations")
    elif classification_coverage < min_classification_coverage:
        reasons.append("Grasp classification coverage below threshold")

    presence_assessable = (
        total > 0
        and len(known) >= 3
        and coverage >= min_coverage
    )

    classification_assessable = (
        presence_assessable
        and len(classified) >= min_classified_observations
        and classification_coverage is not None
        and classification_coverage >= min_classification_coverage
    )

    if not presence_assessable:
        status = "INSUFFICIENT_DATA"
    elif not classification_assessable:
        status = "PARTIAL_ASSESSMENT"
    else:
        status = "DESCRIPTIVE_ASSESSMENT"

    return {
        "status": status,
        "measurement_type": "observation_based_grasp_descriptors",
        "quality_score": None,
        "quality_score_status": "NOT_VALIDATED",
        "observation_count": total,
        "known_observation_count": len(known),
        "grasped_observation_count": len(grasped),
        "classified_observation_count": len(classified),
        "coverage_percent": (
            round(100 * coverage, 1) if coverage is not None else None
        ),
        "grasp_maintenance_percent": (
            round(100 * maintenance, 1)
            if maintenance is not None and presence_assessable else None
        ),
        "classification_coverage_percent": (
            round(100 * classification_coverage, 1)
            if classification_coverage is not None else None
        ),
        "grasp_type_consistency_percent": (
            round(100 * consistency, 1)
            if consistency is not None and classification_assessable else None
        ),
        "dominant_grasp_type": (
            dominant if classification_assessable else None
        ),
        "grasp_type_counts": dict(counts),
        "time_weighted": {
            "method": "each observation weighted by the time to the next observation, capped at max_gap_ms",
            "max_gap_ms": max_gap_ms,
            "observed_time_ms": round(total_time),
            "coverage_percent": _pct(known_time, total_time),
            "grasp_maintenance_percent": (
                _pct(grasped_time, known_time) if presence_assessable else None
            ),
            "classification_coverage_percent": _pct(classified_time, grasped_time),
            "grasp_type_consistency_percent": (
                consistency_tw if classification_assessable else None
            ),
            "dominant_grasp_type": (
                dominant_time_type if classification_assessable else None
            ),
            "grasp_type_time_ms": {k: round(v) for k, v in time_by_type.items()},
        },
        "regrasp_count": (
            regrasp_count if presence_assessable else None
        ),
        "interruption_count": (
            interruption_count if presence_assessable else None
        ),
        "assessment_reasons": reasons,
        "clinical_validation": False,
        "object_association_required": require_object_association,
        "limitations": [
            "Grasp posture does not prove physical object contact",
            "No grip force or object-slip measurement",
            "Uncertain tracking may conceal releases or regrasps",
            "Top-level percentages count observations; the time_weighted block weights by elapsed time",
            "No clinically validated overall quality score",
        ],
    }
