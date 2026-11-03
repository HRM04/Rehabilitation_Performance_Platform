"""Scale validation maths: measure a known marker-to-marker distance using only the marker sides."""
import math
import statistics


def side_px(c):
    return sum(math.dist(c[i], c[(i + 1) % 4]) for i in range(4)) / 4.0


def centre(c):
    return (sum(p[0] for p in c) / 4.0, sum(p[1] for p in c) / 4.0)


def measure_pair(c0, c1, marker_mm=50.0):
    s0, s1 = side_px(c0), side_px(c1)
    mm_per_px = marker_mm / ((s0 + s1) / 2.0)
    d_px = math.dist(centre(c0), centre(c1))
    return {"side0_px": s0, "side1_px": s1, "mm_per_px": mm_per_px,
            "distance_px": d_px, "distance_mm": d_px * mm_per_px, "side_ratio": s0 / s1}


def summarise(pairs, known_mm, tol_fraction=0.02):
    if not pairs:
        return {"frames_with_both": 0, "verdict": "NO_DATA"}
    dist = [p["distance_mm"] for p in pairs]
    med = statistics.median(dist)
    err_pct = 100.0 * (med - known_mm) / known_mm
    return {"frames_with_both": len(pairs), "known_mm": known_mm,
            "median_distance_mm": med, "median_error_mm": med - known_mm, "median_error_pct": err_pct,
            "worst_abs_error_mm": max(abs(x - known_mm) for x in dist),
            "frame_to_frame_spread_mm": max(dist) - min(dist),
            "mean_mm_per_px": statistics.mean(p["mm_per_px"] for p in pairs),
            "median_side_ratio": statistics.median(p["side_ratio"] for p in pairs),
            "tolerance_pct": 100.0 * tol_fraction,
            "verdict": "WITHIN_TOLERANCE" if abs(err_pct) <= 100.0 * tol_fraction else "OUTSIDE_TOLERANCE"}
