from src.preprocessing.scale_validation import measure_pair, summarise


def _sq(cx, cy, s):
    h = s / 2.0
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def test_exact_scale_gives_known_distance():
    m = measure_pair(_sq(100, 100, 100), _sq(400, 100, 100), 50.0)
    assert abs(m["mm_per_px"] - 0.5) < 1e-9 and abs(m["distance_mm"] - 150.0) < 1e-9


def test_summary_flags_error():
    pairs = [measure_pair(_sq(100, 100, 100), _sq(400, 100, 100), 50.0)]
    assert summarise(pairs, 150.0)["verdict"] == "WITHIN_TOLERANCE"
    assert summarise(pairs, 160.0)["verdict"] == "OUTSIDE_TOLERANCE"


def test_no_data():
    assert summarise([], 150.0)["verdict"] == "NO_DATA"
