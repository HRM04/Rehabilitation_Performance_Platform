import inspect
import src.assessment.game_trajectory as gt

def _traj(active=True):
    cls = next(c for c in vars(gt).values() if inspect.isclass(c) and "recover_object" in c.__dict__)
    o = cls.__new__(cls)
    o.attempt_active = active
    o.active_object_id = 4
    o.active_object_class = "cup"
    o.last_seen_position = (300, 400)
    o.start_point = (256, 396)
    o.outcome = "TRACKING LOST"
    return o

NEAR, FAR = (330, 400), (900, 600)

def _d(i, c="cup", conf=0.9, at=NEAR):
    return {"object_id": i, "class_name": c, "confidence": conf, "center": at}

def test_not_active_returns_none():
    assert _traj(False).recover_object([_d(4)]) is None

def test_contact_on_other_class_is_ignored():
    t = _traj()
    assert t.recover_object([_d(9, "cake")], contact_object_ids={9}) is None
    assert t.active_object_id == 4

def test_same_id_near_recovered_without_contact():
    assert _traj().recover_object([_d(4)])["object_id"] == 4

def test_same_id_far_needs_five_frames():
    t = _traj()
    for _ in range(4):
        assert t.recover_object([_d(4, at=FAR)]) is None
    assert t.recover_object([_d(4, at=FAR)])["object_id"] == 4

def test_new_id_same_class_near_needs_three_frames():
    t = _traj()
    assert t.recover_object([_d(7)]) is None
    assert t.recover_object([_d(7)]) is None
    assert t.recover_object([_d(7)])["object_id"] == 7
    assert t.active_object_id == 7

def test_two_cups_are_not_guessed():
    t = _traj()
    for _ in range(6):
        assert t.recover_object([_d(7), _d(8)]) is None

def test_low_confidence_is_not_used_for_reacquisition():
    t = _traj()
    for _ in range(6):
        assert t.recover_object([_d(7, conf=0.2)]) is None
