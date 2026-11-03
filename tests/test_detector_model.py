def test_detector_model_is_selectable_and_defaults_to_m():
    src = open("src/object_detection/yolo_detector.py").read()
    assert 'os.environ.get("YOLO_MODEL", "yolo11m.pt")' in src
    assert "model = YOLO(MODEL_NAME)" in src
    assert 'YOLO("yolo11n.pt")' not in src
