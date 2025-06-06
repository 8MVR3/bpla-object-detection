import pytest
from ultralytics import YOLO


def test_model_loads():
    """Check that YOLO model loads without error"""
    model = YOLO("models/yolov8s.pt")
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.skip(reason="Long test, enable manually")
def test_model_predict():
    """Run simple inference on one image"""
    model = YOLO("models/yolov8s.pt")
    results = model.predict("data/test/images/00001.jpg")
    assert len(results) > 0
