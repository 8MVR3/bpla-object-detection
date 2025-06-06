# tests/test_model.py
import pytest
from ultralytics import YOLO


def test_model_loads():
    """Проверяем, что YOLO-модель загружается без ошибок"""
    model = YOLO("models/yolov8s.pt")
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.skip(reason="Для ускорения тестов — можно включить вручную")
def test_model_predict():
    """Простой инференс на одном изображении"""
    model = YOLO("models/yolov8s.pt")
    results = model.predict("data/test/images/00001.jpg")
    assert len(results) > 0
