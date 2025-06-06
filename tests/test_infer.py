import cv2
import numpy as np

from src.infer import run_inference


def test_run_inference_creates_outputs(tmp_path):
    # Создаём dummy изображение
    dummy_image = 255 * np.ones((640, 640, 3), dtype=np.uint8)
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    test_image_path = input_dir / "test.jpg"
    cv2.imwrite(str(test_image_path), dummy_image)

    output_dir = tmp_path / "outputs"
    model_path = "exports/weights/best.onnx"

    # Запускаем инференс
    run_inference(str(model_path), str(input_dir), str(output_dir))

    # Проверяем, что файл сохранён
    saved = list(output_dir.glob("*.jpg"))
    assert len(saved) == 1
