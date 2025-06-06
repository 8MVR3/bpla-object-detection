import shutil
from pathlib import Path

import numpy as np

from src.infer import predict


def test_predict_runs(tmp_path):
    # Dummy изображение (320x320 белый)
    import cv2

    image = 255 * np.ones((320, 320, 3), dtype=np.uint8)
    image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(image_path), image)

    # Предсказание (в режиме smoke test)
    try:
        predict(str(image_path), model_path="models/yolov8s.pt")
        output_file = Path("outputs/output.jpg")
        assert output_file.exists()
    finally:
        shutil.rmtree("outputs", ignore_errors=True)
