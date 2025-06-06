import logging
import os
from pathlib import Path

import fire
from ultralytics import YOLO

from src.utils.logger import setup_logging
from src.utils.utils import get_git_commit_id


def predict(
    image_path: str,
    model_path: str = "models/yoloe-11s-seg.pt",
    class_ids: list = None,
    save_dir: str = "outputs",
):
    # Настройка директории и логгера
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting inference for {image_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Git commit ID: {get_git_commit_id()}")

    # Загрузка модели
    model = YOLO(model_path)

    # Установка классов
    if class_ids:
        logger.info(f"Filtering by class IDs: {class_ids}")
        model.set_classes([model.names[i] for i in class_ids])

    # Предсказание
    results = model.predict(image_path)
    output_path = Path(save_dir) / "output.jpg"
    results[0].save(output_path)

    logger.info(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(predict)
