# src/predict.py
import logging
import os
from pathlib import Path

from ultralytics import YOLO

from src.utils.logger import setup_logging
from src.utils.utils import get_git_commit_id


def predict(
    image_path: str,
    model_path: str = "models/yolov8s.pt",
    class_ids: list = None,
    save_dir: str = "outputs",
) -> Path:
    """
    Запуск предсказания модели YOLO на одном изображении.

    Args:
        image_path (str): Путь к изображению.
        model_path (str): Путь к .pt модели.
        class_ids (list): Список id классов для фильтрации.
        save_dir (str): Куда сохранять результат.

    Returns:
        Path: Путь к сохранённому изображению с детекцией.
    """
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)
    logger = logging.getLogger(__name__)

    logger.info(f"Start inference | model: {model_path}, image: {image_path}")
    logger.info(f"Git commit: {get_git_commit_id()}")

    model = YOLO(model_path)

    if class_ids:
        logger.info(f"Filtering by class IDs: {class_ids}")
        model.set_classes([model.names[i] for i in class_ids])

    results = model.predict(image_path)
    output_path = Path(save_dir) / "output.jpg"
    results[0].save(output_path)

    logger.info(f"Saved to {output_path}")
    return output_path
