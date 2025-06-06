import logging
import os

import fire
from ultralytics import YOLO

from src.utils.logger import setup_logging
from src.utils.utils import get_git_commit_id


def export_trt(
    model_path: str = "runs/train/exp/weights/best.pt",
    save_dir: str = "exports/trt",
    imgsz: int = 640,
    half: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(save_dir)
    logger = logging.getLogger(__name__)

    logger.info(f"Exporting model to TensorRT: {model_path}")
    logger.info(f"Save dir: {save_dir}")
    logger.info(f"Git commit ID: {get_git_commit_id()}")

    # Загрузка модели
    model = YOLO(model_path)

    # Экспорт в TensorRT
    result = model.export(
        format="engine",  # TensorRT
        imgsz=imgsz,
        half=half,
        device="cuda",
        dynamic=False,
        simplify=True,
        opset=12,
        save_dir=save_dir,
    )

    logger.info(f"Model exported to: {result}")


if __name__ == "__main__":
    fire.Fire(export_trt)
