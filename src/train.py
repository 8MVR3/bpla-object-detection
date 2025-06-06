import logging
import os

from ultralytics import YOLO

from src.utils.logger import setup_logging
from src.utils.plot_metrics import plot_results
from src.utils.utils import get_git_commit_id


def main():
    # Подготовка модели
    model = YOLO("models/yolov8s.pt")

    # Запуск обучения
    results = model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        project="runs/train",
        name="exp",
        verbose=True,
        deterministic=True,
    )

    # Определяем директорию эксперимента
    exp_dir = results.save_dir  # runs/train/expXYZ
    os.makedirs(exp_dir, exist_ok=True)

    # Настройка логирования
    setup_logging(exp_dir)
    logger = logging.getLogger(__name__)

    logger.info(f"Training results saved to: {exp_dir}")
    logger.info(f"Git commit ID: {get_git_commit_id()}")

    # Построение графиков метрик
    csv_path = os.path.join(exp_dir, "results.csv")
    if os.path.exists(csv_path):
        plot_results(csv_path, exp_dir)
        logger.info("Metric plots saved.")
    else:
        logger.warning("results.csv not found — skipping plots.")


if __name__ == "__main__":
    main()
