import logging
import os
import subprocess
import datetime
from ultralytics import YOLO
from src.utils.plot_metrics import plot_results
from src.utils.utils import get_git_commit_id


def setup_logging(save_dir):
    log_file = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_exp_dir(name="exp"):
    runs_dir = os.path.join("runs", "train")
    exps = [d for d in os.listdir(runs_dir) if d.startswith(name)]
    exp_nums = [int(d.replace(name, ""))
                for d in exps if d.replace(name, "").isdigit()]
    next_num = max(exp_nums + [0]) + 1
    return os.path.join(runs_dir, f"{name}{next_num}")


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
        deterministic=True
    )

    # Определяем директорию эксперимента
    exp_dir = results.save_dir  # runs/train/expXYZ

    # Настройка логирования
    setup_logging(exp_dir)

    # Логируем общую информацию
    logging.info(f"Training results saved to: {exp_dir}")
    logging.info(f"Git commit ID: {get_git_commit_id()}")

    # Графики
    csv_path = os.path.join(exp_dir, "results.csv")
    if os.path.exists(csv_path):
        plot_results(csv_path, exp_dir)
        logging.info("Metric plots saved.")
    else:
        logging.warning("results.csv not found — skipping plots.")


if __name__ == "__main__":
    main()
