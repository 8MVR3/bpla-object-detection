import hydra
from omegaconf import DictConfig
from ultralytics import YOLO
import mlflow
import os
from utils.utils import get_git_commit_id
# from ultralytics import set_setting

# # Отключаем внутренний MLflow в ultralytics, чтобы избежать конфликта
# set_setting('mlflow', False)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    # Устанавливаем эксперимент MLflow
    mlflow.set_experiment("yolo_training_experiment")

    # Запускаем MLflow run вручную
    with mlflow.start_run(run_id=None) as run:
        run_id = run.info.run_id
        print(f"MLflow run started with run_id={run_id}")

        # Логируем git commit (если есть)
        try:
            commit_id = get_git_commit_id()
            mlflow.set_tag("git_commit", commit_id)
        except Exception as e:
            print(f"Ошибка при получении git commit: {e}")

        # Логируем параметры конфигурации
        mlflow.log_param("epochs", cfg.training.epochs)
        mlflow.log_param("batch_size", cfg.training.batch_size)
        mlflow.log_param("device", cfg.training.device)
        mlflow.log_param("weights_path", cfg.model.weights_path)
        mlflow.log_param("data_path", cfg.data.data_path)

        # Создаем модель и запускаем обучение
        model = YOLO(cfg.model.weights_path)

        # Запускаем обучение
        results = model.train(
            data=cfg.data.data_path,
            epochs=cfg.training.epochs,
            batch=cfg.training.batch_size,
            device=cfg.training.device,
            workers=cfg.training.workers,
            imgsz=cfg.training.imgsz,
            project=cfg.training.project,
            name=cfg.training.name,
        )

        # Попытка залогировать метрики, если они есть
        try:
            metrics = getattr(model.trainer, "metrics", None)
            if metrics and hasattr(metrics, "results_dict"):
                for key, value in metrics.results_dict.items():
                    try:
                        mlflow.log_metric(key, float(value))
                    except Exception:
                        print(f"Не удалось залогировать метрику {key}")
            else:
                print("Метрики не найдены в model.trainer.metrics")
        except Exception as e:
            print(f"Ошибка при логировании метрик: {e}")

        # Логируем артефакты — папку с результатами тренировки
        run_dir = os.path.join(cfg.training.project, cfg.training.name)
        if os.path.exists(run_dir):
            try:
                mlflow.log_artifacts(run_dir)
            except Exception as e:
                print(f"Ошибка при логировании артефактов: {e}")
        else:
            print(f"Папка с результатами {run_dir} не найдена")

        print(f"MLflow run {run_id} завершен успешно")


if __name__ == "__main__":
    train()
