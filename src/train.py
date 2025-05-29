import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
from typing import Dict, Type
import warnings


def validate_config(cfg: DictConfig, required: Dict[str, Type]) -> None:
    """Валидация структуры и типов конфигурации"""
    for key, type_ in required.items():
        if not OmegaConf.select(cfg, key):
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(OmegaConf.select(cfg, key), type_):
            raise ValueError(
                f"Invalid type for {key}. Expected {type_}, got {type(OmegaConf.select(cfg, key))}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """
    Основная функция обучения с Hydra-конфигурацией

    Args:
        cfg: Конфигурация из Hydra (объединяет все yaml-файлы)
    """
    # Игнорируем предупреждения Hydra о рабочей директории
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=".*working directory.*")

    # Проверка обязательных параметров
    required_config = {
        'data.dataset.train_path': str,
        'model.weights_path': str,
        'training.epochs': int,
        'model.input_size': list,
        'training.batch_size': int
    }
    validate_config(cfg, required_config)

    # Преобразуем конфиг в словарь (для логгирования)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print("\nКонфигурация обучения:")
    print(OmegaConf.to_yaml(cfg))

    # Инициализация модели
    model = YOLO(cfg.model.weights_path)

    # Параметры обучения
    train_kwargs = {
        "data": cfg.data.dataset.train_path,
        "epochs": cfg.training.epochs,
        "imgsz": cfg.model.input_size[0],
        "batch": cfg.training.batch_size,
        "device": cfg.training.device,
        "optimizer": cfg.training.optimizer,
        "lr0": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "fliplr": cfg.transforms.train[0].p,
        "name": f"yoloe_{cfg.model.name}_train"
    }

    # Опциональные параметры (если указаны в конфиге)
    if OmegaConf.select(cfg, "training.patience"):
        train_kwargs["patience"] = cfg.training.patience
    if OmegaConf.select(cfg, "transforms.hsv_h"):
        train_kwargs["hsv_h"] = cfg.transforms.hsv_h
    if OmegaConf.select(cfg, "transforms.hsv_s"):
        train_kwargs["hsv_s"] = cfg.transforms.hsv_s

    # Запуск обучения
    print("\nНачало обучения с параметрами:")
    for k, v in train_kwargs.items():
        print(f"- {k}: {v}")

    results = model.train(**train_kwargs)

    print("\nОбучение завершено. Модель сохранена в:")
    print(f"runs/detect/{train_kwargs['name']}")


if __name__ == "__main__":
    train()
