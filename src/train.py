import torch
from ultralytics import YOLO
import yaml
import os
print(f"Права на папку: {oct(os.stat('data/train').st_mode)[-3:]}")
print(f"Содержимое папки: {os.listdir('data/train')}")


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()

    # Автоматический выбор устройства
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["training"]["device"] = device  # Обновляем конфиг

    model = YOLO(cfg["model"]["weights_path"])

    train_kwargs = {
        "data": cfg["data"]["train_path"],
        "epochs": cfg["training"]["epochs"],
        "imgsz": cfg["model"]["input_size"][0],  # Берем первый элемент списка
        "batch": cfg["training"]["batch_size"],
        "device": device,  # Используем выбранное устройство
    }

    results = model.train(**train_kwargs)


if __name__ == "__main__":
    train()
