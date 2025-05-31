import torch
from ultralytics import YOLO
from pathlib import Path


def validate_paths():
    base = Path("D:/bpla-object-detection")

    # Проверка модели
    model_path = base / "models/yoloe-v8m-seg.pt"
    assert model_path.exists(), f"Модель не найдена по пути: {model_path}"

    # Проверка данных
    data_path = base / "data/data.yaml"
    assert data_path.exists(), f"Файл data.yaml не найден: {data_path}"

    # Проверка папок с данными
    for split in ['train', 'val']:
        img_dir = base / f"data/{split}/images"
        lbl_dir = base / f"data/{split}/labels"

        assert img_dir.exists(), f"Нет папки: {img_dir}"
        assert lbl_dir.exists(), f"Нет папки: {lbl_dir}"
        assert any(img_dir.iterdir()), f"Нет изображений в: {img_dir}"
        assert any(lbl_dir.iterdir()), f"Нет разметки в: {lbl_dir}"


def train():
    # Настройки CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Загрузка модели
    model = YOLO("D:/bpla-object-detection/models/yoloe-v8m-seg.pt")

    # Параметры обучения
    train_args = {
        "data": "D:/bpla-object-detection/data/data.yaml",
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,  # Уменьшите batch_size для сегментации
        "device": "cuda:0",
        "name": "bpla_seg_train",
        "task": "segment",  # Явно указываем задачу
        "mask_ratio": 4,    # Должно совпадать с data.yaml
        "workers": 2        # Уменьшите workers для стабильности
    }

    # Запуск обучения
    results = model.train(**train_args)


if __name__ == "__main__":
    train()
