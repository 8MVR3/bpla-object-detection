import numpy as np
from ultralytics import YOLO
import fire
from pathlib import Path


def predict(image_path, model_path="models/yoloe-11s-seg.pt", class_ids=None):
    model = YOLO(model_path)

    # Фильтрация по классам
    if class_ids:
        model.set_classes([model.names[i] for i in class_ids])

    results = model.predict(image_path)

    out_dir = Path("outputs/")
    out_dir.mkdir(exist_ok=True)
    results[0].save(out_dir / "output.jpg")
    print(f"Prediction saved to {out_dir / 'output.jpg'}")


if __name__ == "__main__":
    fire.Fire(predict)
