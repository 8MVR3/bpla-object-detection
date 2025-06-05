# src/predict.py

import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(config_path="../configs", config_name="predict", version_base="1.3")
def predict(cfg: DictConfig):
    model = YOLO(cfg.model.weights_path)
    results = model.predict(
        source=cfg.predict.source,
        imgsz=cfg.predict.imgsz,
        device=cfg.predict.device,
    )
    print(results)


if __name__ == "__main__":
    predict()
