import hydra
from omegaconf import DictConfig
from ultralytics import YOLO


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    model = YOLO(cfg.model.weights_path)
    results = model.train(
        data=cfg.data.data_path,   # ✅ путь до data.yaml
        epochs=cfg.training.epochs,
        batch=cfg.training.batch_size,
        device=cfg.training.device,
        workers=cfg.training.workers
    )


if __name__ == "__main__":
    train()
