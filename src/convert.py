from ultralytics import YOLO
import fire
from pathlib import Path


def export_to_onnx(model_path="models/yoloe-11s-seg.pt", output_path="models/yoloe-11s-seg.onnx"):
    model = YOLO(model_path)
    model.export(format="onnx")
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    fire.Fire(export_to_onnx)
