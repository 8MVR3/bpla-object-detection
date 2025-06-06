# src/convert.py
import fire
from ultralytics import YOLO


def export_to_onnx(model_path="models/yolov8s.pt", output_path="models/yolov8s.onnx"):
    model = YOLO(model_path)
    model.export(format="onnx")
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    fire.Fire(export_to_onnx)
