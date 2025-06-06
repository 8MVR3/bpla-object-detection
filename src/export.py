import argparse
import os
import subprocess

from ultralytics import YOLO


def export_model(weights_path: str, export_dir: str = "exports", format: str = "onnx"):
    model = YOLO(weights_path)

    # Имя эксперимента
    exp_name = os.path.basename(os.path.dirname(weights_path))
    save_dir = os.path.join(export_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if format == "engine":
        # Экспорт в ONNX
        onnx_path = weights_path.replace(".pt", ".onnx")
        model.export(format="onnx", dynamic=True, half=False, simplify=True, imgsz=640)

        # Проверка, что .onnx появился
        if not os.path.exists(onnx_path):
            print("ONNX export failed — cannot continue to TensorRT.")
            return

        # Конвертация ONNX → TensorRT через trtexec
        engine_path = onnx_path.replace(".onnx", ".engine")
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--explicitBatch",
            "--workspace=1024",
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Перенос в нужную папку
        os.rename(engine_path, os.path.join(save_dir, os.path.basename(engine_path)))
        print(f"Exported TensorRT model to {save_dir}")

    else:
        # Обычный экспорт
        model.export(format=format, dynamic=True, half=False, simplify=True, imgsz=640)
        export_file = weights_path.replace(".pt", f".{format}")
        if os.path.exists(export_file):
            os.rename(
                export_file, os.path.join(save_dir, os.path.basename(export_file))
            )
            print(f"Exported {format.upper()} model to {save_dir}")
        else:
            print(f"Export failed: {format.upper()} file not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to trained weights .pt"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "engine"],
        help="Export format",
    )
    args = parser.parse_args()

    export_model(args.weights, format=args.format)
