import argparse
import logging
import os
import subprocess

from ultralytics import YOLO


def export_model(weights_path: str, export_dir: str = "exports", format: str = "onnx"):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model from {weights_path}: {e}")
        return

    exp_name = os.path.basename(os.path.dirname(weights_path))
    save_dir = os.path.join(export_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    if format == "engine":
        try:
            onnx_path = weights_path.replace(".pt", ".onnx")
            logger.info(f"Exporting to ONNX first: {onnx_path}")
            model.export(
                format="onnx", dynamic=True, half=False, simplify=True, imgsz=640
            )

            if not os.path.exists(onnx_path):
                logger.error("ONNX export failed — cannot continue to TensorRT.")
                return

            engine_path = onnx_path.replace(".onnx", ".engine")
            command = [
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--explicitBatch",
                "--workspace=1024",
            ]
            logger.info(f"Running TensorRT conversion: {' '.join(command)}")
            subprocess.run(command, check=True)

            # Перенос в нужную папку
            os.rename(
                engine_path, os.path.join(save_dir, os.path.basename(engine_path))
            )
            logger.info(f"Exported TensorRT model to {save_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"TensorRT conversion failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during engine export: {e}")

    else:
        try:
            logger.info(f"Exporting model to {format} format")
            model.export(
                format=format, dynamic=True, half=False, simplify=True, imgsz=640
            )
            export_file = weights_path.replace(".pt", f".{format}")
            if os.path.exists(export_file):
                os.rename(
                    export_file, os.path.join(save_dir, os.path.basename(export_file))
                )
                logger.info(f"Exported {format.upper()} model to {save_dir}")
            else:
                logger.error(f"Export failed: {format.upper()} file not found.")

        except Exception as e:
            logger.error(f"Error during export: {e}")


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
