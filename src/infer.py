import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image, img_size=640):
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)  # BCHW
    return image


def run_inference(model_path, input_dir, output_dir="outputs", img_size=640):
    os.makedirs(output_dir, exist_ok=True)
    ort_session = ort.InferenceSession(model_path)

    for image_path in Path(input_dir).glob("*.jpg"):
        image = cv2.imread(str(image_path))
        input_tensor = preprocess(image, img_size)
        _ = ort_session.run(None, {"images": input_tensor})

        # Заглушка: просто сохраняем входное изображение
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), image)

    print(f"[✓] Inference complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    import fire

    fire.Fire(run_inference)
