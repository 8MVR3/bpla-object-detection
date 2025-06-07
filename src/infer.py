import logging
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image, img_size=640):
    """
    Предобработка изображения перед подачей в модель:
    - изменение размера до (img_size, img_size),
    - нормализация значений пикселей,
    - изменение формата на CHW,
    - добавление batch-измерения.

    Args:
        image (np.ndarray): исходное изображение в формате HWC.
        img_size (int): требуемый размер входного изображения (по умолчанию 640).

    Returns:
        np.ndarray: подготовленное изображение в формате NCHW.
    """
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # добавление batch-измерения
    return image


def run_inference(model_path, input_dir, output_dir="outputs", img_size=640):
    """
    Выполняет инференс изображений из указанной директории с использованием ONNX-модели.

    Args:
        model_path (str): путь к ONNX-модели.
        input_dir (str): директория с входными изображениями (.jpg).
        output_dir (str): директория, куда будут сохранены изображения (по умолчанию "outputs").
        img_size (int): размер входного изображения для модели (по умолчанию 640).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Инициализация логгера
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Running inference with model: {model_path}")

    # Загрузка ONNX модели
    ort_session = ort.InferenceSession(model_path)

    # Обработка каждого .jpg файла в папке
    for image_path in Path(input_dir).glob("*.jpg"):
        image = cv2.imread(str(image_path))
        input_tensor = preprocess(image, img_size)

        # Получаем имя входного тензора и запускаем инференс
        input_name = ort_session.get_inputs()[0].name
        _ = ort_session.run(None, {input_name: input_tensor})

        # TODO: Добавить постобработку (например, отрисовку боксов)

        # Сохраняем (пока что необработанное) изображение
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved output: {output_path}")

    logger.info("Inference completed. Results saved to: %s", output_dir)


if __name__ == "__main__":
    import fire

    # Запуск из CLI: python infer.py --model_path=... --input_dir=... [--output_dir=...]
    fire.Fire(run_inference)
