import numpy as np
from ultralytics import YOLOE


def predict(image_path, class_ids=None):
    model = YOLOE("models/yoloe-11s-seg.pt")

    # Фильтрация по выбранным классам
    if class_ids:
        model.set_classes([model.names[i] for i in class_ids])

    # Визуальные промпты (пример)
    visual_prompts = {
        "bboxes": np.array([[100, 100, 200, 200]]),
        "cls": np.array([0])  # person
    }

    results = model.predict(image_path, visual_prompts=visual_prompts)
    results[0].save("output.jpg")
