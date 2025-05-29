from ultralytics.data.dataset import YOLODataset
import numpy as np


class CustomDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_label(self, label_path):
        with open(label_path, 'r') as f:
            lines = [line.strip().split(',') for line in f.readlines()]

        labels = []
        for line in lines:
            x1, y1, w, h = map(float, line[:4])
            class_id = int(line[4])  # Предполагаем, что 5-й элемент - class_id

            # Конвертация в YOLO-формат
            x_center = (x1 + w/2) / self.img_size[0]
            y_center = (y1 + h/2) / self.img_size[1]
            width = w / self.img_size[0]
            height = h / self.img_size[1]

            labels.append([class_id, x_center, y_center, width, height])

        return np.array(labels)
