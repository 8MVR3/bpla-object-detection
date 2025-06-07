# scripts/run_inference.py
#
# Скрипт выполняет инференс изображения с использованием скомпилированного TensorRT Engine.
# Используются библиотеки PyCUDA и TensorRT. Выходной результат — "сырые" предсказания модели.

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

# Путь к TensorRT Engine-файлу
engine_file = "exports/weights/best.engine"
# Путь к изображению, на котором будет выполняться инференс
img_path = "assets/sample.jpg"

# Инициализация логгера TensorRT
TRT_LOGGER = trt.Logger()

# Загрузка скомпилированного движка TensorRT
with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Создание контекста выполнения инференса
context = engine.create_execution_context()

# Получение входной формы (формат CHW, т.е. (C, H, W))
input_shape = tuple(engine.get_binding_shape(0))

# Загрузка изображения и его предварительная обработка
img = cv2.imread(img_path)
# Изменение размера в соответствии с моделью
img = cv2.resize(img, (input_shape[2], input_shape[1]))
img = img.astype(np.float32) / 255.0  # Нормализация значений пикселей
img = np.transpose(img, (2, 0, 1))  # Перестановка каналов: HWC -> CHW
img = np.expand_dims(img, axis=0)  # Добавление батча: CHW -> NCHW
# Убедиться, что массив располагается непрерывно в памяти
img = np.ascontiguousarray(img)

# Выделение GPU-памяти под вход и выход
d_input = cuda.mem_alloc(img.nbytes)
output_shape = tuple(engine.get_binding_shape(1))
output = np.empty(output_shape, dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)

# Копирование входных данных на устройство (GPU)
cuda.memcpy_htod(d_input, img)

# Выполнение инференса
context.execute_v2([int(d_input), int(d_output)])

# Копирование выходных данных обратно в хост-память (CPU)
cuda.memcpy_dtoh(output, d_output)

# Вывод результатов
print("Inference output shape:", output.shape)
print("Output sample:", output.ravel()[:10])  # Пример первых 10 значений
