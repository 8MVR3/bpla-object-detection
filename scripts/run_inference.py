# scripts/run_inference.py

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

engine_file = "exports/weights/best.engine"
img_path = "assets/sample.jpg"  # путь к изображению для инференса

TRT_LOGGER = trt.Logger()

# Загрузка движка
with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Подготовка входного изображения
input_shape = tuple(engine.get_binding_shape(0))
img = cv2.imread(img_path)
img = cv2.resize(img, (input_shape[2], input_shape[1]))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # CHW
img = np.expand_dims(img, axis=0)  # NCHW
img = np.ascontiguousarray(img)

# Выделяем память
d_input = cuda.mem_alloc(img.nbytes)
output_shape = tuple(engine.get_binding_shape(1))
output = np.empty(output_shape, dtype=np.float32)
d_output = cuda.mem_alloc(output.nbytes)

# Копируем и запускаем
cuda.memcpy_htod(d_input, img)
context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(output, d_output)

print("✅ Inference output shape:", output.shape)
print("🔢 Output sample:", output.ravel()[:10])
