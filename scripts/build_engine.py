# scripts/build_engine.py

import sys

import tensorrt as trt

onnx_file = "exports/weights/best.onnx"
engine_file = "exports/weights/best.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Читаем ONNX-модель
with open(onnx_file, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed to parse ONNX:", onnx_file)
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

# Настройки билдера
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1 GB

# Собираем движок
engine = builder.build_engine(network, config)

# Сохраняем в файл
with open(engine_file, "wb") as f:
    f.write(engine.serialize())

print("✅ Engine built and saved:", engine_file)
