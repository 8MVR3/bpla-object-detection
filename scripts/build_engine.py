# scripts/build_engine.py
#
# Этот скрипт выполняет конвертацию модели формата ONNX в формат TensorRT Engine (.engine).
# Используется API TensorRT для создания оптимизированного движка выполнения модели.
# Подходит для ускоренного инференса на NVIDIA GPU.

import sys

import tensorrt as trt

# Путь к входной ONNX-модели
onnx_file = "exports/weights/best.onnx"
# Путь для сохранения скомпилированного TensorRT-движка
engine_file = "exports/weights/best.engine"

# Инициализация логгера TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Создание билдера и сетевой структуры
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    # Явное указание батча (обязательно для ONNX)
    1
    << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# Создание парсера для обработки ONNX-модели
parser = trt.OnnxParser(network, TRT_LOGGER)

# Загрузка и парсинг ONNX-файла
with open(onnx_file, "rb") as model:
    if not parser.parse(model.read()):
        print("Ошибка при разборе ONNX-модели:", onnx_file)
        for i in range(parser.num_errors):
            print(parser.get_error(i))  # Вывод всех ошибок парсинга
        sys.exit(1)

# Создание конфигурации билдера
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # Максимальный размер рабочей области (1 ГБ)

# Построение движка TensorRT из загруженной сети
engine = builder.build_engine(network, config)

# Сохранение движка в файл
with open(engine_file, "wb") as f:
    f.write(engine.serialize())

print("TensorRT Engine успешно построен и сохранён:", engine_file)
