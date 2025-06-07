import os
import sys

import mlflow
import mlflow.pytorch
import torch

# Задаём URI шаблон для загрузки модели из MLflow
logged_model_uri_template = "runs:/{}/model"

# Инициализация клиента MLflow
client = mlflow.tracking.MlflowClient()

# Получение последнего запуска из эксперимента с ID "0"
latest_runs = client.search_runs(
    experiment_ids=["0"],
    order_by=["start_time DESC"],
    max_results=1,
)

# Проверка наличия хотя бы одного запуска
if not latest_runs:
    print("❌ No MLflow runs found in experiment ID '0'.")
    sys.exit(1)

# Извлечение run_id
run = latest_runs[0]
run_id = run.info.run_id
model_uri = logged_model_uri_template.format(run_id)

# Проверка, что артефакт модели действительно существует
artifact_path = os.path.join("mlruns", "0", run_id, "artifacts", "model")
if not os.path.isdir(artifact_path):
    print(f"❌ Model artifact not found at expected path: {artifact_path}")
    print("➡️ Make sure your training script includes `mlflow.pytorch.log_model(model, 'model')`.")
    sys.exit(1)

# Загрузка модели из MLflow
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Пример входного тензора (можно заменить на реальные размеры)
dummy_input = torch.randn(1, 3, 32, 32)

# Создание директории для сохранения ONNX-модели
onnx_dir = "onnx_models"
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, f"model_{run_id}.onnx")

# Экспорт модели в формат ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print(f"✅ Model successfully exported to {onnx_path}")
