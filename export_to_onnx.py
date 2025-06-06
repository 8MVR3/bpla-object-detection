import os

import mlflow.pytorch
import torch

# Загрузка последней модели из MLflow
logged_model_uri = "runs:/{}/model"

# Получим ID последнего запуска (run ID)
client = mlflow.tracking.MlflowClient()
latest_run = client.search_runs(
    experiment_ids=["0"], order_by=["start_time DESC"], max_results=1
)[0]
run_id = latest_run.info.run_id

# Загрузка модели
model = mlflow.pytorch.load_model(logged_model_uri.format(run_id))
model.eval()

# Пример входного тензора
dummy_input = torch.randn(1, 3, 32, 32)

# Путь для сохранения ONNX-модели
onnx_path = os.path.join("onnx_models", f"model_{run_id}.onnx")
os.makedirs("onnx_models", exist_ok=True)

# Экспорт модели в ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print(f"Модель успешно экспортирована в {onnx_path}")
