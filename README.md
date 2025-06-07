# BPLA Object Detection with Prompting

## 📌 Project Overview

This repository implements an end-to-end object detection pipeline for aerial imagery captured by UAVs (unmanned aerial vehicles), using YOLOv8. The project is structured according to MLOps best practices, supporting full lifecycle management: from data and training to inference and deployment.

---

## ⚙️ Environment Setup

### 📥 Clone the Repository

```bash
git clone https://github.com/8MVR3/bpla-object-detection.git
cd bpla-object-detection
```

### 📦 Install Dependencies

Using Poetry:

```bash
poetry install
poetry shell
```

Or via `pip`:

```bash
pip install -r requirements.txt
```

### ✅ Set up Git Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🗂️ Dataset Management with DVC

We use [DVC](https://dvc.org/) to track training/validation/test datasets.

### 📡 Download Data

```bash
dvc pull
```

This will download data from Google Drive via a service account (already configured).

---

## 🏋️ Training

### 📄 Configure Training

Edit `configs/train.yaml` to modify training parameters:

```yaml
model:
    weights_path: models/yolov8s.pt
    input_size: [640, 640]
data:
    data_path: data/data.yaml
training:
    epochs: 5
    batch_size: 16
    device: cuda:0
    workers: 2
    imgsz: 640
    project: runs/train
    name: exp1
```

### 🚀 Run Training

```bash
python src/train.py
```

Training logs, plots, and weights will be saved to `runs/train/exp1/`.

---

## 🔍 Inference

### 💻 ONNX Runtime Inference

```bash
python src/infer.py \
  --model_path=exports/weights/best.onnx \
  --input_dir=data/test/images \
  --output_dir=outputs/
```

### 🌐 Inference Server (FastAPI)

Run:

```bash
poetry run python src/serve.py
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to upload an image and get predictions via Swagger UI.

---

## 📦 Model Export

### ➡️ Export to ONNX

```bash
python scripts/export_onnx.py --model models/yolov8s.pt --output exports/weights/best.onnx
```

### ⚡ Export to TensorRT

```bash
python scripts/build_engine.py  # Builds best.engine from best.onnx
```

---

## 🧪 Tests

Run all tests:

```bash
pytest tests/
```

Includes:

-   `test_utils.py`
-   `test_export.py`
-   `test_infer.py`
-   `test_dataloader.py`
-   `test_model.py`
-   `test_cli_infer.py`
-   `test_api.py`

---

## 📊 Logging & Monitoring

-   Metrics (loss, mAP, precision, recall) are saved in `runs/train/...`
-   Plots saved to `plots/`
-   Git commit ID is captured for reproducibility

---

## 🗃️ Project Structure

```
bpla-object-detection/
├── configs/          # Hydra YAML configs
├── data/             # DVC-tracked datasets
├── models/           # YOLOv8 weights (.pt)
├── outputs/          # ONNX inference results
├── plots/            # Metric plots
├── scripts/          # Export/build scripts
├── src/              # Source code
│   ├── train.py
│   ├── infer.py
│   ├── serve.py      # FastAPI app
│   └── utils/
├── tests/            # Unit tests
├── .gitignore
├── dvc.yaml
├── pyproject.toml    # Poetry config
└── README.md
```

---

## ✅ Checkpoints

-   ✅ Pre-commit hooks (black, isort, flake8, prettier)
-   ✅ Hydra configs for training and inference
-   ✅ Inference CLI and FastAPI server
-   ✅ DVC + GDrive integration
-   ✅ ONNX + TensorRT export
-   ✅ Full CI workflow via GitHub Actions

---

## 👤 Author

Project for MLOps course @ MIPT (Spring 2025)

**Author:** Vyacheslav Mikholap
**Email:** [mikholap.vv@phystech.edu](mailto:mikholap.vv@phystech.edu)
**GitHub:** [8MVR3](https://github.com/8MVR3)

---

## 📎 License

This project is licensed under the MIT License.
