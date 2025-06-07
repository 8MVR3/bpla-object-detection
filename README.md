# BPLA Object Detection with Prompting

## 📌 Project Overview

This repository implements an end-to-end object detection pipeline for aerial imagery captured by UAVs (unmanned aerial vehicles), using YOLOv8. The project is structured according to MLOps best practices, supporting full lifecycle management: from data and training to inference and deployment.

---

## ⚙️ Environment Setup

### 📅 Clone the Repository

```bash
git clone https://github.com/8MVR3/bpla-object-detection.git
cd bpla-object-detection
```

> ⚠️ **Python version requirement:**
> This project requires **Python 3.10 or 3.11**.
> Python 3.12+ is not supported due to known incompatibility with `flake8`.

## 🚀 Install Dependencies

### Option 1: Using Poetry (recommended)

```bash
poetry install
poetry shell
```

### Option 2: Using pip (manual setup)

#### Create and activate virtual environment

```bash
python -m venv .venv
```

##### For Windows:

```
.venv\Scripts\activate
```

##### For Linux/macOS:

```
source .venv/bin/activate
```

#### Then install dependencies

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

This will download data from Google Drive via a service account (see below).

### 🔐 GDrive DVC Authentication

To download data via `dvc pull`, you need a service account key.

📅 **Download the JSON key** from this shared Google Drive folder:
👉 [Download keys](https://drive.google.com/drive/folders/19BrlHrNiocZAojDM6Hs8gfoPZjVbvD_k?usp=sharing)

Then place the file into the project root, and update `.dvc/config` if needed:

```
['remote "gdrive_storage"']
    url = gdrive://1KPNy9iGWudZXNfDNkDGhLXuwY7-v2mqp
    gdrive_use_service_account = true
    gdrive_service_account_json_file_path = cleveland-461918-t2-<your-key>.json
```

Make sure this path is correct relative to your project directory.

---

## 🏋️ Training

> ⚠️ **Required dependency:**
> If you haven't installed [YOLOv8](https://github.com/ultralytics/ultralytics), run:
>
> ```bash
> pip install ultralytics
> ```


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

> ⚠️ **Important for Windows users:**
> To avoid `ModuleNotFoundError: No module named 'src'`, set the Python path before training:
>
> ```bash
> set PYTHONPATH=.
> python src/train.py
> ```


Training logs, plots, and weights will be saved to `runs/train/exp1/`.

---

## 🔍 Inference

### 📦 Required Dependencies

If you haven't installed the inference dependencies yet, run:

```bash
pip install onnxruntime fire python-multipart
```

---

### 💻 ONNX Runtime Inference

Run inference using the exported ONNX model:

```bash
python src/infer.py \
  --model_path=exports/weights/best.onnx \
  --input_dir=data/test/images \
  --output_dir=outputs/
```

The predictions will be saved in the `outputs/` directory.

---

### 🌐 Inference Server (FastAPI)

To run a local inference server via FastAPI:

```bash
python src/serve.py
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
You can upload an image and get predictions via the Swagger UI.

---


## 📦 Model Export

### ✅ Install dependencies (if not already installed)

```bash
pip install mlflow
```

---

### ➡️ Export to ONNX

```bash
python scripts/export_to_onnx.py
```

> ⚠️ This script loads the latest trained model from MLflow and exports it to the `onnx_models/` directory.
> Make sure your training script logs the model to MLflow.

---

### ⚡ Export to TensorRT

```bash
python scripts/build_engine.py  # Builds best.engine from best.onnx
```

> 💡 This assumes you have an ONNX model saved at `onnx_models/model_<RUN_ID>.onnx`.

---

## Tests

Run all tests:

```bash
pytest tests/
```

Includes:

* `test_utils.py`
* `test_export.py`
* `test_infer.py`
* `test_dataloader.py`
* `test_model.py`
* `test_cli_infer.py`
* `test_api.py`

---

## 📊 Logging & Monitoring

* Metrics (loss, mAP, precision, recall) are saved in `runs/train/...`
* Plots saved to `plots/`
* Git commit ID is captured for reproducibility

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

* ✅ Pre-commit hooks (black, isort, flake8, prettier)
* ✅ Hydra configs for training and inference
* ✅ Inference CLI and FastAPI server
* ✅ DVC + GDrive integration
* ✅ ONNX + TensorRT export
* ✅ Full CI workflow via GitHub Actions

---

## 👤 Author

Project for MLOps course @ MIPT (Spring 2025)

**Author:** Vyacheslav Mikholap
**Email:** [mikholap.vv@phystech.edu](mailto:mikholap.vv@phystech.edu)
**GitHub:** [8MVR3](https://github.com/8MVR3)

---

## 📌 License

This project is licensed under the MIT License.
