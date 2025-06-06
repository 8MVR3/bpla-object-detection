# BPLA Object Detection with Prompting

## 📌 Описание проекта

Проект реализует систему автоматического обнаружения объектов на изображениях и видео с беспилотных летательных аппаратов (БЛА). Мы используем мультимодальный подход, объединяя детектор объектов (YOLOv8) с возможностью адаптации к условиям с помощью автоматического генерации промптов.

Основная цель — обеспечить надёжный MLOps-пайплайн: от подготовки данных до развёртывания модели, с учётом требований к переносимости, воспроизводимости и мониторингу.

---

## ⚙️ Установка и настройка окружения

### 📁 Клонируйте репозиторий

```bash
git clone https://github.com/8MVR3/bpla-object-detection.git
cd bpla-object-detection
```

### 🐍 Установите зависимости

**С помощью Poetry:**

```bash
poetry install
poetry shell
```

**Или через pip:**

```bash
pip install -r requirements.txt
```

### ✅ Установите git-хуки

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🏋️‍♂️ Обучение модели

### 🔧 Настройка

Параметры обучения указываются в `configs/train.yaml`, например:

```yaml
epochs: 20
use_gpu: true
data:
    _target_: src.data.DataModule
    data_dir: data/
    batch_size: 32

model:
    _target_: src.models.DetectionModel
    lr: 0.001
```

### 🚀 Запуск обучения

```bash
python src/train.py
```

Hydra автоматически подгрузит конфигурации из `configs/`.

---

## 🔍 Инференс (предсказания)

### 🔧 Настройка

Укажите путь до модели и данных в `configs/infer.yaml`, например:

```yaml
model_path: models/yolov8.onnx
input_data: data/test_images/
```

### 🚀 Запуск инференса

```bash
python src/infer.py
```

Результаты будут сохранены в папке `outputs/`.

---

## 🧠 Экспорт модели

### ➡️ В ONNX

```bash
python scripts/export_onnx.py --model models/best.pt --output models/model.onnx
```

### ➡️ В TensorRT (опционально)

```bash
bash scripts/build_tensorrt.sh
```

---

## 📊 Логирование и метрики

Используется MLflow для трекинга:

-   метрик (mAP, IoU, loss)
-   гиперпараметров
-   модели и конфигурации

По умолчанию логгер сохраняет данные на `http://127.0.0.1:8080`. MLflow можно запустить:

```bash
mlflow ui --port 8080
```

---

## 🗂 Управление данными (DVC)

Данные управляются с помощью DVC. Чтобы загрузить и синхронизировать данные:

```bash
dvc pull           # загрузка данных из удалённого хранилища
dvc add data/      # добавить новые данные
dvc push           # выгрузить изменения
```

Рекомендуется настроить `remote` в `.dvc/config`.

---

## 🗃 Структура проекта

```
bpla-object-detection/
├── configs/         # Конфиги Hydra
├── data/            # Данные (под управлением DVC)
├── models/          # Сохранённые модели (.pt, .onnx и т.д.)
├── notebooks/       # Jupyter-ноутбуки (для исследований)
├── outputs/         # Результаты инференса
├── plots/           # Визуализация метрик
├── scripts/         # Bash-скрипты (экспорт, запуск)
├── src/             # Основной код проекта
│   ├── data/        # DataModule для PyTorch Lightning
│   ├── models/      # Архитектура и логика модели
│   ├── utils/       # Вспомогательные утилиты
│   ├── train.py     # Тренировка модели
│   ├── infer.py     # Инференс
│   └── ...
├── tests/           # Юнит-тесты
├── .gitignore
├── pyproject.toml   # Poetry-зависимости
└── README.md
```

---

## 💡 Полезные советы

-   **Модели (.pt, .onnx, .trt) и данные не должны попадать в Git** — они добавлены в `.gitignore`.
-   **Все конфигурации вынесены в `configs/` и читаются через Hydra** — не хардкодьте параметры в коде.
-   **Используйте MLflow для логирования экспериментов и мониторинга качества.**
-   **Все тяжёлые файлы (.pt/.onnx/.dvc) загружаются/выгружаются через DVC**.
-   **Инструкции по развёртыванию API-сервера будут добавлены позже (FastAPI/MLflow/Triton).**

---

## 👤 Автор

Проект выполнен в рамках курса **MLOps, МФТИ (весна 2025)**
**Автор:** Вячеслав Михолап
**Email:** [mikholap.vv@phystech.edu](mailto:mikholap.vv@phystech.edu)
**GitHub:** [8MVR3](https://github.com/8MVR3)
