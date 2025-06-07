# 🛰️ BPLA Object Detection with Prompting

## 📌 Описание проекта

Проект реализует систему автоматического обнаружения объектов на изображениях и видео с беспилотных летательных аппаратов (БЛА). Используется YOLOv8 и методы генерации промптов для адаптации модели под разные условия.

Цель — создать надёжный, переносимый и воспроизводимый MLOps-пайплайн с логированием, управлением версиями данных и возможностью развёртывания модели.

---

## ⚙️ Установка и настройка окружения

### 📁 Клонирование репозитория

```bash
git clone https://github.com/8MVR3/bpla-object-detection.git
cd bpla-object-detection
```

### 🐍 Установка зависимостей

**Через Poetry:**

```bash
poetry install
poetry shell
```

**Через pip:**

```bash
pip install -r requirements.txt
```

### ✅ Установка pre-commit хуков

```bash
pre-commit install
pre-commit run --all-files
```

---

## 🏋️‍♂️ Обучение модели

### 🔧 Конфигурация обучения

Параметры задаются в `configs/train.yaml`, пример:

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

Hydra автоматически загрузит нужные конфигурации.

---

## 🔍 Инференс (ONNX)

### 🔧 Аргументы:

-   `--model_path` — путь к ONNX-модели
-   `--input_dir` — директория с `.jpg` изображениями
-   `--output_dir` — директория для результатов

### 🚀 Пример запуска:

```bash
python src/infer.py \
  --model_path=exports/weights/best.onnx \
  --input_dir=data/test/images \
  --output_dir=outputs/
```

Файлы будут обработаны и сохранены в `outputs/`. Используется `onnxruntime`, без PyTorch.

---

## 🧠 Экспорт модели

### ➡️ Экспорт в ONNX

```bash
python scripts/export_onnx.py --model models/best.pt --output models/model.onnx
```

### ➡️ Экспорт в TensorRT (опционально)

```bash
bash scripts/build_engine.sh
```

Этот скрипт использует `trtexec` для компиляции ONNX в TensorRT engine-файл. Требуется установленный TensorRT и `trtexec` в PATH.

---

## 📊 Логирование и метрики

Используется MLflow для логирования:

-   метрик: mAP, IoU, loss
-   гиперпараметров
-   модели и конфигураций

Запуск интерфейса:

```bash
mlflow ui --port 8080
```

---

## 📦 Управление данными (DVC)

DVC управляет данными и моделью. Команды:

```bash
dvc pull        # загрузить данные
# dvc add data/ # добавить новые данные
# dvc push      # выгрузить изменения
```

Настройте `.dvc/config` и remote (например, Google Drive).

---

## 🗃️ Структура проекта

```
bpla-object-detection/
├── configs/         # Конфигурации Hydra
├── data/            # Датасет (под DVC)
├── models/          # Модели (YOLOv8 .pt, .onnx)
├── notebooks/       # Jupyter-ноутбуки
├── outputs/         # Результаты инференса
├── plots/           # Графики и визуализации
├── scripts/         # Bash-скрипты
│   └── build_engine.sh  # Экспорт в TensorRT
├── src/             # Исходный код
│   ├── data/        # DataModule
│   ├── models/      # Архитектура модели
│   ├── utils/       # Утилиты
│   ├── train.py     # Тренировка
│   ├── infer.py     # ONNX-инференс
├── tests/           # Тесты
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 💡 Рекомендации

-   **Данные и модели не коммитьте в Git** — используйте DVC
-   **Конфиги — только в `configs/` через Hydra**
-   **Ведение экспериментов — через MLflow**
-   **ONNX и TensorRT — опциональны, но полезны для продакшн-инференса**
-   **Документация и юнит-тесты — обязательны к поддержке**

---

## 👤 Автор

Проект создан в рамках курса **MLOps, МФТИ (весна 2025)**
**Автор:** Вячеслав Михолап
**Email:** [mikholap.vv@phystech.edu](mailto:mikholap.vv@phystech.edu)
**GitHub:** [8MVR3](https://github.com/8MVR3)
