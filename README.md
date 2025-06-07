# BPLA Object Detection with Prompting

## 📌 Описание проекта

Проект реализует систему автоматического обнаружения объектов на изображениях и видео с беспилотных летательных аппаратов (БЛА). Мы используем мультимодальный подход, объединяя детектор объектов (YOLOv8) с возможностью адаптации к условиям с помощью автоматической генерации промптов.

Основная цель — обеспечить надёжный MLOps-пайплайн: от подготовки данных до развёртывания модели, с учётом требований к переносимости, воспроизводимости и мониторингу.

---

## ⚙️ Установка и настройка окружения

### 📁 Клонируйте репозиторий

```bash
git clone https://github.com/8MVR3/bpla-object-detection.git
cd bpla-object-detection
```

### 🐍 Установите зависимости

```bash
poetry install
poetry shell
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

## 🔍 Инференс (ONNX)

### 🔧 Параметры:

-   `--model_path`: путь до `.onnx` модели
-   `--input_dir`: директория с `.jpg` изображениями
-   `--output_dir`: папка для результатов

### 🚀 Пример запуска:

```bash
python src/infer.py \
  --model_path=exports/weights/best.onnx \
  --input_dir=data/test/images \
  --output_dir=outputs/
```

Файлы будут обработаны и сохранены в `outputs/`. Используется `onnxruntime` и `cv2`, без PyTorch.

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

## 📤 DVC: управление данными

Для работы с данными используется DVC + Google Drive.

### 📥 Скачивание данных

```bash
dvc pull
```

### 📤 Загрузка в облако (только для разработчиков)

```bash
dvc push
```

> 🔒 Убедитесь, что `GOOGLE_APPLICATION_CREDENTIALS` указывает на `.json` сервисного аккаунта и не попадает в репозиторий.

---

## 📊 Логирование и метрики

-   Сохраняются графики: `loss`, `mAP`, `precision`, `recall`
-   Фиксируется git commit ID
-   Все графики сохраняются в директорию `plots/`

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
│   ├── infer.py     # ONNX-инференс
│   └── serve.py     # FastAPI сервер
├── tests/           # Юнит-тесты
├── .dvc/            # DVC конфигурация
├── .gitignore
├── pyproject.toml   # Poetry зависимости
└── README.md
```

---

## 🚀 FastAPI Inference Server

После того как модель обучена и экспортирована в ONNX, можно запустить сервер для инференса.

### 📦 Установка зависимостей

```bash
poetry install
```

### 🚀 Запуск сервера

```bash
poetry run python src/serve.py
```

Сервер будет доступен по адресу: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 📤 Отправка запроса

-   Метод: `POST /predict`
-   Формат: `multipart/form-data`
-   Поле: `file` (jpeg изображение)

### ✅ Ответ

```json
{
  "predictions": [
    {
      "bbox": [x, y, w, h],
      "confidence": 42.8,
      "class": 48
    }
  ]
}
```

---

## 👤 Автор

Проект выполнен в рамках курса **MLOps, МФТИ (весна 2025)**
**Автор:** Вячеслав Михолап
**Email:** [mikholap.vv@phystech.edu](mailto:mikholap.vv@phystech.edu)
**GitHub:** [8MVR3](https://github.com/8MVR3)
