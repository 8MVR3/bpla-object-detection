#!/bin/bash

# Скрипт для сборки TensorRT-инференс-движка из ONNX-модели
# Требует установленный TensorRT и доступную утилиту trtexec

ONNX_MODEL="exports/weights/best.onnx"
ENGINE_OUTPUT="exports/weights/best.engine"

if ! command -v trtexec &> /dev/null
then
    echo "[!] trtexec не найден. Убедитесь, что TensorRT установлен и trtexec доступен в PATH."
    exit 1
fi

trtexec \
    --onnx=$ONNX_MODEL \
    --saveEngine=$ENGINE_OUTPUT \
    --explicitBatch \
    --workspace=1024 \
    --fp16

if [ $? -eq 0 ]; then
    echo "[✓] Engine успешно сохранён в $ENGINE_OUTPUT"
else
    echo "[✗] Ошибка при сборке engine"
fi
