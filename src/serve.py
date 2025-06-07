# src/serve.py

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_PATH = Path("exports/weights/best.onnx")
ORT_SESSION = ort.InferenceSession(str(MODEL_PATH))


def preprocess_image(file: UploadFile, img_size=640):
    """Preprocess uploaded image into ONNX tensor"""
    image = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # BCHW
    return image


def postprocess_output(output):
    """Convert raw output to list of boxes (stub version)"""
    # output: [1, num_detections, 85] - YOLOv8 default
    predictions = output[0]  # single batch
    results = []
    for pred in predictions:
        for det in pred:
            conf = det[4]
            if conf > 0.3:
                x1, y1, x2, y2 = det[:4].tolist()
                cls = int(det[5])
                results.append(
                    {"bbox": [x1, y1, x2, y2], "confidence": float(conf), "class": cls}
                )
    return results


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        input_tensor = preprocess_image(file)
        input_name = ORT_SESSION.get_inputs()[0].name
        output = ORT_SESSION.run(None, {input_name: input_tensor})
        results = postprocess_output(output)
        return {"predictions": results}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=True)
