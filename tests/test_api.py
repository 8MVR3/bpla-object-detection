# tests/test_api.py
import requests


def test_api_predict():
    url = "http://127.0.0.1:8000/predict"
    with open("data/test/images/0000006_02138_d_0000006.jpg", "rb") as img:
        files = {"file": img}
        response = requests.post(url, files=files)
    assert response.status_code == 200
    assert "predictions" in response.json()
