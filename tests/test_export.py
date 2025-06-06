from unittest import mock

from src import export_trt


@mock.patch("src.export_trt.YOLO")
def test_export_trt_runs(mock_yolo):
    mock_model = mock_yolo.return_value
    mock_model.export.return_value = "fake.engine"

    export_trt.export_trt("some_model.pt", save_dir="some_dir", imgsz=640)

    mock_yolo.assert_called_once_with("some_model.pt")
    mock_model.export.assert_called_once()
