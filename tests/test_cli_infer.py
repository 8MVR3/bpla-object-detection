import subprocess
from pathlib import Path

import pytest


def test_infer_cli_runs(tmp_path):
    """Проверка запуска infer.py через CLI с ONNX моделью"""

    model_path = Path("exports/weights/best.onnx")
    if not model_path.exists():
        pytest.skip("ONNX model not found: exports/weights/best.onnx")

    input_dir = Path("data/test/images")
    if not input_dir.exists():
        pytest.skip("Input images not found")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    cmd = [
        "poetry",
        "run",
        "python",
        "src/infer.py",
        f"--model_path={model_path}",
        f"--input_dir={input_dir}",
        f"--output_dir={output_dir}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Проверка успешного завершения
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

    # Проверка наличия хотя бы одного выходного файла
    outputs = list(output_dir.glob("*.jpg"))
    assert len(outputs) > 0, "No output images generated"
