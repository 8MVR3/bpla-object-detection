import os
import tempfile

import pandas as pd

from src.utils.plot_metrics import plot_results
from src.utils.utils import get_git_commit_id


def test_get_git_commit_id():
    commit = get_git_commit_id()
    assert isinstance(commit, str)
    assert len(commit) >= 7  # Обычно короткий hash


def test_plot_results_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "results.csv")
        # Подготовим dummy CSV
        df = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "train/box_loss": [1.0, 0.8, 0.6],
                "train/cls_loss": [0.5, 0.4, 0.3],
                "train/dfl_loss": [0.3, 0.2, 0.1],
                "metrics/mAP50(B)": [0.2, 0.3, 0.4],
                "metrics/mAP50-95(B)": [0.1, 0.2, 0.3],
                "metrics/precision(B)": [0.9, 0.85, 0.87],
                "metrics/recall(B)": [0.88, 0.86, 0.89],  # ← ДОБАВЬ ЭТО
            }
        )

        df.to_csv(csv_path, index=False)

        plot_results(csv_path, tmpdir)
        # Проверим, что график создан
        assert any(f.endswith(".png") for f in os.listdir(tmpdir))
