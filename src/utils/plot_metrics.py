import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(csv_path, save_dir):
    df = pd.read_csv(csv_path)

    # График 1 — Losses
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "losses.png"))
    plt.close()

    # График 2 — Metrics
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics.png"))
    plt.close()
