from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(p: Path):
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Live plot metrics.csv during training.")
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    ap.add_argument("--interval", type=float, default=2.0, help="Refresh interval (seconds)")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)

    plt.ion()
    fig1 = plt.figure()
    fig2 = plt.figure()

    last_len = 0

    while True:
        rows = read_rows(metrics_path)
        if not rows:
            time.sleep(args.interval)
            continue

        if len(rows) != last_len:
            last_len = len(rows)

            epoch = [int(r["epoch"]) for r in rows]
            train_loss = [float(r["train_loss"]) for r in rows]
            val_loss = [float(r["val_loss"]) for r in rows]
            train_acc = [float(r["train_acc"]) for r in rows]
            val_acc = [float(r["val_acc"]) for r in rows]

            plt.figure(fig1.number)
            plt.clf()
            plt.plot(epoch, train_loss, label="train_loss")
            plt.plot(epoch, val_loss, label="val_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()

            plt.figure(fig2.number)
            plt.clf()
            plt.plot(epoch, train_acc, label="train_acc")
            plt.plot(epoch, val_acc, label="val_acc")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.tight_layout()

            plt.pause(0.001)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
