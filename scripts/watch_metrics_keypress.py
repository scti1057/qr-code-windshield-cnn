from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt

"""
Watch and plot metrics.csv on keypress.
Usage:

python3 scripts/watch_metrics_keypress.py \
  --metrics runs/cnn_scratch/2026-01-11_234202_transfer_resnet18/metrics.csv

"""


def read_rows(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def plot_from_rows(fig_loss, fig_acc, rows):
    if not rows:
        return

    epoch = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    # Loss figure
    fig_loss.clf()
    ax1 = fig_loss.add_subplot(111)
    ax1.plot(epoch, train_loss, label="train_loss")
    ax1.plot(epoch, val_loss, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    fig_loss.tight_layout()
    fig_loss.canvas.draw_idle()

    # Acc figure
    fig_acc.clf()
    ax2 = fig_acc.add_subplot(111)
    ax2.plot(epoch, train_acc, label="train_acc")
    ax2.plot(epoch, val_acc, label="val_acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    fig_acc.tight_layout()
    fig_acc.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description="Plot metrics.csv on keypress (r=refresh, q/esc=quit).")
    ap.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)

    fig_loss = plt.figure()
    fig_acc = plt.figure()

    help_text = (
        "Keys:\n"
        "  r = refresh plots from metrics.csv\n"
        "  q or ESC = quit\n"
    )
    print(help_text)

    state = {"last_len": 0}

    def refresh():
        rows = read_rows(metrics_path)
        if not rows:
            print(f"[WARN] metrics not found or empty: {metrics_path}")
            return
        n = len(rows)
        if n == state["last_len"]:
            print(f"[OK] no change (rows={n})")
        else:
            print(f"[OK] refreshed (rows={n}, +{n - state['last_len']})")
            state["last_len"] = n
        plot_from_rows(fig_loss, fig_acc, rows)

    def on_key(event):
        k = (event.key or "").lower()
        if k == "r":
            refresh()
        elif k in ("q", "escape"):
            plt.close("all")

    # attach key handler to both windows
    fig_loss.canvas.mpl_connect("key_press_event", on_key)
    fig_acc.canvas.mpl_connect("key_press_event", on_key)

    # optional: initial refresh once
    refresh()

    plt.show()  # blocks, no polling


if __name__ == "__main__":
    main()
