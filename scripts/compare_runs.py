from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]

"""
Compare CNN scratch runs: summary table + overlay plots.
Example usage (all runs in runs/cnn_scratch):
python3 scripts/compare_runs.py --runs-dir runs/cnn_scratch

With filter (only runs whose folder name contains "lr_"):
python3 scripts/compare_runs.py --runs-dir runs/cnn_scratch --filter lr_

Limit number of runs drawn in overlay plots:
python3 scripts/compare_runs.py --runs-dir runs/cnn_scratch --max-lines 10 --no-legend

"""

def read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def summarize_run(run_dir: Path) -> Optional[Dict]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    rows = read_metrics_csv(metrics_path)
    if not rows:
        return None

    # Extract series
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [to_float(r.get("train_loss", "0")) for r in rows]
    val_loss = [to_float(r.get("val_loss", "0")) for r in rows]
    train_acc = [to_float(r.get("train_acc", "0")) for r in rows]
    val_acc = [to_float(r.get("val_acc", "0")) for r in rows]

    # Bests
    best_val_loss = min(val_loss)
    best_val_loss_epoch = epochs[val_loss.index(best_val_loss)]

    best_val_acc = max(val_acc)
    best_val_acc_epoch = epochs[val_acc.index(best_val_acc)]

    final_epoch = epochs[-1]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]

    # Optional final_metrics.json
    final_metrics_path = run_dir / "final_metrics.json"
    test_loss = None
    test_acc = None
    if final_metrics_path.exists():
        try:
            fm = json.loads(final_metrics_path.read_text(encoding="utf-8"))
            test_loss = fm.get("final_test", {}).get("loss", None)
            test_acc = fm.get("final_test", {}).get("acc", None)
        except Exception:
            pass

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "metrics_path": str(metrics_path),
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_acc": best_val_acc,
        "best_val_acc_epoch": best_val_acc_epoch,
        "final_epoch": final_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_yaml_if_exists(path: Path) -> Optional[object]:
    """
    Returns:
      - dict/list if PyYAML is available and parse succeeds
      - raw text string if not parseable / PyYAML missing
      - None if file missing
    """
    txt = read_text_if_exists(path)
    if txt is None:
        return None
    if yaml is None:
        return txt  # fallback: keep full content
    try:
        return yaml.safe_load(txt)
    except Exception:
        return txt  # fallback: keep full content


def write_summary_json(runs: List[Dict], out_json: Path):
    """
    Create a detailed JSON that contains, for each run:
      - config_used.yaml content (parsed if possible)
      - final_metrics.json content
      - full metrics.csv rows (every epoch)
      - plus some derived summary values already computed by summarize_run()
    """
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_runs": len(runs),
        "runs": [],
    }

    for r in runs:
        run_dir = Path(r["run_dir"])

        config_used_path = run_dir / "config_used.yaml"
        final_metrics_path = run_dir / "final_metrics.json"
        metrics_path = run_dir / "metrics.csv"

        run_entry = {
            "run_name": r["run_name"],
            "run_dir": str(run_dir),

            # include the already-computed summary values
            "summary": {
                "best_val_loss": r["best_val_loss"],
                "best_val_loss_epoch": r["best_val_loss_epoch"],
                "best_val_acc": r["best_val_acc"],
                "best_val_acc_epoch": r["best_val_acc_epoch"],
                "final_epoch": r["final_epoch"],
                "final_train_loss": r["final_train_loss"],
                "final_val_loss": r["final_val_loss"],
                "final_train_acc": r["final_train_acc"],
                "final_val_acc": r["final_val_acc"],
                "test_loss": r["test_loss"],
                "test_acc": r["test_acc"],
            },

            # include full inputs/outputs
            "config_used": read_yaml_if_exists(config_used_path),
            "final_metrics": read_json_if_exists(final_metrics_path),

            # full per-epoch rows (exactly what is in metrics.csv)
            "metrics_rows": read_metrics_csv(metrics_path) if metrics_path.exists() else None,
        }

        payload["runs"].append(run_entry)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_runs(runs_dir: Path) -> List[Path]:
    runs = []
    if not runs_dir.exists():
        return runs
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / "metrics.csv").exists():
            runs.append(p)
    return runs


def plot_overlay(
    runs: List[Dict],
    metric_key: str,
    title: str,
    out_path: Path,
    *,
    max_lines: int = 20,
    legend: bool = True,
):
    plt.figure()
    shown = 0
    for r in runs:
        if shown >= max_lines:
            break
        x = r["epochs"]
        y = r[metric_key]
        plt.plot(x, y, label=r["run_name"])
        shown += 1

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(metric_key)
    if legend:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def write_summary_csv(runs: List[Dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "run_name",
            "best_val_loss", "best_val_loss_epoch",
            "best_val_acc", "best_val_acc_epoch",
            "final_epoch",
            "final_train_loss", "final_val_loss",
            "final_train_acc", "final_val_acc",
            "test_loss", "test_acc",
            "run_dir",
        ])
        for r in runs:
            wr.writerow([
                r["run_name"],
                f"{r['best_val_loss']:.6f}", r["best_val_loss_epoch"],
                f"{r['best_val_acc']:.6f}", r["best_val_acc_epoch"],
                r["final_epoch"],
                f"{r['final_train_loss']:.6f}", f"{r['final_val_loss']:.6f}",
                f"{r['final_train_acc']:.6f}", f"{r['final_val_acc']:.6f}",
                "" if r["test_loss"] is None else f"{float(r['test_loss']):.6f}",
                "" if r["test_acc"] is None else f"{float(r['test_acc']):.6f}",
                r["run_dir"],
            ])


def write_summary_md(runs: List[Dict], out_md: Path, sort_by: str):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Run comparison ({sort_by})")
    lines.append("")
    lines.append("| run | best val_loss (epoch) | best val_acc (epoch) | final val_loss | final val_acc | test acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for r in runs:
        test_acc = "" if r["test_acc"] is None else f"{float(r['test_acc']):.3f}"
        lines.append(
            f"| {r['run_name']} | {r['best_val_loss']:.4f} ({r['best_val_loss_epoch']}) | "
            f"{r['best_val_acc']:.4f} ({r['best_val_acc_epoch']}) | "
            f"{r['final_val_loss']:.4f} | {r['final_val_acc']:.4f} | {test_acc} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Compare CNN scratch runs: summary table + overlay plots.")
    ap.add_argument("--runs-dir", type=str, default="runs/cnn_scratch", help="Folder containing run subfolders.")
    ap.add_argument("--out-dir", type=str, default="", help="Where to write comparison outputs (default: <runs-dir>/_compare).")
    ap.add_argument("--filter", type=str, default="", help="Only include runs whose folder name contains this substring.")
    ap.add_argument("--sort-by", type=str, default="best_val_loss", choices=["best_val_loss", "best_val_acc"], help="Sort criterion.")
    ap.add_argument("--max-lines", type=int, default=15, help="Max number of runs to draw in overlay plots.")
    ap.add_argument("--no-legend", action="store_true", help="Disable legend in plots (useful if many runs).")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir

    out_dir = Path(args.out_dir) if args.out_dir else (runs_dir / "_compare")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_runs(runs_dir)
    if args.filter:
        run_dirs = [p for p in run_dirs if args.filter in p.name]

    summaries: List[Dict] = []
    for rd in run_dirs:
        s = summarize_run(rd)
        if s is not None:
            summaries.append(s)

    if not summaries:
        raise SystemExit(f"No runs found in {runs_dir} (or filter removed all).")

    # sort
    if args.sort_by == "best_val_loss":
        summaries.sort(key=lambda r: r["best_val_loss"])
    else:
        summaries.sort(key=lambda r: r["best_val_acc"], reverse=True)

    # write detailed JSON
    write_summary_json(summaries, out_dir / "summary.json")

    # write tables
    write_summary_csv(summaries, out_dir / "summary.csv")
    write_summary_md(summaries, out_dir / "summary.md", sort_by=args.sort_by)

    # overlay plots (val curves are usually what you compare for HP tuning)
    plot_overlay(
        summaries, "val_loss",
        title="Validation loss (overlay)",
        out_path=out_dir / "overlay_val_loss.png",
        max_lines=args.max_lines,
        legend=not args.no_legend,
    )
    plot_overlay(
        summaries, "val_acc",
        title="Validation accuracy (overlay)",
        out_path=out_dir / "overlay_val_acc.png",
        max_lines=args.max_lines,
        legend=not args.no_legend,
    )

    # optional train curves too
    plot_overlay(
        summaries, "train_loss",
        title="Train loss (overlay)",
        out_path=out_dir / "overlay_train_loss.png",
        max_lines=args.max_lines,
        legend=not args.no_legend,
    )
    plot_overlay(
        summaries, "train_acc",
        title="Train accuracy (overlay)",
        out_path=out_dir / "overlay_train_acc.png",
        max_lines=args.max_lines,
        legend=not args.no_legend,
    )

    print("[OK] wrote:", out_dir / "summary.csv")
    print("[OK] wrote:", out_dir / "summary.md")
    print("[OK] plots:", out_dir / "overlay_val_loss.png")
    print("[OK] plots:", out_dir / "overlay_val_acc.png")
    print("[OK] plots:", out_dir / "overlay_train_loss.png")
    print("[OK] plots:", out_dir / "overlay_train_acc.png")
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
