from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


"""
Run multiple CNN trainings sequentially (one after another).

This is useful for overnight hyperparameter sweeps:
- first run: base config only
- next runs: base + exactly one override YAML each

Example usage:

# Use built-in default override list (recommended order):
python3 scripts/run_sweep_sequential.py \
  --base configs/cnn/baseline.yaml

# Provide explicit overrides (in this exact order):
python3 scripts/run_sweep_sequential.py \
  --base configs/cnn/baseline.yaml \
  --overrides \
    configs/cnn/exp_lr_3e-4.yaml \
    configs/cnn/exp_lr_3e-3.yaml \
    configs/cnn/exp_bs_32.yaml \
    configs/cnn/exp_act_leakyrelu.yaml \
    configs/cnn/exp_width_64.yaml \
    configs/cnn/exp_deeper.yaml \
    configs/cnn/exp_loss_focal.yaml \
    configs/cnn/exp_opt_adamw.yaml \
    configs/cnn/exp_opt_sgd.yaml

# Dry-run (print commands only):
python3 scripts/run_sweep_sequential.py --dry-run
"""

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_OVERRIDES = [
    "configs/cnn/exp_lr_3e-4.yaml",
    "configs/cnn/exp_lr_3e-3.yaml",
    "configs/cnn/exp_bs_32.yaml",
    "configs/cnn/exp_act_leakyrelu.yaml",
    "configs/cnn/exp_width_64.yaml",
    "configs/cnn/exp_deeper.yaml",
    "configs/cnn/exp_loss_focal.yaml",
    "configs/cnn/exp_opt_adamw.yaml",  # recommended: optimizer only
    "configs/cnn/exp_opt_sgd.yaml",    # recommended: optimizer only (LR/WD same as baseline)
]


def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def resolve_repo_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return int(proc.returncode)


def main():
    ap = argparse.ArgumentParser(description="Sequential sweep runner for train_cnn_from_scratch.py")
    ap.add_argument(
        "--base",
        type=str,
        default="configs/cnn/baseline.yaml",
        help="Base YAML/JSON config (default: configs/cnn/baseline.yaml)",
    )
    ap.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Override YAML/JSON configs (space-separated). If omitted, uses an internal recommended list.",
    )
    ap.add_argument(
        "--train-script",
        type=str,
        default="scripts/train_cnn_from_scratch.py",
        help="Training script path (default: scripts/train_cnn_from_scratch.py)",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    ap.add_argument(
        "--sweep-dir",
        type=str,
        default="runs/sweeps",
        help="Where to store sweep logs (default: runs/sweeps)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute.",
    )
    args = ap.parse_args()

    base_cfg = resolve_repo_path(args.base)
    train_script = resolve_repo_path(args.train_script)
    sweep_root = resolve_repo_path(args.sweep_dir)

    if not base_cfg.exists():
        raise SystemExit(f"[ERROR] Base config not found: {base_cfg}")
    if not train_script.exists():
        raise SystemExit(f"[ERROR] Training script not found: {train_script}")

    overrides = args.overrides if args.overrides is not None else DEFAULT_OVERRIDES
    override_paths: List[Path] = []
    for ov in overrides:
        p = resolve_repo_path(ov)
        if not p.exists():
            print(f"[WARN] Override not found, skipping: {p}")
            continue
        override_paths.append(p)

    sweep_id = f"{now_tag()}_cnn_sweep"
    sweep_dir = sweep_root / sweep_id
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = sweep_dir / "sweep_summary.txt"
    if not args.dry_run:
        summary_path.write_text(
            "\n".join(
                [
                    f"Sweep: {sweep_id}",
                    f"Base: {base_cfg}",
                    "Overrides:",
                    *[f"  - {p}" for p in override_paths],
                    "",
                    "Results:",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    print(f"[INFO] Sweep dir: {sweep_dir}")
    print(f"[INFO] Base config: {base_cfg}")
    print(f"[INFO] #overrides: {len(override_paths)}")
    if args.dry_run:
        print("[INFO] DRY RUN enabled (no commands executed)")

    # --- 0) Baseline run ---
    baseline_cmd = [
        args.python,
        str(train_script),
        "--base",
        str(base_cfg),
    ]
    rc = run_cmd(baseline_cmd, logs_dir / "00_baseline.log", args.dry_run)
    if not args.dry_run:
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(f"baseline\treturncode={rc}\tcmd={' '.join(baseline_cmd)}\n")
    if rc != 0:
        print(f"[ERROR] Baseline failed (return code {rc}). Stopping sweep.")
        raise SystemExit(rc)

    # --- 1) One-override runs ---
    for i, ov_path in enumerate(override_paths, start=1):
        tag = f"{i:02d}_{ov_path.stem}"
        cmd = [
            args.python,
            str(train_script),
            "--base",
            str(base_cfg),
            "--override",
            str(ov_path),
        ]
        rc = run_cmd(cmd, logs_dir / f"{tag}.log", args.dry_run)
        if not args.dry_run:
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(f"{ov_path.name}\treturncode={rc}\tcmd={' '.join(cmd)}\n")

        if rc != 0:
            print(f"[WARN] Run failed for override {ov_path.name} (return code {rc}). Continuing...")

    print("\n[OK] Sweep finished.")
    print("Sweep dir:", sweep_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
