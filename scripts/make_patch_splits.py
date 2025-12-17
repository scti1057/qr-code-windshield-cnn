from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

"""
Create train/val/test splits from extracted patches.
Example usage:

python3 scripts/make_patch_splits.py \
  --root-all data/patches/all \
  --classes no_qr qr \
  --train 0.80 --val 0.10 --test 0.10 \
  --seed 42 \
  --out data/splits/patches_split_seed42.json
  
"""


REPO_ROOT = Path(__file__).resolve().parents[1]


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    if not folder.exists():
        return files
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def split_list(items: List[Path], train: float, val: float, test: float, seed: int) -> Dict[str, List[Path]]:
    assert abs((train + val + test) - 1.0) < 1e-6, "train+val+test must sum to 1.0"
    items = items[:]  # copy
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n = len(items)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # rest goes to test
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = max(0, n - n_train - n_val)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:n_train + n_val + n_test]
    return {"train": train_items, "val": val_items, "test": test_items}


def main():
    ap = argparse.ArgumentParser(description="Create reproducible train/val/test splits from data/patches/all/<class>/")
    ap.add_argument("--root-all", type=str, default="data/patches/all", help="Root folder containing class subfolders.")
    ap.add_argument("--classes", nargs=2, default=["no_qr", "qr"], help="Two class folder names: negative positive")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/splits/patches_split_seed42.json")
    args = ap.parse_args()

    root_all = Path(args.root_all)
    if not root_all.is_absolute():
        root_all = REPO_ROOT / root_all

    class0, class1 = args.classes[0], args.classes[1]
    dir0 = root_all / class0
    dir1 = root_all / class1

    files0 = list_images(dir0)
    files1 = list_images(dir1)

    if not files0 or not files1:
        raise SystemExit(
            f"Could not find images in:\n  {dir0} ({len(files0)} files)\n  {dir1} ({len(files1)} files)\n"
            "Check your folder names and paths."
        )

    splits0 = split_list(files0, args.train, args.val, args.test, args.seed)
    splits1 = split_list(files1, args.train, args.val, args.test, args.seed)

    # Build unified split lists with labels
    class_to_idx = {class0: 0, class1: 1}
    out: Dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "root_all": relpath(root_all),
        "classes": [class0, class1],
        "class_to_idx": class_to_idx,
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "splits": {"train": [], "val": [], "test": []},
        "counts": {},
    }

    for split in ["train", "val", "test"]:
        rows = []
        for p in splits0[split]:
            rows.append({"path": relpath(p), "y": 0})
        for p in splits1[split]:
            rows.append({"path": relpath(p), "y": 1})
        # shuffle within split for convenience
        rnd = random.Random(args.seed + (0 if split == "train" else (1 if split == "val" else 2)))
        rnd.shuffle(rows)
        out["splits"][split] = rows

    # counts
    def count_split(split_rows):
        c0 = sum(1 for r in split_rows if r["y"] == 0)
        c1 = sum(1 for r in split_rows if r["y"] == 1)
        return {"total": len(split_rows), class0: c0, class1: c1}

    out["counts"] = {
        "train": count_split(out["splits"]["train"]),
        "val": count_split(out["splits"]["val"]),
        "test": count_split(out["splits"]["test"]),
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("[OK] wrote split file:", out_path)
    print("Counts:", json.dumps(out["counts"], indent=2))


if __name__ == "__main__":
    main()