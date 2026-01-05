from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

"""
Create train/val/test splits from extracted patches:
- GROUPED by source image (prevents leakage: no source appears in multiple splits)
- BALANCED per split (50/50 patches no_qr vs qr within each split)

Example:
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
    files: List[Path] = []
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


def source_id_from_patch(p: Path) -> str:
    """
    Extract source image id from patch filename.
    Expected pattern: <SRC>__<...>.jpg  => source id = <SRC>
    If no '__' found, fallback to stem.
    """
    stem = p.stem  # filename without extension
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def split_counts(n: int, train: float, val: float, test: float) -> Dict[str, int]:
    assert abs((train + val + test) - 1.0) < 1e-6, "train+val+test must sum to 1.0"
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test}


def split_sources(
    items: List[str],
    train: float,
    val: float,
    test: float,
    seed: int,
    ensure_each_split_if_possible: bool = True,
) -> Dict[str, List[str]]:
    items = items[:]
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n = len(items)
    sz = split_counts(n, train, val, test)
    n_train, n_val, n_test = sz["train"], sz["val"], sz["test"]

    # ensure each split has at least 1 source if possible (prevents "0 qr in val/test" on small datasets)
    if ensure_each_split_if_possible and n >= 3:
        if n_val == 0:
            n_val = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_test = max(0, n_test - 1)
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(0, n_val - 1)
        # recompute n_test as remainder
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val : n_train + n_val + n_test]
    return {"train": train_items, "val": val_items, "test": test_items}


def main():
    ap = argparse.ArgumentParser(description="Create grouped train/val/test splits (no source leakage), balanced per split.")
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

    class0, class1 = args.classes[0], args.classes[1]  # class0=no_qr, class1=qr
    dir0 = root_all / class0
    dir1 = root_all / class1

    files_no = list_images(dir0)
    files_qr = list_images(dir1)

    if not files_no or not files_qr:
        raise SystemExit(
            f"Could not find images in:\n  {dir0} ({len(files_no)} files)\n  {dir1} ({len(files_qr)} files)\n"
            "Check your folder names and paths."
        )

    # group patches by source id, per class
    no_by_src: Dict[str, List[Path]] = {}
    qr_by_src: Dict[str, List[Path]] = {}

    for p in files_no:
        sid = source_id_from_patch(p)
        no_by_src.setdefault(sid, []).append(p)

    for p in files_qr:
        sid = source_id_from_patch(p)
        qr_by_src.setdefault(sid, []).append(p)

    all_sources: Set[str] = set(no_by_src.keys()) | set(qr_by_src.keys())
    qr_sources: List[str] = sorted([s for s in all_sources if len(qr_by_src.get(s, [])) > 0])
    no_only_sources: List[str] = sorted([s for s in all_sources if len(qr_by_src.get(s, [])) == 0 and len(no_by_src.get(s, [])) > 0])

    if len(qr_sources) < 3:
        print(f"[WARN] Very few QR sources ({len(qr_sources)}). val/test may be tiny.")

    # 1) split QR-sources across train/val/test
    qr_src_splits = split_sources(qr_sources, args.train, args.val, args.test, seed=args.seed + 10)

    # 2) also split no-only sources across splits (optional but useful negatives)
    noonly_src_splits = split_sources(no_only_sources, args.train, args.val, args.test, seed=args.seed + 20, ensure_each_split_if_possible=False)

    # combined source splits (still no overlap, because qr_sources and no_only_sources are disjoint)
    src_splits: Dict[str, List[str]] = {
        "train": qr_src_splits["train"] + noonly_src_splits["train"],
        "val": qr_src_splits["val"] + noonly_src_splits["val"],
        "test": qr_src_splits["test"] + noonly_src_splits["test"],
    }

    # safety check: no source overlap
    train_set, val_set, test_set = set(src_splits["train"]), set(src_splits["val"]), set(src_splits["test"])
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise SystemExit(f"[BUG] Source overlap detected! train∩val={len(overlap_tv)} train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}")

    # collect patches per split
    def collect_patches(split_sources_list: List[str]) -> Tuple[List[Path], List[Path]]:
        no_list: List[Path] = []
        qr_list: List[Path] = []
        for sid in split_sources_list:
            no_list.extend(no_by_src.get(sid, []))
            qr_list.extend(qr_by_src.get(sid, []))
        return no_list, qr_list

    no_split: Dict[str, List[Path]] = {}
    qr_split: Dict[str, List[Path]] = {}

    for split in ["train", "val", "test"]:
        nolist, qrlist = collect_patches(src_splits[split])
        no_split[split] = nolist
        qr_split[split] = qrlist

        if len(qrlist) == 0:
            raise SystemExit(
                f"[ERROR] Split '{split}' has 0 QR patches. "
                f"QR sources total={len(qr_sources)}. "
                "Increase QR source count or adjust split fractions."
            )

        if len(nolist) == 0:
            raise SystemExit(f"[ERROR] Split '{split}' has 0 no_qr patches.")

    # balance per split (sample without replacement)
    rnd = random.Random(args.seed + 999)
    balanced_no: Dict[str, List[Path]] = {}
    balanced_qr: Dict[str, List[Path]] = {}

    for split in ["train", "val", "test"]:
        qrlist = qr_split[split][:]
        nolist = no_split[split][:]

        # shuffle for deterministic sampling
        random.Random(args.seed + (0 if split == "train" else (1 if split == "val" else 2))).shuffle(qrlist)
        random.Random(args.seed + (10 if split == "train" else (11 if split == "val" else 12))).shuffle(nolist)

        n = min(len(qrlist), len(nolist))
        if n == 0:
            raise SystemExit(f"[ERROR] Cannot balance split '{split}' (n=min(qr,no_qr)=0).")

        balanced_qr[split] = qrlist[:n]
        balanced_no[split] = nolist[:n]

    # Build unified split lists with labels
    class_to_idx = {class0: 0, class1: 1}
    out: Dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "root_all": relpath(root_all),
        "classes": [class0, class1],
        "class_to_idx": class_to_idx,
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "grouping": {
            "group_key": "source_id_from_filename_prefix_before__",
            "num_sources_total": len(all_sources),
            "num_sources_with_qr": len(qr_sources),
            "num_sources_no_only": len(no_only_sources),
            "sources_per_split": {
                "train": len(src_splits["train"]),
                "val": len(src_splits["val"]),
                "test": len(src_splits["test"]),
            },
            "no_source_overlap": True,
        },
        "balancing": {
            "type": "per_split_equal_patch_counts",
            "original_patch_counts": {class0: len(files_no), class1: len(files_qr)},
            "policy": "Split by source first (no leakage), then downsample majority class per split to match minority.",
        },
        "splits": {"train": [], "val": [], "test": []},
        "counts": {},
    }

    for split in ["train", "val", "test"]:
        rows = []
        for p in balanced_no[split]:
            rows.append({"path": relpath(p), "y": 0, "source": source_id_from_patch(p)})
        for p in balanced_qr[split]:
            rows.append({"path": relpath(p), "y": 1, "source": source_id_from_patch(p)})

        # shuffle within split
        rr = random.Random(args.seed + (1000 if split == "train" else (1001 if split == "val" else 1002)))
        rr.shuffle(rows)
        out["splits"][split] = rows

    def count_split(split_rows):
        c0 = sum(1 for r in split_rows if r["y"] == 0)
        c1 = sum(1 for r in split_rows if r["y"] == 1)
        srcs = {r.get("source", "") for r in split_rows}
        return {"total": len(split_rows), class0: c0, class1: c1, "unique_sources": len([s for s in srcs if s])}

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
    print("Patch counts:", json.dumps(out["counts"], indent=2))
    print("Grouping:", json.dumps(out["grouping"], indent=2))


if __name__ == "__main__":
    main()
