from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

"""
Duplicate patch dataset by saving horizontally flipped versions.
Example usage:

python3 scripts/augment_flip_patches.py \
  --root-all data/patches/all \
  --classes no_qr qr
  
"""


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    files: List[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def flip_and_save(img_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    flipped = cv2.flip(img, 1)  # horizontal mirror
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), flipped)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="Duplicate patch dataset by saving horizontally flipped versions.")
    ap.add_argument("--root-all", type=str, default="data/patches/all", help="Root folder containing class subfolders.")
    ap.add_argument("--classes", nargs=2, default=["no_qr", "qr"], help="Two class folder names: negative positive")
    ap.add_argument("--suffix", type=str, default="__flip", help="Suffix appended to filename stem for flipped copies.")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done, do not write files.")
    args = ap.parse_args()

    root_all = Path(args.root_all)
    if not root_all.is_absolute():
        root_all = REPO_ROOT / root_all

    class0, class1 = args.classes[0], args.classes[1]
    dirs = [root_all / class0, root_all / class1]

    total_written = 0
    total_skipped = 0
    total_failed = 0

    for d in dirs:
        files = list_images(d)
        if not files:
            print(f"[WARN] No images found in: {d}")
            continue

        written = 0
        skipped = 0
        failed = 0

        for p in files:
            out_p = p.with_name(p.stem + args.suffix + p.suffix)
            if out_p.exists():
                skipped += 1
                continue

            if args.dry_run:
                written += 1
                continue

            ok = flip_and_save(p, out_p)
            if ok:
                written += 1
            else:
                failed += 1

        print(f"[OK] {d}: found={len(files)} new_flips={written} skipped_existing={skipped} failed={failed}")

        total_written += written
        total_skipped += skipped
        total_failed += failed

    print("\nDone.")
    print(f"Total new flipped images: {total_written}")
    print(f"Total skipped (already existed): {total_skipped}")
    print(f"Total failed reads/writes: {total_failed}")


if __name__ == "__main__":
    main()
