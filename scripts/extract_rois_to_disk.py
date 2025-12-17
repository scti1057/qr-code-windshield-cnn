from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

# src-layout import
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from qr_cnn.data.roi_c2f import (
    RoiC2FConfig,
    iter_image_files,
    propose_centers_c2f,
    pad_and_crop,
)


def parse_args():
    p = argparse.ArgumentParser(description="Extract ROI patches (C2F + multi-scale) to disk for manual labeling.")
    p.add_argument("--input", type=str, required=True, help="Folder with images (or single image).")
    p.add_argument("--output", type=str, required=True, help="Output folder to save patches + metadata.csv")

    p.add_argument("--roi-config", type=str, required=True, help="Path to ROI config JSON (from GUI tuner).")

    p.add_argument("--save-debug", action="store_true", help="Save debug overlays with drawn boxes.")
    p.add_argument("--debug-out", type=str, default="", help="Optional debug folder override.")
    return p.parse_args()


def draw_boxes(img_bgr, centers, patch_size: int):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    half = patch_size // 2
    for c in centers:
        cx, cy = int(c["cx"]), int(c["cy"])
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + patch_size, y0 + patch_size
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    out_patches = out_root / "patches"
    out_root.mkdir(parents=True, exist_ok=True)
    out_patches.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.roi_config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"ROI config not found: {cfg_path}")

    # Load config JSON via simple dict->dataclass (robust: missing keys => defaults)
    import json
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = RoiC2FConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if k == "top_k_fine":
            cfg.top_k_fine_each = v

    debug_dir = Path(args.debug_out) if args.debug_out else (out_root / "debug")
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "metadata.csv"
    write_header = not meta_path.exists()

    with meta_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "source_image", "patch_file", "idx",
                "cx", "cy", "score", "win",
                "patch_x0", "patch_y0", "patch_x1", "patch_y1",
            ])

        for img_path in iter_image_files(in_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] skip unreadable: {img_path}")
                continue

            centers = propose_centers_c2f(img, cfg)
            if not centers:
                print(f"[WARN] no centers: {img_path.name}")
                continue

            stem = img_path.stem
            half = cfg.patch_size // 2
            boxes_for_debug = []

            for i, c in enumerate(centers):
                cx, cy = int(c["cx"]), int(c["cy"])
                score = float(c["score"])
                win = int(c["win"])

                patch = pad_and_crop(img, cx, cy, int(cfg.patch_size))

                patch_name = f"{stem}__{i:03d}__cx{cx}_cy{cy}__s{score:.4f}__win{win}.jpg"
                patch_path = out_patches / patch_name
                cv2.imwrite(str(patch_path), patch)

                x0, y0 = cx - half, cy - half
                x1, y1 = x0 + int(cfg.patch_size), y0 + int(cfg.patch_size)

                writer.writerow([
                    str(img_path), str(patch_path), i,
                    cx, cy, score, win,
                    x0, y0, x1, y1
                ])

                boxes_for_debug.append({"cx": cx, "cy": cy})

            if args.save_debug and boxes_for_debug:
                dbg = draw_boxes(img, boxes_for_debug, int(cfg.patch_size))
                cv2.imwrite(str(debug_dir / f"{stem}__rois.jpg"), dbg)

            print(f"[OK] {img_path.name}: saved {len(centers)} patches (cfg top_k_final={cfg.top_k_final})")

    print(f"\nDone.\nPatches: {out_patches}\nMetadata: {meta_path}")
    if args.save_debug:
        print(f"Debug: {debug_dir}")


if __name__ == "__main__":
    main()
