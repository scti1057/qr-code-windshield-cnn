from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2

# src-layout import
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

"""
Extract ROI patches (C2F + multi-scale) to disk for manual labeling.
Example usage:

python3 scripts/extract_rois_to_disk.py \
  --input data/raw \
  --output data/interim/roi_dump_rot \
  --roi-config configs/roi_tuner_params.json \
  --save-debug

"""

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

    # Rotation augmentation (default ON)
    p.add_argument("--no-rotate", action="store_true", help="Disable saving rotated patches.")
    p.add_argument("--rot-prob", type=float, default=1.0, help="Probability to save a rotated version per ROI (default: 1.0).")
    p.add_argument("--rot-deg", type=float, default=20.0, help="Max absolute rotation in degrees (default: 20).")
    p.add_argument("--rot-min-deg", type=float, default=5.0, help="Minimum abs rotation (avoid near-zero) (default: 5).")
    p.add_argument("--rot-safety-px", type=int, default=3, help="Extra safety padding (px) in feasibility check (default: 3).")

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


def _fmt_angle_for_name(deg: float) -> str:
    # filesystem friendly: rotp12p3 / rotm7p5
    sign = "p" if deg >= 0 else "m"
    s = f"{abs(deg):.1f}".replace(".", "p")
    return f"rot{sign}{s}"


def _sample_angle(max_deg: float, min_deg: float) -> float | None:
    max_deg = float(max_deg)
    min_deg = float(min_deg)
    if max_deg <= 0:
        return None
    min_deg = max(0.0, min(min_deg, max_deg))

    for _ in range(12):
        a = random.uniform(-max_deg, max_deg)
        if abs(a) >= min_deg:
            return a

    # fallback
    return max_deg if random.random() < 0.5 else -max_deg


def _rotation_needed_side(patch_size: int, angle_deg: float, safety_px: int) -> int:
    """
    Needed local crop side length so that rotating by angle_deg and then center-cropping
    patch_size x patch_size does NOT require pixels outside the local crop.

    For a square rotated by theta, the axis-aligned bounding box side is:
      patch_size * (|cos| + |sin|)

    We add some safety pixels.
    """
    theta = math.radians(float(angle_deg))
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    needed = int(math.ceil(float(patch_size) * (c + s))) + 2 * int(safety_px)
    needed = max(needed, patch_size)

    # force odd size for clean centering
    if needed % 2 == 0:
        needed += 1
    return needed


def _can_rotate_at(img_w: int, img_h: int, cx: int, cy: int, needed_side: int) -> bool:
    half = needed_side // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + needed_side
    y1 = y0 + needed_side
    return (x0 >= 0) and (y0 >= 0) and (x1 <= img_w) and (y1 <= img_h)


def _make_rotated_patch(img_bgr, cx: int, cy: int, patch_size: int, angle_deg: float, needed_side: int) -> tuple[bool, any]:
    """
    Crop a local window of size needed_side around (cx,cy), rotate it, then center-crop patch_size.
    Uses BORDER_CONSTANT (black). If feasibility check is correct, the final crop should not include black corners.
    """
    h, w = img_bgr.shape[:2]
    if not _can_rotate_at(w, h, cx, cy, needed_side):
        return False, None

    half_big = needed_side // 2
    x0 = cx - half_big
    y0 = cy - half_big
    x1 = x0 + needed_side
    y1 = y0 + needed_side

    big = img_bgr[y0:y1, x0:x1]
    if big.shape[0] != needed_side or big.shape[1] != needed_side:
        return False, None

    center = (needed_side / 2.0, needed_side / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

    rot_big = cv2.warpAffine(
        big,
        M,
        (needed_side, needed_side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # center crop patch_size
    start = (needed_side - patch_size) // 2
    rot_patch = rot_big[start:start + patch_size, start:start + patch_size]
    if rot_patch.shape[0] != patch_size or rot_patch.shape[1] != patch_size:
        return False, None

    return True, rot_patch


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
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = RoiC2FConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if k == "top_k_fine":
            cfg.top_k_fine_each = v

    patch_size = int(cfg.patch_size)

    debug_dir = Path(args.debug_out) if args.debug_out else (out_root / "debug")
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # metadata: if an old metadata.csv exists without new columns, write metadata_v2.csv instead
    meta_path = out_root / "metadata.csv"
    if meta_path.exists():
        try:
            first = meta_path.read_text(encoding="utf-8").splitlines()[0]
            if "aug_type" not in first or "angle_deg" not in first:
                meta_path = out_root / "metadata_v2.csv"
        except Exception:
            meta_path = out_root / "metadata_v2.csv"

    write_header = not meta_path.exists()

    rotate_enabled = not args.no_rotate
    rot_prob = float(args.rot_prob)
    rot_deg = float(args.rot_deg)
    rot_min_deg = float(args.rot_min_deg)
    rot_safety = int(args.rot_safety_px)

    with meta_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "source_image", "patch_file", "idx",
                "cx", "cy", "score", "win",
                "patch_x0", "patch_y0", "patch_x1", "patch_y1",
                "aug_type", "angle_deg",
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
            half = patch_size // 2
            boxes_for_debug = []

            saved_orig = 0
            saved_rot = 0

            H, W = img.shape[:2]

            for i, c in enumerate(centers):
                cx, cy = int(c["cx"]), int(c["cy"])
                score = float(c["score"])
                win = int(c["win"])

                # ORIGINAL patch
                patch = pad_and_crop(img, cx, cy, patch_size)

                patch_name = f"{stem}__{i:03d}__orig__cx{cx}_cy{cy}__s{score:.4f}__win{win}.jpg"
                patch_path = out_patches / patch_name
                cv2.imwrite(str(patch_path), patch)
                saved_orig += 1

                x0, y0 = cx - half, cy - half
                x1, y1 = x0 + patch_size, y0 + patch_size

                writer.writerow([
                    str(img_path), str(patch_path), i,
                    cx, cy, score, win,
                    x0, y0, x1, y1,
                    "orig", 0.0,
                ])

                boxes_for_debug.append({"cx": cx, "cy": cy})

                # ROTATED patch (optional)
                if rotate_enabled and rot_prob > 0 and random.random() < rot_prob:
                    ang = _sample_angle(rot_deg, rot_min_deg)
                    if ang is not None:
                        needed = _rotation_needed_side(patch_size, ang, rot_safety)
                        if _can_rotate_at(W, H, cx, cy, needed):
                            ok, rot_patch = _make_rotated_patch(img, cx, cy, patch_size, ang, needed)
                            if ok and rot_patch is not None:
                                ang_tag = _fmt_angle_for_name(ang)
                                rot_name = f"{stem}__{i:03d}__{ang_tag}__cx{cx}_cy{cy}__s{score:.4f}__win{win}.jpg"
                                rot_path = out_patches / rot_name
                                cv2.imwrite(str(rot_path), rot_patch)
                                saved_rot += 1

                                writer.writerow([
                                    str(img_path), str(rot_path), i,
                                    cx, cy, score, win,
                                    x0, y0, x1, y1,
                                    "rot", float(ang),
                                ])
                        # else: too close to border -> skip rotation

            if args.save_debug and boxes_for_debug:
                dbg = draw_boxes(img, boxes_for_debug, patch_size)
                cv2.imwrite(str(debug_dir / f"{stem}__rois.jpg"), dbg)

            print(
                f"[OK] {img_path.name}: saved {saved_orig} orig"
                + (f" + {saved_rot} rot" if rotate_enabled else "")
                + f" (cfg top_k_final={cfg.top_k_final}, patch_size={patch_size})"
            )

    print(f"\nDone.\nPatches: {out_patches}\nMetadata: {meta_path}")
    if args.save_debug:
        print(f"Debug: {debug_dir}")


if __name__ == "__main__":
    main()
