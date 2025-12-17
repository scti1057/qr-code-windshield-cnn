from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

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

# Torch is required for prediction
try:
    import torch
except Exception as e:
    torch = None  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Run ROI->CNN classification on a folder of images.")
    p.add_argument("--input", type=str, required=True, help="Folder with images (or single image).")
    p.add_argument("--roi-config", type=str, required=True, help="ROI config JSON (from ROI tuner GUI).")

    p.add_argument("--model", type=str, required=True, help="TorchScript model path (.pt / .ts).")
    p.add_argument("--output", type=str, required=True, help="Output CSV path (per-image summary).")
    p.add_argument("--output-patches", type=str, default="", help="Optional CSV path (per-patch results).")

    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5, help="Image is positive if any patch prob >= threshold.")
    p.add_argument("--early-exit", action="store_true", help="Stop evaluating patches once threshold is reached.")
    p.add_argument("--max-patches", type=int, default=0, help="Cap number of patches per image (0 => use cfg.top_k_final).")

    # preprocessing / normalization
    p.add_argument("--mean", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="RGB mean for normalization.")
    p.add_argument("--std", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="RGB std for normalization.")

    # debug outputs
    p.add_argument("--save-debug", action="store_true", help="Save debug overlay images with top patches.")
    p.add_argument("--debug-dir", type=str, default="", help="Debug output folder (default: <output_dir>/debug_pred).")
    p.add_argument("--debug-top", type=int, default=5, help="How many top patches to draw on debug overlay.")
    p.add_argument("--save-positive-patches", action="store_true", help="Save patches that exceed threshold.")
    p.add_argument("--pos-patches-dir", type=str, default="", help="Folder for saved positive patches.")

    return p.parse_args()


def load_roi_config(path: Path) -> RoiC2FConfig:
    cfg = RoiC2FConfig()
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if k == "top_k_fine":  # backward compat
            cfg.top_k_fine_each = v
    return cfg


def load_torchscript_model(model_path: Path, device: str):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch or run this on an environment with torch.")
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    m = torch.jit.load(str(model_path), map_location=dev)
    m.eval()
    return m, dev


def patches_to_tensor(
    patches_bgr: List[np.ndarray],
    mean: List[float],
    std: List[float],
    device,
):
    """
    patches_bgr: list of HxWx3 uint8 BGR patches (already 265x265)
    returns: torch float tensor (N,3,H,W) normalized
    """
    # BGR -> RGB, uint8 -> float32 0..1, HWC -> CHW
    arr = []
    for p in patches_bgr:
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        arr.append(x)
    x = np.stack(arr, axis=0)  # NCHW

    x_t = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    x_t = (x_t - mean_t) / (std_t + 1e-12)
    return x_t


def probs_from_model_output(y):
    """
    Accepts output of various shapes:
    - (N,) or (N,1): sigmoid
    - (N,2): softmax -> take class1
    """
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, dict):
        # try common keys
        for k in ["logits", "output", "pred", "y"]:
            if k in y:
                y = y[k]
                break

    if y.ndim == 1:
        # logits
        return torch.sigmoid(y).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 1:
        return torch.sigmoid(y[:, 0]).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 2:
        p = torch.softmax(y, dim=1)[:, 1]
        return p.detach().cpu().numpy()

    # fallback: flatten and sigmoid
    yy = y.reshape(y.shape[0], -1)[:, 0]
    return torch.sigmoid(yy).detach().cpu().numpy()


def draw_debug_overlay(img_bgr: np.ndarray, centers: List[Dict], probs: List[float], patch_size: int, top_n: int):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    half = patch_size // 2

    # draw in descending prob
    idxs = list(range(len(centers)))
    idxs.sort(key=lambda i: probs[i], reverse=True)
    idxs = idxs[: min(top_n, len(idxs))]

    for rank, i in enumerate(idxs):
        c = centers[i]
        cx, cy = int(c["cx"]), int(c["cy"])
        p = float(probs[i])

        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + patch_size, y0 + patch_size
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)

        color = (0, 255, 0) if rank == 0 else (0, 255, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out, f"p={p:.3f}",
            (x0 + 4, max(15, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
    return out


def main():
    args = parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for this script. Please install torch.")

    in_path = Path(args.input)

    roi_cfg_path = Path(args.roi_config)
    if not roi_cfg_path.is_absolute():
        roi_cfg_path = REPO_ROOT / roi_cfg_path
    cfg = load_roi_config(roi_cfg_path)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model, device = load_torchscript_model(model_path, args.device)

    out_csv = Path(args.output)
    if not out_csv.is_absolute():
        out_csv = REPO_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    patch_csv = None
    if args.output_patches:
        patch_csv = Path(args.output_patches)
        if not patch_csv.is_absolute():
            patch_csv = REPO_ROOT / patch_csv
        patch_csv.parent.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if args.save_debug:
        debug_dir = Path(args.debug_dir) if args.debug_dir else (out_csv.parent / "debug_pred")
        if not debug_dir.is_absolute():
            debug_dir = REPO_ROOT / debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

    pos_dir = None
    if args.save_positive_patches:
        pos_dir = Path(args.pos_patches_dir) if args.pos_patches_dir else (out_csv.parent / "positive_patches")
        if not pos_dir.is_absolute():
            pos_dir = REPO_ROOT / pos_dir
        pos_dir.mkdir(parents=True, exist_ok=True)

    max_patches = args.max_patches if args.max_patches and args.max_patches > 0 else int(cfg.top_k_final)

    # CSV writers
    out_f = out_csv.open("w", newline="", encoding="utf-8")
    out_wr = csv.writer(out_f)
    out_wr.writerow(["source_image", "n_patches", "max_prob", "is_qr", "threshold"])

    patch_f = None
    patch_wr = None
    if patch_csv:
        patch_f = patch_csv.open("w", newline="", encoding="utf-8")
        patch_wr = csv.writer(patch_f)
        patch_wr.writerow(["source_image", "idx", "cx", "cy", "roi_score", "win", "prob_qr", "threshold", "is_qr_patch"])

    try:
        for img_path in iter_image_files(in_path):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] skip unreadable: {img_path}")
                continue

            centers = propose_centers_c2f(img, cfg)
            centers = centers[:max_patches]

            if not centers:
                out_wr.writerow([str(img_path), 0, 0.0, 0, args.threshold])
                print(f"[WARN] no centers: {img_path.name}")
                continue

            # evaluate patches in batches
            probs_all: List[float] = []
            patches_all: List[np.ndarray] = []
            meta_all: List[Dict] = []

            # prepare patches list first (cheap) so we can save positive patches later
            for c in centers:
                cx, cy = int(c["cx"]), int(c["cy"])
                patch = pad_and_crop(img, cx, cy, int(cfg.patch_size))
                patches_all.append(patch)
                meta_all.append(c)

            is_pos_image = False
            max_prob = 0.0

            bs = max(1, int(args.batch_size))
            for start in range(0, len(patches_all), bs):
                batch_patches = patches_all[start:start + bs]
                x = patches_to_tensor(batch_patches, args.mean, args.std, device)

                with torch.no_grad():
                    y = model(x)
                probs = probs_from_model_output(y).tolist()

                # append
                probs_all.extend([float(p) for p in probs])

                # early exit if requested
                if args.early_exit:
                    batch_max = max(probs) if probs else 0.0
                    if batch_max >= args.threshold:
                        is_pos_image = True
                        max_prob = max(max_prob, float(batch_max))
                        # fill remaining probs with -1 for alignment? (we stop early)
                        # We'll stop early and ignore remaining patches.
                        # Keep only already computed probs; crop meta/patch lists accordingly.
                        cut = start + len(batch_patches)
                        centers = centers[:cut]
                        patches_all = patches_all[:cut]
                        meta_all = meta_all[:cut]
                        break

                if probs:
                    max_prob = max(max_prob, float(max(probs)))

            # if not early exited, decide
            if not is_pos_image:
                is_pos_image = (max_prob >= args.threshold)

            out_wr.writerow([str(img_path), len(centers), f"{max_prob:.6f}", int(is_pos_image), args.threshold])

            # per-patch csv
            if patch_wr is not None:
                for i, (c, p) in enumerate(zip(meta_all, probs_all)):
                    roi_score = float(c.get("score", 0.0))
                    win = int(c.get("win", 0))
                    is_qr_patch = int(float(p) >= args.threshold)
                    patch_wr.writerow([str(img_path), i, int(c["cx"]), int(c["cy"]), f"{roi_score:.6f}", win, f"{float(p):.6f}", args.threshold, is_qr_patch])

            # debug overlay
            if debug_dir is not None:
                dbg = draw_debug_overlay(img, meta_all, probs_all, int(cfg.patch_size), top_n=int(args.debug_top))
                cv2.imwrite(str(debug_dir / f"{img_path.stem}__pred.jpg"), dbg)

            # save positive patches
            if pos_dir is not None:
                for i, (c, p, patch) in enumerate(zip(meta_all, probs_all, patches_all)):
                    if float(p) >= args.threshold:
                        name = f"{img_path.stem}__{i:03d}__p{float(p):.3f}__cx{int(c['cx'])}_cy{int(c['cy'])}.jpg"
                        cv2.imwrite(str(pos_dir / name), patch)

            print(f"[OK] {img_path.name}: patches={len(centers)} max_prob={max_prob:.3f} -> is_qr={int(is_pos_image)}")

    finally:
        out_f.close()
        if patch_f:
            patch_f.close()

    print(f"\nDone.\nPer-image CSV: {out_csv}")
    if patch_csv:
        print(f"Per-patch CSV: {patch_csv}")
    if debug_dir:
        print(f"Debug overlays: {debug_dir}")
    if pos_dir:
        print(f"Positive patches: {pos_dir}")


if __name__ == "__main__":
    main()
