from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import cv2
import numpy as np


# --------- helpers ----------
def iter_image_files(input_path: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
        return
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def resize_max_dim(gray: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return gray, 1.0
    s = max_dim / float(m)
    out = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def integral_image(x: np.ndarray) -> np.ndarray:
    return cv2.integral(x)


def window_mean(integ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    s = float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
    area = float((x1 - x0) * (y1 - y0))
    return s / (area + 1e-9)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0, inter_x1 - inter_x0)
    ih = max(0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / float(area_a + area_b - inter + 1e-9)


def nms(cands: List[Dict], iou_thresh: float, top_k: int) -> List[Dict]:
    cands = sorted(cands, key=lambda d: d["score"], reverse=True)
    keep: List[Dict] = []
    for c in cands:
        if len(keep) >= top_k:
            break
        ok = True
        for k in keep:
            if iou(c["box_s"], k["box_s"]) > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def make_win_sizes(win_min: int, win_max: int, n_scales: int) -> List[int]:
    win_min = int(max(24, min(768, win_min)))
    win_max = int(max(24, min(1024, win_max)))
    if win_min > win_max:
        win_min, win_max = win_max, win_min
    n_scales = int(max(1, min(8, n_scales)))
    if n_scales == 1:
        return [win_max]
    vals = np.linspace(win_min, win_max, n_scales)
    sizes = sorted({int(round(v)) for v in vals})
    return [max(24, s) for s in sizes]


# --------- scoring (edges + corners + grad + anisotropy) ----------
def compute_score_map(
    gray: np.ndarray,
    *,
    canny_low: int,
    canny_high: int,
    harris_k: float,
    harris_thr_rel: float,
    st_sigma: float,
    w_edges: float,
    w_corners: float,
    w_grad: float,
    w_aniso: float,
) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(g, int(canny_low), int(canny_high))
    edges01 = edges.astype(np.float32) / 255.0

    g32 = g.astype(np.float32) / 255.0
    harris = cv2.cornerHarris(g32, blockSize=2, ksize=3, k=float(harris_k))
    harris = cv2.dilate(harris, None)
    thr = float(harris_thr_rel) * float(harris.max()) if float(harris.max()) > 0 else 0.0
    corners01 = (harris > thr).astype(np.float32)

    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    mag01 = normalize01(mag)

    # rotation-invariant "parallel edge" measure: structure tensor anisotropy
    jxx = dx * dx
    jyy = dy * dy
    jxy = dx * dy

    sigma = float(max(0.1, st_sigma))
    ksize = int(max(3, round(sigma * 6))) | 1
    jxx = cv2.GaussianBlur(jxx, (ksize, ksize), sigma)
    jyy = cv2.GaussianBlur(jyy, (ksize, ksize), sigma)
    jxy = cv2.GaussianBlur(jxy, (ksize, ksize), sigma)

    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)
    aniso = (l1 - l2) / (l1 + l2 + 1e-6)
    aniso01 = normalize01(aniso)

    score = (
        float(w_edges) * edges01 +
        float(w_corners) * corners01 +
        float(w_grad) * mag01 +
        float(w_aniso) * aniso01
    )
    return normalize01(score)


def scan_windows(
    integ: np.ndarray,
    win: int,
    stride: int,
    score_thr: float,
    top_k: int,
) -> List[Dict]:
    h, w = integ.shape[0] - 1, integ.shape[1] - 1
    win = int(win)
    stride = int(max(1, stride))
    half = win // 2

    if win >= min(h, w):
        return []

    cands: List[Dict] = []
    for cy in range(half, h - half, stride):
        for cx in range(half, w - half, stride):
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + win, y0 + win
            m = window_mean(integ, x0, y0, x1, y1)
            if m >= score_thr:
                cands.append({
                    "cx_s": cx, "cy_s": cy,
                    "score": float(m),
                    "win": win,
                    "box_s": (x0, y0, x1, y1),
                })

    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands[:top_k]


def coarse_to_fine(
    score_map: np.ndarray,
    win_sizes: List[int],
    stride_coarse: int,
    stride_fine: int,
    refine_radius: int,
    score_thr: float,
    top_k_coarse: int,
    top_k_fine_each: int,
    iou_thresh: float,
    top_k_final: int,
) -> List[Dict]:
    integ = integral_image(score_map)
    all_cands: List[Dict] = []

    # 1) coarse
    coarse: List[Dict] = []
    for win in win_sizes:
        coarse.extend(scan_windows(integ, win, stride_coarse, score_thr, top_k_coarse))

    coarse = nms(coarse, iou_thresh, top_k=max(300, top_k_coarse * max(1, len(win_sizes))))

    # 2) fine around each coarse
    h, w = score_map.shape[:2]
    rr = int(max(0, refine_radius))

    for c in coarse:
        cx0, cy0 = int(c["cx_s"]), int(c["cy_s"])
        win = int(c["win"])
        half = win // 2

        x_min = max(half, cx0 - rr)
        x_max = min(w - half - 1, cx0 + rr)
        y_min = max(half, cy0 - rr)
        y_max = min(h - half - 1, cy0 + rr)

        local: List[Dict] = []
        for cy in range(y_min, y_max + 1, max(1, int(stride_fine))):
            for cx in range(x_min, x_max + 1, max(1, int(stride_fine))):
                x0, y0 = cx - half, cy - half
                x1, y1 = x0 + win, y0 + win
                m = window_mean(integ, x0, y0, x1, y1)
                if m >= score_thr:
                    local.append({
                        "cx_s": cx, "cy_s": cy,
                        "score": float(m),
                        "win": win,
                        "box_s": (x0, y0, x1, y1),
                    })
        local.sort(key=lambda d: d["score"], reverse=True)
        all_cands.extend(local[:top_k_fine_each])

    all_cands.sort(key=lambda d: d["score"], reverse=True)
    all_cands = nms(all_cands, iou_thresh, top_k_final)
    return all_cands


def pad_and_crop(img: np.ndarray, cx: int, cy: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + patch_size, y0 + patch_size

    h, w = img.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101
        )
        x0 += pad_left; x1 += pad_left
        y0 += pad_top;  y1 += pad_top

    patch = img[y0:y1, x0:x1]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return patch


# --------- config ----------
@dataclass
class RoiC2FConfig:
    patch_size: int = 265
    score_max_dim: int = 1600

    win_min: int = 128
    win_max: int = 320
    n_scales: int = 3

    stride_coarse: int = 96
    stride_fine: int = 24
    refine_radius: int = 80

    top_k_coarse: int = 60
    top_k_fine_each: int = 20
    top_k_final: int = 120

    iou_thresh: float = 0.25
    score_thr: float = 0.35

    w_edges: float = 1.0
    w_corners: float = 0.7
    w_grad: float = 0.6
    w_aniso: float = 0.8

    canny_low: int = 50
    canny_high: int = 150
    harris_k: float = 0.04
    harris_thr_rel: float = 0.01
    st_sigma: float = 2.0


def load_c2f_config(json_path: Path) -> RoiC2FConfig:
    cfg = RoiC2FConfig()
    if not json_path.exists():
        return cfg
    data = json.loads(json_path.read_text(encoding="utf-8"))  # type: ignore[name-defined]
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        # backward compat: top_k_fine -> top_k_fine_each
        if k == "top_k_fine" and hasattr(cfg, "top_k_fine_each"):
            setattr(cfg, "top_k_fine_each", v)
    return cfg


def propose_centers_c2f(img_bgr: np.ndarray, cfg: RoiC2FConfig) -> List[Dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_s, s = resize_max_dim(gray, int(cfg.score_max_dim))

    score_map = compute_score_map(
        gray_s,
        canny_low=int(cfg.canny_low),
        canny_high=int(cfg.canny_high),
        harris_k=float(cfg.harris_k),
        harris_thr_rel=float(cfg.harris_thr_rel),
        st_sigma=float(cfg.st_sigma),
        w_edges=float(cfg.w_edges),
        w_corners=float(cfg.w_corners),
        w_grad=float(cfg.w_grad),
        w_aniso=float(cfg.w_aniso),
    )

    wins = make_win_sizes(int(cfg.win_min), int(cfg.win_max), int(cfg.n_scales))

    cands_s = coarse_to_fine(
        score_map=score_map,
        win_sizes=wins,
        stride_coarse=int(cfg.stride_coarse),
        stride_fine=int(cfg.stride_fine),
        refine_radius=int(cfg.refine_radius),
        score_thr=float(cfg.score_thr),
        top_k_coarse=int(cfg.top_k_coarse),
        top_k_fine_each=int(cfg.top_k_fine_each),
        iou_thresh=float(cfg.iou_thresh),
        top_k_final=int(cfg.top_k_final),
    )

    # map to original coords
    out: List[Dict] = []
    for c in cands_s:
        cx = int(round(c["cx_s"] / s))
        cy = int(round(c["cy_s"] / s))
        out.append({
            "cx": cx,
            "cy": cy,
            "score": float(c["score"]),
            "win": int(c["win"]),
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
