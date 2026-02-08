# scripts/qr_rectify_gui_v3.py
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

"""
QR Rectify GUI V3

Prinzip:
- Kandidaten (ROI) über CNN wie in V2
- Plan A: OpenCV QRCodeDetector (wenn klappt -> super)
- Plan B (neu): "mask-first quad fit"
    1) Otsu/adap/etc. -> Binary
    2) Morph: CLOSE (k=3, it=4) + OPEN (it=0) -> used
    3) Beste Kontur -> convex hull
    4) Quad aus approxPolyDP (4 Ecken) sonst minAreaRect
    5) Optional: fitLine pro Seite auf Hull-Punkten -> bessere Ecken
- Kein Hough.

Views:
  1) Detected (full image)
  2) Warp (top view)
  3) Crop + quad
  4) Binary raw
  5) Morph + contour/edges + line-fit debug grid
"""

# ---------- repo / src layout ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from qr_cnn.data.roi_c2f import RoiC2FConfig, iter_image_files, propose_centers_c2f, pad_and_crop

# ---------- torch ----------
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

# ---------- tkinter ----------
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# Config loading
# ============================================================
def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pyyaml. Install via: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_roi_config(path: Path) -> RoiC2FConfig:
    cfg = RoiC2FConfig()
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if k == "top_k_fine":
            cfg.top_k_fine_each = v
    return cfg


# ============================================================
# Model build (scratch + transfer)
# ============================================================
def make_activation(name: str) -> "nn.Module":
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(alpha=1.0, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, act: str, bn: bool):
        super().__init__()
        padding = k // 2
        layers: List[nn.Module] = [nn.Conv2d(cin, cout, kernel_size=k, padding=padding, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(make_activation(act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyQRNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        blocks_per_stage: List[int],
        kernel_size: int,
        activation: str,
        batch_norm: bool,
        dropout: float,
        num_outputs: int = 1,
        global_pool: str = "avg",
    ):
        super().__init__()
        c0 = int(base_channels)
        k = int(kernel_size)
        bn = bool(batch_norm)
        act = str(activation)

        stages: List[nn.Module] = []
        cin = int(in_channels)

        for si, nb in enumerate(blocks_per_stage):
            cout = c0 * (2 ** si)
            blocks: List[nn.Module] = []
            for _ in range(int(nb)):
                blocks.append(ConvBlock(cin, cout, k, act, bn))
                cin = cout
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*blocks))

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if (global_pool or "avg").lower() == "avg" else nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(cin, int(num_outputs))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.head(x)


def build_transfer_model(model_name: str, num_outputs: int = 1, dropout: float = 0.2) -> "nn.Module":
    try:
        import torchvision.models as M
    except Exception as e:
        raise RuntimeError("torchvision is required for transfer models. Install torchvision.") from e

    n = model_name.lower().strip()

    if n == "resnet18":
        model = M.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_f, int(num_outputs)))
        return model

    if n == "efficientnet_b0":
        model = M.efficientnet_b0(weights=None)
        if isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, int(num_outputs))
            for m in model.classifier:
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)
        return model

    if n == "mobilenet_v3_large":
        model = M.mobilenet_v3_large(weights=None)
        if isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, int(num_outputs))
            for m in model.classifier:
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)
        return model

    raise ValueError(f"Unknown transfer model: {model_name}")


@dataclass
class LoadedModel:
    model: "nn.Module"
    device: "torch.device"
    input_size: int
    mean: List[float]
    std: List[float]
    kind: str
    name: str


def pick_device(device_str: str) -> "torch.device":
    ds = (device_str or "auto").lower()
    if ds == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if ds == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if ds == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(ds)


def probs_from_logits(y: "torch.Tensor") -> np.ndarray:
    if y.ndim == 1:
        return torch.sigmoid(y).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 1:
        return torch.sigmoid(y[:, 0]).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 2:
        return torch.softmax(y, dim=1)[:, 1].detach().cpu().numpy()
    yy = y.reshape(y.shape[0], -1)[:, 0]
    return torch.sigmoid(yy).detach().cpu().numpy()


def discover_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    out = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "best.pt").exists():
            out.append(d)
    return out


def load_model_from_run_dir(run_dir: Path, device_str: str) -> LoadedModel:
    device = pick_device(device_str)
    ckpt = torch.load(str(run_dir / "best.pt"), map_location=device)

    cfg: Dict[str, Any] = {}
    for cand in [run_dir / "config_used.yaml", run_dir / "config_used.json"]:
        if cand.exists():
            cfg = load_yaml_or_json(cand)
            break

    input_size = int(((cfg.get("data") or {}).get("img_size", 265)) if cfg else 265)
    pp = (cfg.get("preprocess") or {}) if cfg else {}
    mean = list(pp.get("mean", [0.5, 0.5, 0.5]))
    std = list(pp.get("std", [0.5, 0.5, 0.5]))

    transfer_name = None
    mn = ckpt.get("model_name", None) if isinstance(ckpt, dict) else None
    if isinstance(mn, str) and mn.lower() in {"resnet18", "efficientnet_b0", "mobilenet_v3_large"}:
        transfer_name = mn.lower()

    if transfer_name:
        head_cfg = (cfg.get("model_head") or {}) if cfg else {}
        dropout = float(head_cfg.get("dropout", 0.2))
        num_outputs = int(head_cfg.get("num_outputs", 1))
        model = build_transfer_model(transfer_name, num_outputs=num_outputs, dropout=dropout)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(device).eval()

        if "preprocess" not in cfg:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        input_size = int((cfg.get("data") or {}).get("img_size", 224)) if cfg else 224
        if input_size <= 0:
            input_size = 224

        return LoadedModel(model=model, device=device, input_size=input_size, mean=mean, std=std, kind="transfer", name=run_dir.name)

    mcfg = (cfg.get("model") or {}) if cfg else {}
    model = TinyQRNet(
        in_channels=int(mcfg.get("in_channels", 3)),
        base_channels=int(mcfg.get("base_channels", 32)),
        blocks_per_stage=list(mcfg.get("blocks_per_stage", [2, 2, 2, 2])),
        kernel_size=int(mcfg.get("kernel_size", 3)),
        activation=str(mcfg.get("activation", "relu")),
        batch_norm=bool(mcfg.get("batch_norm", True)),
        dropout=float(mcfg.get("dropout", 0.2)),
        num_outputs=int(mcfg.get("num_outputs", 1)),
        global_pool=str(mcfg.get("global_pool", "avg")),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return LoadedModel(model=model, device=device, input_size=input_size, mean=mean, std=std, kind="scratch", name=run_dir.name)


def patches_to_tensor(
    patches_bgr: List[np.ndarray],
    input_size: int,
    mean: List[float],
    std: List[float],
    device: "torch.device",
) -> "torch.Tensor":
    mean_np = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    arr = []
    for p in patches_bgr:
        if p.shape[0] != input_size or p.shape[1] != input_size:
            p = cv2.resize(p, (input_size, input_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - mean_np) / (std_np + 1e-12)
        x = np.transpose(rgb, (2, 0, 1))
        arr.append(x)

    x = np.stack(arr, axis=0)
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


# ============================================================
# Boxes + merging
# ============================================================
def clamp_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    inter = float(iw * ih)
    area_a = float(max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1))
    area_b = float(max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else (inter / denom)


@dataclass
class PatchBox:
    idx: int
    box: Tuple[int, int, int, int]
    p: float


@dataclass
class MergedROI:
    box: Tuple[int, int, int, int]
    members: List[int]
    score: float


def cluster_boxes_by_iou(boxes: List[PatchBox], iou_thr: float) -> List[MergedROI]:
    n = len(boxes)
    if n == 0:
        return []

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if box_iou(boxes[i].box, boxes[j].box) >= float(iou_thr):
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    merged: List[MergedROI] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        xs0, ys0, xs1, ys1, probs, members = [], [], [], [], [], []
        for k in comp:
            x0, y0, x1, y1 = boxes[k].box
            xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
            probs.append(boxes[k].p)
            members.append(boxes[k].idx)
        union = (min(xs0), min(ys0), max(xs1), max(ys1))
        merged.append(MergedROI(box=union, members=sorted(members), score=float(max(probs)) if probs else 0.0))

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


def expand_box(box: Tuple[int, int, int, int], pad_frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))
    return clamp_box(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)


def crop_bgr(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()


# ============================================================
# Quad utils + Plan A
# ============================================================
def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.stack([tl, tr, br, bl], axis=0)


def quad_area(quad: np.ndarray) -> float:
    q = quad.reshape(-1, 1, 2).astype(np.float32)
    return float(abs(cv2.contourArea(q)))


def side_lengths(quad: np.ndarray) -> Tuple[float, float, float, float]:
    tl, tr, br, bl = quad.reshape(4, 2)
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    return top, right, bottom, left


def is_degenerate_quad(quad: np.ndarray, min_rel_area: float = 0.03) -> bool:
    a = quad_area(quad)
    if a <= 1.0:
        return True
    xs = quad[:, 0]
    ys = quad[:, 1]
    bw = float(xs.max() - xs.min() + 1.0)
    bh = float(ys.max() - ys.min() + 1.0)
    bbox_area = bw * bh
    if bbox_area <= 1.0:
        return True
    if a / bbox_area < float(min_rel_area):
        return True

    top, right, bottom, left = side_lengths(quad)
    smin = min(top, right, bottom, left)
    smax = max(top, right, bottom, left)
    if smin < 8.0 or (smin / (smax + 1e-6)) < 0.07:
        return True
    return False


def quad_has_4_distinct_corners(quad: np.ndarray, min_dist: float = 6.0) -> bool:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(q[i] - q[j])) < float(min_dist):
                return False
    return True


def detect_qr_corners_opencv(det: cv2.QRCodeDetector, crop: np.ndarray) -> Optional[np.ndarray]:
    ok, pts = det.detect(crop)
    if not ok or pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        return None
    quad = order_quad(pts)
    if is_degenerate_quad(quad):
        return None
    return quad


# ============================================================
# Plan B V3: mask-first quad fit + optional fitLine refinement
# ============================================================
def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def morph(img: np.ndarray, op: int, k: int, iters: int) -> np.ndarray:
    if int(iters) <= 0:
        return img.copy()
    k = _ensure_odd(int(k))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(img, op, kernel, iterations=int(iters))


def preprocess_variants(crop_bgr_img: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(crop_bgr_img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 50, 160)
    adap = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    _, otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return {"edges": edges, "adap": adap, "otsu": otsu, "otsu_inv": otsu_inv}


@dataclass
class LineABC:
    a: float
    b: float
    c: float


def _canonicalize_line(a: float, b: float, c: float) -> Tuple[float, float, float]:
    n = math.hypot(a, b) + 1e-9
    a /= n
    b /= n
    c /= n
    # canonical sign to avoid flips
    if (a < 0.0) or (abs(a) < 1e-10 and b < 0.0):
        a, b, c = -a, -b, -c
    return a, b, c


def line_from_points(p0: np.ndarray, p1: np.ndarray) -> LineABC:
    x1, y1 = float(p0[0]), float(p0[1])
    x2, y2 = float(p1[0]), float(p1[1])
    dx = x2 - x1
    dy = y2 - y1
    # normal = (dy, -dx)
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    a, b, c = _canonicalize_line(a, b, c)
    return LineABC(a=a, b=b, c=c)


def intersect_lines(l1: LineABC, l2: LineABC) -> Optional[np.ndarray]:
    D = l1.a * l2.b - l2.a * l1.b
    if abs(D) < 1e-7:
        return None
    x = (l1.b * l2.c - l2.b * l1.c) / D
    y = (l2.a * l1.c - l1.a * l2.c) / D
    return np.array([x, y], dtype=np.float32)


def fit_line_abc(points_xy: np.ndarray) -> Optional[LineABC]:
    # points_xy: (N,2) float32
    if points_xy is None or len(points_xy) < 10:
        return None
    pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
    # returns vx, vy, x0, y0
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
    # normal = (vy, -vx)
    a = vy
    b = -vx
    c = -(a * x0 + b * y0)
    a, b, c = _canonicalize_line(a, b, c)
    return LineABC(a=a, b=b, c=c)


def best_contour_from_binary(
    used: np.ndarray,
    min_area_frac: float = 0.006,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (contour, mask) where mask is filled contour mask.
    """
    if used.ndim != 2:
        used = cv2.cvtColor(used, cv2.COLOR_BGR2GRAY)
    H, W = used.shape[:2]

    # ensure binary
    used_bin = cv2.threshold(used, 0, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(used_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    mask = None
    if not cnts:
        return None, None

    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts_sorted[:12]:
        a = float(cv2.contourArea(c))
        if a < float(H * W) * float(min_area_frac):
            continue
        x, y, w, h = cv2.boundingRect(c)
        # reject heavy border touch
        if x <= 1 or y <= 1 or (x + w) >= (W - 2) or (y + h) >= (H - 2):
            continue
        if a / float(H * W) > 0.93:
            continue
        contour = c
        break

    if contour is None:
        return None, None

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return contour, mask


def quad_from_contour_v3(contour: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Build quad from contour:
    1) convex hull + approxPolyDP sweep for 4 corners
    2) fallback: minAreaRect
    """
    if contour is None or len(contour) < 4:
        return None, "no contour"

    hull = cv2.convexHull(contour)
    per = float(cv2.arcLength(hull, True))

    # try approxPolyDP (prefer true corners if possible)
    for eps in np.linspace(0.005, 0.06, 24):
        poly = cv2.approxPolyDP(hull, eps * per, True)
        if len(poly) == 4 and cv2.isContourConvex(poly):
            quad = order_quad(poly.reshape(4, 2).astype(np.float32))
            if (not is_degenerate_quad(quad, min_rel_area=0.005)) and quad_has_4_distinct_corners(quad):
                return quad, f"approxPolyDP eps={eps:.3f}"

    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    quad = order_quad(box.astype(np.float32))
    if (not is_degenerate_quad(quad, min_rel_area=0.005)) and quad_has_4_distinct_corners(quad):
        return quad, "minAreaRect"
    return None, "degenerate quad from contour"


def refine_quad_with_fitlines(
    quad_init: np.ndarray,
    contour: np.ndarray,
    min_pts_per_side: int = 25,
) -> Tuple[np.ndarray, List[LineABC], str]:
    """
    Given initial quad and contour, fit a line for each side using contour/hull points,
    then intersect adjacent lines to get refined corners.
    Returns (quad_refined, lines4, note).
    If refinement fails, returns init with fallback note.
    """
    quad_init = np.asarray(quad_init, dtype=np.float32).reshape(4, 2)

    hull = cv2.convexHull(contour)
    pts = hull.reshape(-1, 2).astype(np.float32)
    if len(pts) < 40:
        # too few points to fit robustly
        lines = [
            line_from_points(quad_init[0], quad_init[1]),
            line_from_points(quad_init[1], quad_init[2]),
            line_from_points(quad_init[2], quad_init[3]),
            line_from_points(quad_init[3], quad_init[0]),
        ]
        return quad_init, lines, "fitLine skipped (few hull points)"

    # initial edge lines (top, right, bottom, left)
    l0 = line_from_points(quad_init[0], quad_init[1])  # top
    l1 = line_from_points(quad_init[1], quad_init[2])  # right
    l2 = line_from_points(quad_init[2], quad_init[3])  # bottom
    l3 = line_from_points(quad_init[3], quad_init[0])  # left
    init_lines = [l0, l1, l2, l3]

    # assign each hull point to nearest line (abs distance)
    groups = [[], [], [], []]  # indices
    for p in pts:
        x, y = float(p[0]), float(p[1])
        d = [abs(L.a * x + L.b * y + L.c) for L in init_lines]
        gi = int(np.argmin(d))
        groups[gi].append(p)

    fitted_lines: List[LineABC] = []
    used_fit = 0
    for i in range(4):
        g = np.asarray(groups[i], dtype=np.float32)
        if len(g) >= int(min_pts_per_side):
            fl = fit_line_abc(g)
            if fl is not None:
                fitted_lines.append(fl)
                used_fit += 1
                continue
        fitted_lines.append(init_lines[i])

    # intersect adjacent: top&right=TR, right&bottom=BR, bottom&left=BL, left&top=TL
    top, right, bottom, left = fitted_lines
    tr = intersect_lines(top, right)
    br = intersect_lines(right, bottom)
    bl = intersect_lines(bottom, left)
    tl = intersect_lines(left, top)

    if tr is None or br is None or bl is None or tl is None:
        return quad_init, fitted_lines, "fitLine refinement failed (parallel intersections)"

    quad_ref = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    if is_degenerate_quad(quad_ref, min_rel_area=0.005) or (not quad_has_4_distinct_corners(quad_ref)):
        return quad_init, fitted_lines, "fitLine refinement produced degenerate quad"

    note = f"fitLine refined (sides_fit={used_fit}/4)"
    return quad_ref, fitted_lines, note


@dataclass
class PlanBDebugV3:
    variant: str
    raw: np.ndarray
    close_img: np.ndarray
    open_img: np.ndarray
    used: np.ndarray
    contour: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    edges: np.ndarray
    quad_init: Optional[np.ndarray]
    quad_final: Optional[np.ndarray]
    lines: List[LineABC]
    ok: bool
    note: str


def planb_quad_from_variant_v3(
    crop_bgr_img: np.ndarray,
    variant: str,
    k: int,
    close_it: int,
    open_it: int,
    min_area_frac: float = 0.006,
    do_refine: bool = True,
) -> PlanBDebugV3:
    H, W = crop_bgr_img.shape[:2]
    variants = preprocess_variants(crop_bgr_img)
    v = variant if variant in variants else "otsu"
    raw = variants[v]

    # V3 pipeline: CLOSE then OPEN (open may be 0)
    k = _ensure_odd(k)
    close_img = morph(raw, cv2.MORPH_CLOSE, k, close_it)
    open_img = morph(close_img, cv2.MORPH_OPEN, k, open_it)
    used = open_img

    contour, mask = best_contour_from_binary(used, min_area_frac=min_area_frac)

    # edges visualization: use boundary of mask if possible
    if mask is not None:
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, ker)
    else:
        # fallback: gradient of used
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(cv2.threshold(used, 0, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_GRADIENT, ker)

    if contour is None:
        return PlanBDebugV3(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, used=used,
            contour=None, mask=None, edges=edges, quad_init=None, quad_final=None,
            lines=[], ok=False, note="No suitable contour found"
        )

    quad_init, note0 = quad_from_contour_v3(contour)
    if quad_init is None:
        return PlanBDebugV3(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, used=used,
            contour=contour, mask=mask, edges=edges, quad_init=None, quad_final=None,
            lines=[], ok=False, note=f"Contour found but quad failed: {note0}"
        )

    # lines from init
    lines_init = [
        line_from_points(quad_init[0], quad_init[1]),
        line_from_points(quad_init[1], quad_init[2]),
        line_from_points(quad_init[2], quad_init[3]),
        line_from_points(quad_init[3], quad_init[0]),
    ]

    quad_final = quad_init
    lines_final = lines_init
    note = f"OK | {note0}"

    if do_refine:
        quad_ref, fitted_lines, note1 = refine_quad_with_fitlines(quad_init, contour, min_pts_per_side=25)
        quad_final = quad_ref
        lines_final = fitted_lines
        note = f"OK | {note0} | {note1}"

    # sanity bounds: allow a small margin (fitLine can extrapolate slightly)
    margin = 25.0
    if np.any(quad_final[:, 0] < -margin) or np.any(quad_final[:, 0] > (W - 1 + margin)) or np.any(quad_final[:, 1] < -margin) or np.any(quad_final[:, 1] > (H - 1 + margin)):
        # fallback to init if init is inside
        if not (np.any(quad_init[:, 0] < -margin) or np.any(quad_init[:, 0] > (W - 1 + margin)) or np.any(quad_init[:, 1] < -margin) or np.any(quad_init[:, 1] > (H - 1 + margin))):
            quad_final = quad_init
            lines_final = lines_init
            note = f"OK | {note0} | refine rejected (too far outside)"
        else:
            return PlanBDebugV3(
                variant=v, raw=raw, close_img=close_img, open_img=open_img, used=used,
                contour=contour, mask=mask, edges=edges, quad_init=quad_init, quad_final=None,
                lines=lines_final, ok=False, note="Quad too far outside crop bounds"
            )

    return PlanBDebugV3(
        variant=v, raw=raw, close_img=close_img, open_img=open_img, used=used,
        contour=contour, mask=mask, edges=edges, quad_init=quad_init, quad_final=quad_final,
        lines=lines_final, ok=True, note=note
    )


# ============================================================
# "Rotation around X/Y" from quad (camera-agnostic heuristic)
# ============================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_rot_xy_deg_from_quad(quad: np.ndarray) -> Tuple[float, float]:
    top, right, bottom, left = side_lengths(quad)
    r_tb = min(top, bottom) / (max(top, bottom) + 1e-6)
    rot_x = math.degrees(math.acos(clamp01(r_tb)))
    r_lr = min(left, right) / (max(left, right) + 1e-6)
    rot_y = math.degrees(math.acos(clamp01(r_lr)))

    rot_x *= 1.0 if (bottom > top) else -1.0
    rot_y *= 1.0 if (right > left) else -1.0
    return rot_x, rot_y


# ============================================================
# Warping to top view
# ============================================================
def warp_quad_to_square(img_bgr: np.ndarray, quad_full: np.ndarray, out_size: int = 420) -> np.ndarray:
    S = int(out_size)
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], dtype=np.float32)
    src = quad_full.astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, Hm, (S, S), flags=cv2.INTER_CUBIC)
    return warped


# ============================================================
# Drawing helpers / Tk helpers
# ============================================================
def draw_box(img: np.ndarray, box: Tuple[int, int, int, int], color=(255, 255, 0), thickness=3):
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def draw_poly(img: np.ndarray, pts: np.ndarray, color=(0, 255, 0), thickness=3):
    p = pts.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [p], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for (x, y) in pts.astype(np.int32):
        cv2.circle(img, (int(x), int(y)), 5, color, -1, lineType=cv2.LINE_AA)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def bgr_to_tk_photo(img_bgr: np.ndarray) -> tk.PhotoImage:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for Tk display.")
    b64 = base64.b64encode(buf.tobytes())
    return tk.PhotoImage(data=b64)


def make_grid(images: List[np.ndarray], cols: int = 2, pad: int = 6, bg: int = 30) -> np.ndarray:
    imgs = [im.copy() for im in images]
    hmax = max(im.shape[0] for im in imgs)
    wmax = max(im.shape[1] for im in imgs)

    norm = []
    for im in imgs:
        if im.shape[0] != hmax or im.shape[1] != wmax:
            im = cv2.resize(im, (wmax, hmax), interpolation=cv2.INTER_AREA)
        norm.append(im)

    rows = int(math.ceil(len(norm) / cols))
    out_h = rows * hmax + (rows + 1) * pad
    out_w = cols * wmax + (cols + 1) * pad
    canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)

    for idx, im in enumerate(norm):
        r = idx // cols
        c = idx % cols
        y = pad + r * (hmax + pad)
        x = pad + c * (wmax + pad)
        canvas[y : y + hmax, x : x + wmax] = im
    return canvas


# ============================================================
# Candidate struct
# ============================================================
@dataclass
class Candidate:
    idx: int
    merged: MergedROI
    crop_box: Tuple[int, int, int, int]
    src: str                    # "A" or "B" or "none"
    quad_crop: Optional[np.ndarray]
    quad_full: Optional[np.ndarray]
    rot_x_deg: Optional[float]
    rot_y_deg: Optional[float]
    warp_bgr: Optional[np.ndarray]


# ============================================================
# GUI V3
# ============================================================
class RectifyGUIv3:
    """
    Views:
      1) Detected (full image, A quad + initial B quad if any)
      2) Warp (selected, live B recompute)
      3) Crop + quad (selected, live B recompute)
      4) Binary raw (selected variant)
      5) Morph + contour/edges + line-fit debug grid (selected, live B recompute)
    """
    def __init__(
        self,
        root: tk.Tk,
        image_paths: List[Path],
        roi_cfg: RoiC2FConfig,
        model: LoadedModel,
        patch_thr: float,
        merge_iou: float,
        roi_pad_frac: float,
        max_patches: int,
        batch_size: int,
        warp_size: int,
    ):
        self.root = root
        self.root.title("QR Rectify GUI V3 (Plan B = contour quad + fitLine refinement)")

        self.image_paths = image_paths
        self.roi_cfg = roi_cfg
        self.model = model

        self.patch_thr = float(patch_thr)
        self.merge_iou = float(merge_iou)
        self.roi_pad_frac = float(roi_pad_frac)
        self.max_patches = int(max_patches)
        self.batch_size = int(batch_size)
        self.warp_size = int(warp_size)

        self.det = cv2.QRCodeDetector()

        self.idx_img = 0
        self.view = 0
        self.cur_img_bgr: Optional[np.ndarray] = None
        self.candidates: List[Candidate] = []
        self.selected_cand = 0

        # live controls (defaults per your request)
        self.var_variant = tk.StringVar(value="otsu")
        self.var_k = tk.IntVar(value=3)
        self.var_close_it = tk.IntVar(value=4)
        self.var_open_it = tk.IntVar(value=0)
        self.var_refine = tk.IntVar(value=1)  # checkbox: fitLine refine on/off

        self._build_ui()
        self._bind_live_traces()
        self._load_image(0)

    def _bind_live_traces(self):
        for v in (self.var_variant, self.var_k, self.var_close_it, self.var_open_it, self.var_refine):
            try:
                v.trace_add("write", lambda *args: self._render())
            except Exception:
                pass

    def _build_ui(self):
        self.root.geometry("1500x900")

        self.frm_left = ttk.Frame(self.root, padding=10)
        self.frm_left.pack(side=tk.LEFT, fill=tk.Y)

        self.frm_right = ttk.Frame(self.root, padding=10)
        self.frm_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.frm_left, text="View:").pack(anchor="w")
        ttk.Button(self.frm_left, text="1) Detected (full)", command=lambda: self._set_view(0)).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(self.frm_left, text="2) Warp (top view)", command=lambda: self._set_view(1)).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(self.frm_left, text="3) Crop + quad", command=lambda: self._set_view(2)).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(self.frm_left, text="4) Binary raw", command=lambda: self._set_view(3)).pack(anchor="w", fill=tk.X, pady=2)
        ttk.Button(self.frm_left, text="5) Morph + debug", command=lambda: self._set_view(4)).pack(anchor="w", fill=tk.X, pady=2)

        ttk.Separator(self.frm_left).pack(fill=tk.X, pady=10)

        ttk.Label(self.frm_left, text="Images:").pack(anchor="w")
        ttk.Button(self.frm_left, text="◀ Prev", command=self.prev_image).pack(anchor="w", fill=tk.X)
        ttk.Button(self.frm_left, text="Next ▶", command=self.next_image).pack(anchor="w", fill=tk.X, pady=2)

        self.btn_recompute = ttk.Button(self.frm_left, text="Recompute CNN (r)", command=self.recompute)
        self.btn_recompute.pack(anchor="w", fill=tk.X, pady=(6, 0))

        ttk.Separator(self.frm_left).pack(fill=tk.X, pady=10)

        ttk.Label(self.frm_left, text="Candidates (merged):").pack(anchor="w")
        self.lst = tk.Listbox(self.frm_left, height=10, width=64, exportselection=False)
        self.lst.pack(anchor="w", fill=tk.X)
        self.lst.bind("<<ListboxSelect>>", lambda e: self._on_select_candidate())

        ttk.Separator(self.frm_left).pack(fill=tk.X, pady=10)

        ttk.Label(self.frm_left, text="Live preprocessing controls (Plan B V3):").pack(anchor="w")

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Variant:").pack(side=tk.LEFT)
        self.cmb = ttk.Combobox(
            row, textvariable=self.var_variant, width=12, state="readonly",
            values=["auto", "edges", "adap", "otsu", "otsu_inv"]
        )
        self.cmb.pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Kernel k (odd):").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=21, textvariable=self.var_k, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Close iters:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=10, textvariable=self.var_close_it, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Open iters:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=10, textvariable=self.var_open_it, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row, text="Refine (fitLine)", variable=self.var_refine).pack(side=tk.LEFT)

        ttk.Separator(self.frm_left).pack(fill=tk.X, pady=10)

        ttk.Label(self.frm_left, text="Info:").pack(anchor="w")
        self.txt = tk.Text(self.frm_left, height=22, width=64)
        self.txt.pack(anchor="w", fill=tk.BOTH, expand=False)

        self.lbl_img = ttk.Label(self.frm_right)
        self.lbl_img.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<space>", lambda e: self._cycle_view())
        self.root.bind("r", lambda e: self.recompute())
        self.root.bind("R", lambda e: self.recompute())

    def _set_view(self, v: int):
        self.view = int(v)
        self._render()

    def _cycle_view(self):
        self.view = (self.view + 1) % 5
        self._render()

    def prev_image(self):
        self._load_image(self.idx_img - 1)

    def next_image(self):
        self._load_image(self.idx_img + 1)

    def recompute(self):
        self._load_image(self.idx_img)

    def _load_image(self, idx: int):
        if not self.image_paths:
            return
        self.idx_img = int(np.clip(idx, 0, len(self.image_paths) - 1))
        p = self.image_paths[self.idx_img]

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showwarning("Unreadable image", f"Could not read:\n{p}")
            return

        self.cur_img_bgr = img
        self._compute_candidates()
        self.selected_cand = 0
        self._update_listbox()
        self._render()

    def _compute_candidates(self):
        self.candidates = []
        if self.cur_img_bgr is None:
            return

        img = self.cur_img_bgr
        H, W = img.shape[:2]

        centers = propose_centers_c2f(img, self.roi_cfg)[: self.max_patches]
        patches = [pad_and_crop(img, int(c["cx"]), int(c["cy"]), int(self.roi_cfg.patch_size)) for c in centers]

        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(patches), self.batch_size):
                batch = patches[start : start + self.batch_size]
                x = patches_to_tensor(batch, int(self.model.input_size), self.model.mean, self.model.std, self.model.device)
                y = self.model.model(x)
                p = probs_from_logits(y)
                probs.extend([float(v) for v in p.tolist()])

        ps = int(self.roi_cfg.patch_size)
        half = ps // 2
        pos: List[PatchBox] = []
        for i, (c, pr) in enumerate(zip(centers, probs)):
            if float(pr) < float(self.patch_thr):
                continue
            cx, cy = int(c["cx"]), int(c["cy"])
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + ps, y0 + ps
            x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
            pos.append(PatchBox(idx=i, box=(x0, y0, x1, y1), p=float(pr)))

        merged = cluster_boxes_by_iou(pos, float(self.merge_iou))
        merged = merged[:60]

        # initial candidate quads: Plan A first, else Plan B with current defaults
        for mi, m in enumerate(merged):
            crop_box = expand_box(m.box, float(self.roi_pad_frac), W, H)
            crop = crop_bgr(img, crop_box)

            quad_crop = None
            quad_full = None
            src = "none"
            rot_x = None
            rot_y = None
            warp = None

            qA = detect_qr_corners_opencv(self.det, crop)
            if qA is not None:
                quad_crop = qA
                quad_full = qA.copy()
                quad_full[:, 0] += crop_box[0]
                quad_full[:, 1] += crop_box[1]
                src = "A"
            else:
                v = self._pick_variant_for_debug(None)
                dbg = planb_quad_from_variant_v3(
                    crop_bgr_img=crop,
                    variant=v,
                    k=int(self.var_k.get()),
                    close_it=int(self.var_close_it.get()),
                    open_it=int(self.var_open_it.get()),
                    do_refine=bool(int(self.var_refine.get())),
                )
                if dbg.ok and dbg.quad_final is not None:
                    quad_crop = dbg.quad_final
                    quad_full = dbg.quad_final.copy()
                    quad_full[:, 0] += crop_box[0]
                    quad_full[:, 1] += crop_box[1]
                    src = "B"

            if quad_full is not None:
                rot_x, rot_y = estimate_rot_xy_deg_from_quad(quad_full)
                try:
                    warp = warp_quad_to_square(img, quad_full, out_size=self.warp_size)
                except Exception:
                    warp = None

            self.candidates.append(
                Candidate(
                    idx=mi,
                    merged=m,
                    crop_box=crop_box,
                    src=src,
                    quad_crop=quad_crop,
                    quad_full=quad_full,
                    rot_x_deg=rot_x,
                    rot_y_deg=rot_y,
                    warp_bgr=warp,
                )
            )

    def _update_listbox(self):
        self.lst.delete(0, tk.END)
        for i, c in enumerate(self.candidates):
            x0, y0, x1, y1 = c.merged.box
            self.lst.insert(
                tk.END,
                f"{i:02d} | score={c.merged.score:.3f} | {c.src} | box=({x0},{y0})-({x1},{y1})"
            )
        if self.candidates:
            self.lst.selection_set(0)
            self.lst.activate(0)

    def _on_select_candidate(self):
        sel = self.lst.curselection()
        if not sel:
            return
        self.selected_cand = int(sel[0])
        self._render()

    def _pick_variant_for_debug(self, cand: Optional[Candidate]) -> str:
        v = self.var_variant.get()
        if v == "auto":
            return "otsu"
        return v

    def _live_planb_debug(self, cand: Candidate) -> PlanBDebugV3:
        assert self.cur_img_bgr is not None
        crop = crop_bgr(self.cur_img_bgr, cand.crop_box)
        v = self._pick_variant_for_debug(cand)
        dbg = planb_quad_from_variant_v3(
            crop_bgr_img=crop,
            variant=v,
            k=int(self.var_k.get()),
            close_it=int(self.var_close_it.get()),
            open_it=int(self.var_open_it.get()),
            do_refine=bool(int(self.var_refine.get())),
        )
        return dbg

    def _render(self):
        if self.cur_img_bgr is None:
            return

        if not self.candidates:
            view = self.cur_img_bgr.copy()
            self._show_image(view)
            self._update_info(None, None)
            return

        i = int(np.clip(self.selected_cand, 0, len(self.candidates) - 1))
        cand = self.candidates[i]
        img = self.cur_img_bgr

        live_dbg = None
        if cand.src != "A":
            live_dbg = self._live_planb_debug(cand)

        if self.view == 0:
            view = img.copy()
            for j, c in enumerate(self.candidates):
                th = 5 if j == i else 3
                draw_box(view, c.merged.box, color=(255, 255, 0), thickness=th)
                x0, y0, _, _ = c.merged.box
                cv2.putText(view, f"#{j} {c.src}", (x0 + 3, max(18, y0 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
                if c.quad_full is not None:
                    col = (0, 255, 0) if c.src == "A" else (255, 0, 255)
                    draw_poly(view, c.quad_full, color=col, thickness=4 if j == i else 3)

            self._show_image(view)
            self._update_info(cand, live_dbg)

        elif self.view == 1:
            if cand.src == "A" and cand.warp_bgr is not None:
                warp = cand.warp_bgr.copy()
                rx = cand.rot_x_deg or 0.0
                ry = cand.rot_y_deg or 0.0
                cv2.putText(warp, "method=A", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(warp, f"rot_x={rx:.2f} deg  rot_y={ry:.2f} deg", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                self._show_image(warp)
                self._update_info(cand, live_dbg)
                return

            if live_dbg is None or (not live_dbg.ok) or (live_dbg.quad_final is None):
                ph = np.zeros((self.warp_size, self.warp_size, 3), dtype=np.uint8)
                cv2.putText(ph, "No warp (Plan B failed)", (20, self.warp_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self._show_image(ph)
                self._update_info(cand, live_dbg)
                return

            quad_crop = live_dbg.quad_final
            quad_full = quad_crop.copy()
            quad_full[:, 0] += cand.crop_box[0]
            quad_full[:, 1] += cand.crop_box[1]
            warp = warp_quad_to_square(img, quad_full, out_size=self.warp_size)

            rx, ry = estimate_rot_xy_deg_from_quad(quad_full)
            cv2.putText(warp, f"method=B(contour) var={live_dbg.variant}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(warp, f"rot_x={rx:.2f} deg  rot_y={ry:.2f} deg", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            self._show_image(warp)
            self._update_info(cand, live_dbg)

        elif self.view == 2:
            crop = crop_bgr(img, cand.crop_box)
            view = crop.copy()

            if cand.src == "A" and cand.quad_crop is not None:
                draw_poly(view, cand.quad_crop, color=(0, 255, 0), thickness=3)
                cv2.putText(view, "A:opencv", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if live_dbg is not None and live_dbg.contour is not None:
                    cv2.drawContours(view, [live_dbg.contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
                if live_dbg is not None and live_dbg.ok and live_dbg.quad_final is not None:
                    draw_poly(view, live_dbg.quad_final, color=(255, 0, 255), thickness=3)
                    cv2.putText(view, f"B:contour var={live_dbg.variant}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(view, f"B failed: {live_dbg.note if live_dbg else 'no dbg'}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            self._show_image(view)
            self._update_info(cand, live_dbg)

        elif self.view == 3:
            crop = crop_bgr(img, cand.crop_box)
            v = self._pick_variant_for_debug(cand)
            raw = preprocess_variants(crop).get(v, preprocess_variants(crop)["otsu"])
            view = gray_to_bgr(raw)
            cv2.putText(view, f"RAW variant={v}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            self._show_image(view)
            self._update_info(cand, live_dbg)

        else:
            crop = crop_bgr(img, cand.crop_box)
            v = self._pick_variant_for_debug(cand)

            dbg = live_dbg if live_dbg is not None else planb_quad_from_variant_v3(
                crop_bgr_img=crop,
                variant=v,
                k=int(self.var_k.get()),
                close_it=int(self.var_close_it.get()),
                open_it=int(self.var_open_it.get()),
                do_refine=bool(int(self.var_refine.get())),
            )

            crop_vis = crop.copy()
            if dbg.contour is not None:
                cv2.drawContours(crop_vis, [dbg.contour], -1, (0, 255, 255), 2, cv2.LINE_AA)  # contour in yellow-ish
            if dbg.ok and dbg.quad_final is not None:
                draw_poly(crop_vis, dbg.quad_final, color=(255, 0, 255), thickness=3)
            cv2.putText(crop_vis, f"crop | var={dbg.variant} | {dbg.note}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                crop_vis,
                f"k={self.var_k.get()} close={self.var_close_it.get()} open={self.var_open_it.get()} refine={int(self.var_refine.get())}",
                (10, crop_vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA
            )

            p_raw = gray_to_bgr(dbg.raw)
            cv2.putText(p_raw, "RAW", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_close = gray_to_bgr(dbg.close_img)
            cv2.putText(p_close, "CLOSE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_open = gray_to_bgr(dbg.open_img)
            cv2.putText(p_open, "OPEN", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_used = gray_to_bgr(dbg.used)
            cv2.putText(p_used, "USED (close->open)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_edges = gray_to_bgr(dbg.edges)
            cv2.putText(p_edges, "EDGES (mask boundary)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # overlay fitted lines on edges
            lines_vis = gray_to_bgr(dbg.edges)
            Hc, Wc = dbg.edges.shape[:2]

            def draw_line_abc(vis: np.ndarray, ln: LineABC, color: Tuple[int, int, int], thick: int = 2):
                a, b, c = ln.a, ln.b, ln.c
                pts2 = []
                if abs(b) > 1e-7:
                    y0 = int(round((-c - a * 0.0) / b))
                    y1 = int(round((-c - a * (Wc - 1.0)) / b))
                    pts2 = [(0, y0), (Wc - 1, y1)]
                elif abs(a) > 1e-7:
                    x0 = int(round((-c - b * 0.0) / a))
                    x1 = int(round((-c - b * (Hc - 1.0)) / a))
                    pts2 = [(x0, 0), (x1, Hc - 1)]
                if pts2:
                    cv2.line(vis, pts2[0], pts2[1], color, thick, cv2.LINE_AA)

            for ln in dbg.lines:
                draw_line_abc(lines_vis, ln, (0, 0, 255), 2)

            if dbg.ok and dbg.quad_final is not None:
                draw_poly(lines_vis, dbg.quad_final, (255, 0, 255), 2)

            cv2.putText(lines_vis, "Fitted lines (red) + quad (magenta)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            grid = make_grid([crop_vis, p_raw, p_close, p_open, p_used, p_edges, lines_vis], cols=2)
            self._show_image(grid)
            self._update_info(cand, dbg)

    def _show_image(self, img_bgr: np.ndarray):
        max_w = 1050
        max_h = 860
        h, w = img_bgr.shape[:2]
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 1.0:
            disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = img_bgr

        photo = bgr_to_tk_photo(disp)
        self.lbl_img.configure(image=photo)
        self.lbl_img.image = photo

    def _update_info(self, cand: Optional[Candidate], dbg: Optional[PlanBDebugV3]):
        p = self.image_paths[self.idx_img]
        lines = []
        lines.append(f"Image: {p.name}  ({self.idx_img+1}/{len(self.image_paths)})")
        lines.append(f"View: {self.view+1}/5")
        lines.append("")
        lines.append(f"Model run: {self.model.name}  kind={self.model.kind}  device={self.model.device.type}")
        lines.append(f"patch_thr={self.patch_thr:.2f}   merge_iou={self.merge_iou:.2f}   roi_pad_frac={self.roi_pad_frac:.2f}")
        lines.append("")
        lines.append(f"Candidates: {len(self.candidates)}")

        if cand is not None:
            idx = int(np.clip(self.selected_cand, 0, max(0, len(self.candidates) - 1)))
            lines.append("")
            lines.append(f"Selected: #{idx}  score={cand.merged.score:.3f}  src={cand.src}")
            lines.append(f"Box: {cand.merged.box}  members={len(cand.merged.members)}")
            lines.append(f"Crop box: {cand.crop_box}")

            lines.append("")
            lines.append("Live Plan B controls:")
            lines.append(f"  variant={self.var_variant.get()}  k={self.var_k.get()}  close={self.var_close_it.get()}  open={self.var_open_it.get()}  refine={int(self.var_refine.get())}")

            if dbg is not None:
                lines.append("")
                lines.append(f"Live Plan B: var={dbg.variant} ok={dbg.ok} note={dbg.note}")
                lines.append(f"  contour={'yes' if dbg.contour is not None else 'no'}  lines={len(dbg.lines)}")
                if dbg.ok and dbg.quad_final is not None:
                    quad_full = dbg.quad_final + np.array([cand.crop_box[0], cand.crop_box[1]], dtype=np.float32)
                    rx, ry = estimate_rot_xy_deg_from_quad(quad_full)
                    lines.append(f"  rot_x={rx:.2f} deg  rot_y={ry:.2f} deg")

        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(lines))


# ============================================================
# CLI / main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("GUI V3: Plan B = contour quad + optional fitLine refinement.")
    ap.add_argument("--input", type=str, required=True, help="Folder with images (or single image).")
    ap.add_argument("--roi-config", type=str, required=True, help="ROI config JSON.")
    ap.add_argument("--runs-dir", type=str, default="runs/cnn_scratch", help="Runs directory to scan (best.pt).")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")

    ap.add_argument("--run-index", type=int, default=0, help="Which run to use (0=newest).")
    ap.add_argument("--patch-thr", type=float, default=0.95, help="Patch score threshold.")
    ap.add_argument("--merge-iou", type=float, default=0.30, help="IoU threshold for clustering patches.")
    ap.add_argument("--roi-pad-frac", type=float, default=0.20, help="Padding for merged union box before quad detection.")

    ap.add_argument("--max-patches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warp-size", type=int, default=420)
    return ap.parse_args()


def main():
    if torch is None:
        raise SystemExit("PyTorch is required for this script. Please install torch.")

    args = parse_args()

    in_path = Path(args.input)
    roi_cfg_path = Path(args.roi_config)
    if not roi_cfg_path.is_absolute():
        roi_cfg_path = REPO_ROOT / roi_cfg_path
    roi_cfg = load_roi_config(roi_cfg_path)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir

    runs = discover_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No runs found in: {runs_dir} (expected run subdirs with best.pt)")
    ridx = int(np.clip(args.run_index, 0, len(runs) - 1))
    run_dir = runs[ridx]
    model = load_model_from_run_dir(run_dir, args.device)

    img_paths = list(iter_image_files(in_path))
    if not img_paths:
        raise SystemExit(f"No images found in: {in_path}")

    root = tk.Tk()
    _ = RectifyGUIv3(
        root=root,
        image_paths=img_paths,
        roi_cfg=roi_cfg,
        model=model,
        patch_thr=float(args.patch_thr),
        merge_iou=float(args.merge_iou),
        roi_pad_frac=float(args.roi_pad_frac),
        max_patches=int(args.max_patches),
        batch_size=int(args.batch_size),
        warp_size=int(args.warp_size),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
