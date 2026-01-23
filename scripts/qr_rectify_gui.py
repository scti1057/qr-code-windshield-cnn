# scripts/qr_rectify_gui_v2.py
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


def touches_border(quad: np.ndarray, H: int, W: int, m: int = 3) -> bool:
    return (
        np.any(quad[:, 0] < m) or np.any(quad[:, 0] > (W - 1 - m)) or
        np.any(quad[:, 1] < m) or np.any(quad[:, 1] > (H - 1 - m))
    )


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
# Plan B (Method B): lines + intersections from binary/edges
# ============================================================
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


@dataclass
class LineSeg:
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    theta: float  # direction angle in [0, pi)


@dataclass
class LineABC:
    a: float
    b: float
    c: float
    length: float


@dataclass
class PlanBLinesDebug:
    variant: str
    raw: np.ndarray
    close_img: np.ndarray
    open_img: np.ndarray
    close_open: np.ndarray
    used: np.ndarray           # image used for contours/edges (close_open)
    contour: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    edges: np.ndarray
    lines: List[LineSeg]
    chosen_lines: List[LineABC]  # 4 lines (2 per cluster)
    points: Optional[np.ndarray] # (4,2) in crop coords
    ok: bool
    note: str


def _segments_from_hough(edges: np.ndarray) -> List[LineSeg]:
    H, W = edges.shape[:2]
    m = float(min(H, W))
    # heuristic params (scale with crop size)
    threshold = max(20, int(0.12 * m))
    min_len  = max(15, int(0.18 * m))
    max_gap = max(5, int(0.03 * m))

    linesP = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=threshold, minLineLength=min_len, maxLineGap=max_gap
    )
    segs: List[LineSeg] = []
    if linesP is None:
        return segs
    for l in linesP.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, l.tolist())
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 10:
            continue
        theta = math.atan2(dy, dx)
        # map to [0, pi)
        if theta < 0:
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        segs.append(LineSeg(x1, y1, x2, y2, length, theta))
    return segs


def _line_abc_from_seg(seg: LineSeg) -> LineABC:
    # from 2 points to ax+by+c=0, normalized
    x1, y1, x2, y2 = float(seg.x1), float(seg.y1), float(seg.x2), float(seg.y2)
    dx = x2 - x1
    dy = y2 - y1
    # normal = (dy, -dx)
    a = dy
    b = -dx
    c = -(a * x1 + b * y1)
    n = math.hypot(a, b) + 1e-9
    a /= n
    b /= n
    c /= n
    return LineABC(a=a, b=b, c=c, length=seg.length)


def _kmeans2_unit(v: np.ndarray, iters: int = 10) -> np.ndarray:
    """
    v: (N,2) unit-ish vectors
    returns labels in {0,1}
    """
    N = v.shape[0]
    if N <= 1:
        return np.zeros((N,), dtype=np.int32)

    c0 = v[0].copy()
    d = np.linalg.norm(v - c0[None, :], axis=1)
    c1 = v[int(np.argmax(d))].copy()

    labels = np.zeros((N,), dtype=np.int32)
    for _ in range(iters):
        d0 = np.linalg.norm(v - c0[None, :], axis=1)
        d1 = np.linalg.norm(v - c1[None, :], axis=1)
        new_labels = (d1 < d0).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels

        if np.any(labels == 0):
            c0 = v[labels == 0].mean(axis=0)
        if np.any(labels == 1):
            c1 = v[labels == 1].mean(axis=0)

        n0 = np.linalg.norm(c0) + 1e-9
        n1 = np.linalg.norm(c1) + 1e-9
        c0 = c0 / n0
        c1 = c1 / n1

    return labels


def _pick_extreme_lines(lines: List[LineABC], center_xy: Tuple[float, float]) -> Optional[Tuple[LineABC, LineABC]]:
    """
    Pick two roughly parallel lines on opposite sides of center:
      use signed distance of line to center, choose min and max (with robustness).
    """
    if len(lines) < 2:
        return None
    cx, cy = center_xy
    dists = []
    for ln in lines:
        # signed distance (since normalized)
        d = ln.a * cx + ln.b * cy + ln.c
        dists.append(d)

    idx = np.argsort(np.array(dists))
    k = max(1, len(lines) // 5)

    low_idxs = idx[:k]
    high_idxs = idx[-k:]

    low = max((lines[i] for i in low_idxs), key=lambda L: L.length)
    high = max((lines[i] for i in high_idxs), key=lambda L: L.length)

    # if too close, widen selection
    if abs((low.a * cx + low.b * cy + low.c) - (high.a * cx + high.b * cy + high.c)) < 5.0:
        low = lines[int(idx[0])]
        high = lines[int(idx[-1])]
    return (low, high)


def _intersect(l1: LineABC, l2: LineABC) -> Optional[np.ndarray]:
    D = l1.a * l2.b - l2.a * l1.b
    if abs(D) < 1e-6:
        return None
    x = (l1.b * l2.c - l2.b * l1.c) / D
    y = (l2.a * l1.c - l1.a * l2.c) / D
    return np.array([x, y], dtype=np.float32)


def planb_lines_from_variant(
    crop_bgr_img: np.ndarray,
    variant: str,
    k: int,
    close_it: int,
    open_it: int,
    min_area_frac: float = 0.006,
) -> PlanBLinesDebug:
    """
    Build: raw -> CLOSE -> OPEN (optional) => used image
    Find contour (green) for mask; edges from used; Hough lines; cluster directions; pick 4 lines; intersect => quad.
    """
    H, W = crop_bgr_img.shape[:2]
    variants = preprocess_variants(crop_bgr_img)
    v = variant if variant in variants else "edges"
    raw = variants[v]

    k = _ensure_odd(k)
    close_img = morph(raw, cv2.MORPH_CLOSE, k, close_it)
    open_img = morph(raw, cv2.MORPH_OPEN, k, open_it)
    close_open = morph(close_img, cv2.MORPH_OPEN, k, open_it)  # used

    used = close_open

    # contours from used (for mask & green outline)
    cnts, _ = cv2.findContours(used, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = None
    mask = None
    if cnts:
        # pick best contour: large but not almost whole crop
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts_sorted[:10]:
            a = float(cv2.contourArea(c))
            if a < float(H * W) * float(min_area_frac):
                continue
            x, y, w, h = cv2.boundingRect(c)
            # reject if it touches border heavily (usually crop border / large blob)
            if x <= 1 or y <= 1 or (x + w) >= (W - 2) or (y + h) >= (H - 2):
                continue
            # reject if covers almost whole crop
            if a / float(H * W) > 0.90:
                continue
            contour = c
            break

    if contour is not None:
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    # edges for Hough
    if v == "edges":
        edges = used.copy()
    else:
        k2 = _ensure_odd(max(3, k))
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
        edges = cv2.morphologyEx(used, cv2.MORPH_GRADIENT, ker)  # dünner Rand

    edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]

    # restrict to contour region if available
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    segs = _segments_from_hough(edges)

    if len(segs) < 4:
        return PlanBLinesDebug(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
            used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=[],
            points=None, ok=False, note=f"Too few Hough segments: {len(segs)}"
        )

    # convert to lines and cluster by direction using (cos2t,sin2t)
    V = []
    for s in segs:
        V.append([math.cos(2.0 * s.theta), math.sin(2.0 * s.theta)])
    V = np.asarray(V, dtype=np.float32)

    labels = _kmeans2_unit(V, iters=12)

    # build line lists per cluster
    lines0: List[LineABC] = []
    lines1: List[LineABC] = []
    for s, lab in zip(segs, labels.tolist()):
        ln = _line_abc_from_seg(s)
        if lab == 0:
            lines0.append(ln)
        else:
            lines1.append(ln)

    # if one cluster empty -> fail
    if len(lines0) < 2 or len(lines1) < 2:
        return PlanBLinesDebug(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
            used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=[],
            points=None, ok=False, note=f"Bad clustering: lines0={len(lines0)} lines1={len(lines1)}"
        )

    center_xy = (0.5 * W, 0.5 * H)
    ex0 = _pick_extreme_lines(lines0, center_xy)
    ex1 = _pick_extreme_lines(lines1, center_xy)
    if ex0 is None or ex1 is None:
        return PlanBLinesDebug(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
            used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=[],
            points=None, ok=False, note="Could not pick extreme lines"
        )

    lA, lB = ex0
    lC, lD = ex1
    chosen = [lA, lB, lC, lD]

    # compute 4 intersections
    pts = []
    for la in (lA, lB):
        for lb in (lC, lD):
            p = _intersect(la, lb)
            if p is None:
                continue
            pts.append(p)
    if len(pts) != 4:
        return PlanBLinesDebug(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
            used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=chosen,
            points=None, ok=False, note=f"Intersections not 4 (got {len(pts)})"
        )

    pts = np.stack(pts, axis=0).astype(np.float32)
    quad = order_quad(pts)

    # validate quad
    if is_degenerate_quad(quad, min_rel_area=0.005):
        if contour is not None:
            hull = cv2.convexHull(contour)
            per = cv2.arcLength(hull, True)
            for eps in np.linspace(0.01, 0.08, 20):
                poly = cv2.approxPolyDP(hull, eps * per, True)
                if len(poly) == 4 and cv2.isContourConvex(poly):
                    quad2 = order_quad(poly.reshape(4,2).astype(np.float32))
                    if not is_degenerate_quad(quad2, min_rel_area=0.005):
                        quad = quad2
                        # -> dann NICHT returnen, sondern weitergehen (ok)
                        break
        else:
            return PlanBLinesDebug(
                variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
                used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=chosen,
                points=None, ok=False, note="Degenerate quad"
            )

    # keep points even if slightly outside (hough can extrapolate) but not too far
    margin = 20.0
    if np.any(quad[:, 0] < -margin) or np.any(quad[:, 0] > (W - 1 + margin)) or np.any(quad[:, 1] < -margin) or np.any(quad[:, 1] > (H - 1 + margin)):
        return PlanBLinesDebug(
            variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
            used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=chosen,
            points=None, ok=False, note="Quad too far outside crop bounds"
        )

    return PlanBLinesDebug(
        variant=v, raw=raw, close_img=close_img, open_img=open_img, close_open=close_open,
        used=used, contour=contour, mask=mask, edges=edges, lines=segs, chosen_lines=chosen,
        points=quad, ok=True, note="OK"
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
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", rgb)
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
# GUI v2
# ============================================================
class RectifyGUIv2:
    """
    Views:
      1) Detected (full image, A quad + initial B quad if any)
      2) Warp (selected, live B recompute with morph params)
      3) Crop + quad (selected, live B recompute)
      4) Binary raw (selected variant)
      5) Morph + edges + lines debug grid (selected, live B recompute)
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
        self.root.title("QR Rectify GUI v2 (Plan B = lines + intersections)")

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

        # debug controls (morph)
        self.var_variant = tk.StringVar(value="auto")
        self.var_k = tk.IntVar(value=3)
        self.var_close_it = tk.IntVar(value=2)
        self.var_open_it = tk.IntVar(value=1)

        self._build_ui()
        self._bind_live_traces()
        self._load_image(0)

    def _bind_live_traces(self):
        # ensure typed changes update too
        for v in (self.var_variant, self.var_k, self.var_close_it, self.var_open_it):
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
        ttk.Button(self.frm_left, text="5) Morph + lines debug", command=lambda: self._set_view(4)).pack(anchor="w", fill=tk.X, pady=2)

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

        ttk.Label(self.frm_left, text="Live preprocessing controls (Plan B):").pack(anchor="w")

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Variant:").pack(side=tk.LEFT)
        self.cmb = ttk.Combobox(row, textvariable=self.var_variant, width=12, state="readonly",
                                values=["auto", "edges", "adap", "otsu", "otsu_inv"])
        self.cmb.pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Kernel k (odd):").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=1, to=21, textvariable=self.var_k, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Close iters:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=8, textvariable=self.var_close_it, width=6).pack(side=tk.RIGHT)

        row = ttk.Frame(self.frm_left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Open iters:").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=8, textvariable=self.var_open_it, width=6).pack(side=tk.RIGHT)

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
        # re-run CNN -> rebuild candidates
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

        # initial candidate quads: Plan A first, else Plan B with default current params at compute-time
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
                # initial B using current controls (not "live", just initial)
                v = self._pick_variant_for_debug(None)
                dbg = planb_lines_from_variant(
                    crop, v,
                    k=int(self.var_k.get()),
                    close_it=int(self.var_close_it.get()),
                    open_it=int(self.var_open_it.get()),
                )
                if dbg.ok and dbg.points is not None:
                    quad_crop = dbg.points
                    quad_full = dbg.points.copy()
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
        if v != "auto":
            return v
        return "edges"

    def _live_planb_debug(self, cand: Candidate) -> PlanBLinesDebug:
        """
        Live recomputation on selected crop using current morph settings (no CNN re-run).
        """
        assert self.cur_img_bgr is not None
        crop = crop_bgr(self.cur_img_bgr, cand.crop_box)
        v = self._pick_variant_for_debug(cand)
        dbg = planb_lines_from_variant(
            crop_bgr_img=crop,
            variant=v,
            k=int(self.var_k.get()),
            close_it=int(self.var_close_it.get()),
            open_it=int(self.var_open_it.get()),
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
            # View 1: full image with initial boxes/quads (stable)
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
            # View 2: warp (live for B)
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

            if live_dbg is None or (not live_dbg.ok) or (live_dbg.points is None):
                ph = np.zeros((self.warp_size, self.warp_size, 3), dtype=np.uint8)
                cv2.putText(ph, "No warp (Plan B failed)", (20, self.warp_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                self._show_image(ph)
                self._update_info(cand, live_dbg)
                return

            quad_crop = live_dbg.points
            quad_full = quad_crop.copy()
            quad_full[:, 0] += cand.crop_box[0]
            quad_full[:, 1] += cand.crop_box[1]
            warp = warp_quad_to_square(img, quad_full, out_size=self.warp_size)

            rx, ry = estimate_rot_xy_deg_from_quad(quad_full)
            cv2.putText(warp, f"method=B(lines) var={live_dbg.variant}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(warp, f"rot_x={rx:.2f} deg  rot_y={ry:.2f} deg", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            self._show_image(warp)
            self._update_info(cand, live_dbg)

        elif self.view == 2:
            # View 3: crop + quad (live for B)
            crop = crop_bgr(img, cand.crop_box)
            view = crop.copy()

            if cand.src == "A" and cand.quad_crop is not None:
                draw_poly(view, cand.quad_crop, color=(0, 255, 0), thickness=3)
                cv2.putText(view, "A:opencv", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                if live_dbg is not None and live_dbg.contour is not None:
                    cv2.drawContours(view, [live_dbg.contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
                if live_dbg is not None and live_dbg.ok and live_dbg.points is not None:
                    draw_poly(view, live_dbg.points, color=(255, 0, 255), thickness=3)
                    cv2.putText(view, f"B:lines var={live_dbg.variant}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(view, f"B failed: {live_dbg.note if live_dbg else 'no dbg'}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            self._show_image(view)
            self._update_info(cand, live_dbg)

        elif self.view == 3:
            # View 4: binary raw
            crop = crop_bgr(img, cand.crop_box)
            v = self._pick_variant_for_debug(cand)
            raw = preprocess_variants(crop).get(v, preprocess_variants(crop)["edges"])
            view = gray_to_bgr(raw)
            cv2.putText(view, f"RAW variant={v}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            self._show_image(view)
            self._update_info(cand, live_dbg)

        else:
            # View 5: morph + edges + lines debug grid (live)
            crop = crop_bgr(img, cand.crop_box)
            v = self._pick_variant_for_debug(cand)

            dbg = live_dbg if live_dbg is not None else planb_lines_from_variant(
                crop, v, k=int(self.var_k.get()),
                close_it=int(self.var_close_it.get()),
                open_it=int(self.var_open_it.get()),
            )

            # panels
            crop_vis = crop.copy()
            if dbg.contour is not None:
                cv2.drawContours(crop_vis, [dbg.contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
            if dbg.ok and dbg.points is not None:
                draw_poly(crop_vis, dbg.points, color=(255, 0, 255), thickness=3)
            cv2.putText(crop_vis, f"crop | var={dbg.variant} | {dbg.note}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(crop_vis, f"k={self.var_k.get()} close={self.var_close_it.get()} open={self.var_open_it.get()}",
                        (10, crop_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            p_raw = gray_to_bgr(dbg.raw)
            cv2.putText(p_raw, "RAW", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_close = gray_to_bgr(dbg.close_img)
            cv2.putText(p_close, "CLOSE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_open = gray_to_bgr(dbg.open_img)
            cv2.putText(p_open, "OPEN", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_co = gray_to_bgr(dbg.close_open)
            cv2.putText(p_co, "CLOSE->OPEN (USED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            p_edges = gray_to_bgr(dbg.edges)
            cv2.putText(p_edges, f"EDGES (masked) | segs={len(dbg.lines)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # lines overlay
            lines_vis = gray_to_bgr(dbg.edges)
            # draw all segments
            for s in dbg.lines[:200]:
                cv2.line(lines_vis, (s.x1, s.y1), (s.x2, s.y2), (0, 255, 0), 2, cv2.LINE_AA)
            # draw chosen lines (extend across crop)
            Hc, Wc = dbg.edges.shape[:2]
            if dbg.ok and dbg.points is not None:
                draw_poly(lines_vis, dbg.points, (255, 0, 255), 3)

            def draw_line_abc(vis: np.ndarray, ln: LineABC, color: Tuple[int, int, int], thick: int = 2):
                # intersection with image borders by sampling two points
                # compute points for x=0 and x=W-1 (solve for y), fallback to y borders
                Wp, Hp = Wc, Hc
                a, b, c = ln.a, ln.b, ln.c
                pts = []
                if abs(b) > 1e-6:
                    y0 = int(round((-c - a * 0.0) / b))
                    y1 = int(round((-c - a * (Wp - 1.0)) / b))
                    pts = [(0, y0), (Wp - 1, y1)]
                elif abs(a) > 1e-6:
                    x0 = int(round((-c - b * 0.0) / a))
                    x1 = int(round((-c - b * (Hp - 1.0)) / a))
                    pts = [(x0, 0), (x1, Hp - 1)]
                if pts:
                    cv2.line(vis, pts[0], pts[1], color, thick, cv2.LINE_AA)

            for ln in dbg.chosen_lines:
                draw_line_abc(lines_vis, ln, (0, 0, 255), 2)

            cv2.putText(lines_vis, "HOUGH segs (green) + chosen lines (red) + quad (magenta)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            grid = make_grid([crop_vis, p_raw, p_close, p_open, p_co, p_edges, lines_vis], cols=2)
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

    def _update_info(self, cand: Optional[Candidate], dbg: Optional[PlanBLinesDebug]):
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
            lines.append(f"  variant={self.var_variant.get()}  k={self.var_k.get()}  close={self.var_close_it.get()}  open={self.var_open_it.get()}")

            if dbg is not None:
                lines.append("")
                lines.append(f"Live Plan B: var={dbg.variant} ok={dbg.ok} note={dbg.note}")
                lines.append(f"  segs={len(dbg.lines)}  chosen_lines={len(dbg.chosen_lines)}  contour={'yes' if dbg.contour is not None else 'no'}")
                if dbg.ok and dbg.points is not None:
                    rx, ry = estimate_rot_xy_deg_from_quad(dbg.points + np.array([cand.crop_box[0], cand.crop_box[1]], dtype=np.float32))
                    lines.append(f"  rot_x={rx:.2f} deg  rot_y={ry:.2f} deg")

        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(lines))


# ============================================================
# CLI / main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("GUI v2: Plan B with Hough lines + intersections + debug views.")
    ap.add_argument("--input", type=str, required=True, help="Folder with images (or single image).")
    ap.add_argument("--roi-config", type=str, required=True, help="ROI config JSON.")
    ap.add_argument("--runs-dir", type=str, default="runs/cnn_scratch", help="Runs directory to scan (best.pt).")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")

    # defaults you requested earlier
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
    app = RectifyGUIv2(
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
