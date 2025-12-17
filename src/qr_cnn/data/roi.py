from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import cv2
import numpy as np


class DebugViewer:
    def __init__(self, enabled: bool, window_name: str = "ROI Debug", max_dim: int = 1200):
        self.enabled = enabled
        self.window_name = window_name
        self.max_dim = max_dim
        self.abort = False
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _resize_vis(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        m = max(h, w)
        if m <= self.max_dim:
            return img_bgr
        s = self.max_dim / float(m)
        return cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def _overlay_text(self, img_bgr: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
        out = img_bgr.copy()
        pad = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        text_lines = [title] + lines + ["[SPACE]=weiter  [q/ESC]=abbrechen"]
        sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in text_lines]
        box_w = max(s[0] for s in sizes) + 2 * pad
        box_h = (len(text_lines) * (sizes[0][1] + 8)) + 2 * pad

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

        y = pad + sizes[0][1]
        for t in text_lines:
            cv2.putText(out, t, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += sizes[0][1] + 8

        return out

    def show(self, img_bgr: np.ndarray, title: str, lines: list[str] | None = None):
        if not self.enabled or self.abort:
            return
        if lines is None:
            lines = []

        vis = self._resize_vis(img_bgr)
        vis = self._overlay_text(vis, title, lines)

        while True:
            cv2.imshow(self.window_name, vis)
            k = cv2.waitKey(0) & 0xFF
            if k == 32:  # SPACE
                break
            if k in (27, ord("q")):  # ESC oder q
                self.abort = True
                break

    def close(self):
        if self.enabled:
            cv2.destroyWindow(self.window_name)


@dataclass
class RoiParams:
    patch_size: int = 265
    stride: int = 96                 # fürs coarse tiling / scoring grid
    top_k: int = 20                  # wie viele Patches pro Bild
    iou_thresh: float = 0.25         # NMS-artige Entdoppelung
    score_max_dim: int = 1600        # Bild wird zum Scoring ggf. runter skaliert
    min_edge_density: float = 0.01   # grober Filter: wenn Patch zu "leer", skip
    mode: str = "hybrid"             # candidates | tiling | hybrid

    # Score-Gewichte
    w_edges: float = 1.0
    w_corners: float = 0.7
    w_grad: float = 0.6


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _to_bgr_vis(x: np.ndarray) -> np.ndarray:
    """macht aus 1-channel/float arrays ein schönes BGR Debug-Bild."""
    if x.ndim == 2:
        y = x
        if y.dtype != np.uint8:
            y = _normalize01(y)
            y = (y * 255).astype(np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    if x.ndim == 3 and x.shape[2] == 3:
        if x.dtype != np.uint8:
            y = np.clip(x, 0, 255).astype(np.uint8)
            return y
        return x
    raise ValueError("Unsupported debug image shape")


def iter_image_files(input_path: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
        return
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _resize_for_scoring(gray: np.ndarray, score_max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= score_max_dim:
        return gray, 1.0
    scale = score_max_dim / float(m)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _compute_score_map(gray: np.ndarray, params: RoiParams, debug: DebugViewer | None = None) -> np.ndarray:
    """
    Score map (0..1): Kombination aus edges + corners + gradient magnitude
    """
    if debug:
        debug.show(_to_bgr_vis(gray), "Step: Input gray", [f"shape={gray.shape} dtype={gray.dtype}"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug:
        debug.show(_to_bgr_vis(g), "Step: GaussianBlur", ["kernel=(5,5), sigma=0"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)

    # edges (auto thresholds from median)
    v = np.median(g)
    t1 = int(max(0, 0.66 * v))
    t2 = int(min(255, 1.33 * v))
    edges = cv2.Canny(g, t1, t2)
    edges01 = edges.astype(np.float32) / 255.0
    if debug:
        debug.show(_to_bgr_vis(edges), "Step: Canny edges", [f"median={v:.2f}", f"t1={t1}, t2={t2}"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)

    # corners (Harris)
    g32 = np.float32(g) / 255.0
    harris = cv2.cornerHarris(g32, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    thr = 0.01 * harris.max() if harris.max() > 0 else 0.0
    corners = (harris > thr).astype(np.float32)

    if debug:
        dbg = _to_bgr_vis(_normalize01(harris))
        debug.show(dbg, "Step: Harris response", ["blockSize=2 ksize=3 k=0.04", f"thr={thr:.6f}"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)
        debug.show(_to_bgr_vis(corners), "Step: Harris corners mask", ["mask = harris > thr"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)

    # gradient magnitude (Sobel)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    mag01 = _normalize01(mag)

    if debug:
        debug.show(_to_bgr_vis(mag01), "Step: Gradient magnitude (Sobel)", ["ksize=3", "normalized to 0..1"])
        if debug.abort:
            return np.zeros_like(gray, dtype=np.float32)

    score = (
        params.w_edges * edges01 +
        params.w_corners * corners +
        params.w_grad * mag01
    )
    score = _normalize01(score)

    if debug:
        heat = (score * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        debug.show(
            heat,
            "Step: Final score map",
            [f"weights: edges={params.w_edges}, corners={params.w_corners}, grad={params.w_grad}"],
        )

    return score


def _integral_sum(integ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    # integ shape: (h+1, w+1)
    return float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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
    union = area_a + area_b - inter
    return inter / float(union + 1e-9)


def _pad_and_crop(img: np.ndarray, cx: int, cy: int, patch_size: int) -> np.ndarray:
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
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    patch = img[y0:y1, x0:x1]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    return patch


def propose_patch_centers(img_bgr: np.ndarray, params: RoiParams, debug: DebugViewer | None = None) -> List[Dict]:
    """
    Liefert eine Liste von Kandidaten-Patches als Dict:
    { 'cx','cy','score','box':(x0,y0,x1,y1) } in ORIGINAL-Koordinaten.
    """
    h0, w0 = img_bgr.shape[:2]
    gray0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray_s, scale = _resize_for_scoring(gray0, params.score_max_dim)
    hs, ws = gray_s.shape[:2]
    ps_s = max(32, int(round(params.patch_size * scale)))  # patch size in scoring space
    stride = max(16, int(round(params.stride * scale)))
    half = ps_s // 2

    if debug:
        debug.show(
            _to_bgr_vis(gray_s),
            "Step: Scoring image (resized)",
            [
                f"orig={gray0.shape[::-1]} resized={gray_s.shape[::-1]}",
                f"scale={scale:.4f}",
                f"patch_s={ps_s}px",
                f"stride_s={stride}px",
            ],
        )
        if debug.abort:
            return []

    score_map = _compute_score_map(gray_s, params, debug=debug)
    if debug and debug.abort:
        return []

    integ = cv2.integral(score_map)

    windows = []
    for cy in range(half, hs - half, stride):
        for cx in range(half, ws - half, stride):
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + ps_s, y0 + ps_s
            s = _integral_sum(integ, x0, y0, x1, y1)
            windows.append((s, cx, cy, (x0, y0, x1, y1)))

    windows.sort(key=lambda t: t[0], reverse=True)

    if debug:
        vis = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)
        for (s, cx, cy, (x0, y0, x1, y1)) in windows[:30]:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
        debug.show(vis, "Step: Top window candidates (pre-NMS)", [f"showing top=30 of {len(windows)} windows"])
        if debug.abort:
            return []

    selected = []
    selected_boxes = []
    for s, cx, cy, box in windows:
        if len(selected) >= params.top_k:
            break
        ok = True
        for sb in selected_boxes:
            if _iou(box, sb) > params.iou_thresh:
                ok = False
                break
        if not ok:
            continue
        selected.append((s, cx, cy))
        selected_boxes.append(box)

    if debug:
        vis = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)
        for (s, cx, cy), box in zip(selected, selected_boxes):
            x0, y0, x1, y1 = box
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        debug.show(vis, "Step: Selected windows (post-NMS)", [f"selected={len(selected)}", f"iou_thresh={params.iou_thresh}"])
        if debug.abort:
            return []

    # Map back to original coordinates
    out = []
    for s, cx_s, cy_s in selected:
        cx0 = int(round(cx_s / scale))
        cy0 = int(round(cy_s / scale))

        half0 = params.patch_size // 2
        x0, y0 = cx0 - half0, cy0 - half0
        x1, y1 = x0 + params.patch_size, y0 + params.patch_size

        out.append({
            "cx": int(np.clip(cx0, 0, w0 - 1)),
            "cy": int(np.clip(cy0, 0, h0 - 1)),
            "score": float(s),
            "box": (x0, y0, x1, y1),
        })

    return out


def tiling_patch_centers(img_bgr: np.ndarray, params: RoiParams) -> List[Dict]:
    h, w = img_bgr.shape[:2]
    half = params.patch_size // 2
    stride = params.stride
    out = []
    for cy in range(half, h - half, stride):
        for cx in range(half, w - half, stride):
            out.append({
                "cx": cx,
                "cy": cy,
                "score": 0.0,
                "box": (cx - half, cy - half, cx + half, cy + half),
            })
    return out


def extract_patches(img_bgr: np.ndarray, params: RoiParams, debug: DebugViewer | None = None) -> List[Dict]:
    """
    Gibt Patches + Metadaten zurück:
    [
      {'patch': np.ndarray(BGR, patch_size x patch_size), 'cx','cy','score','box', ...}
    ]
    """
    mode = params.mode.lower()
    if mode not in {"candidates", "tiling", "hybrid"}:
        raise ValueError(f"Unbekannter mode={params.mode}")

    if debug:
        debug.show(
            _to_bgr_vis(img_bgr),
            "Step: Input image (BGR)",
            [f"shape={img_bgr.shape}", f"mode={params.mode}", f"patch_size={params.patch_size}"],
        )
        if debug.abort:
            return []

    if mode == "tiling":
        centers = tiling_patch_centers(img_bgr, params)
    else:
        centers = propose_patch_centers(img_bgr, params, debug=debug)
        if debug and debug.abort:
            return []
        if mode == "hybrid" and len(centers) < max(5, params.top_k // 4):
            centers += tiling_patch_centers(img_bgr, params)

    patches = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for idx, c in enumerate(centers):
        if debug and debug.abort:
            break

        cx, cy = c["cx"], c["cy"]
        patch = _pad_and_crop(img_bgr, cx, cy, params.patch_size)

        if debug:
            debug.show(
                patch,
                "Step: Cropped patch",
                [
                    f"idx={idx}",
                    f"cx={cx}, cy={cy}",
                    f"score={float(c.get('score', 0.0)):.4f}",
                    f"patch={params.patch_size}x{params.patch_size}",
                ],
            )
            if debug.abort:
                break

        # Edge-density filter
        pgray = _pad_and_crop(gray, cx, cy, params.patch_size)
        edges = cv2.Canny(pgray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        accepted = edge_density >= params.min_edge_density

        if debug:
            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            debug.show(
                edges_vis,
                "Step: Patch edges (Canny)",
                [f"t1=50 t2=150", f"edge_density={edge_density:.4f}", f"min_edge_density={params.min_edge_density}", f"accepted={accepted}"],
            )
            if debug.abort:
                break

        if not accepted:
            continue

        patches.append({
            "patch": patch,
            "cx": cx,
            "cy": cy,
            "score": float(c.get("score", 0.0)),
            "box": c["box"],
            "edge_density": edge_density,
        })

    # bei candidates/hybrid: nach score sortieren und top_k halten
    if mode != "tiling":
        patches.sort(key=lambda d: d["score"], reverse=True)
        patches = patches[:params.top_k]

    if debug and patches and not debug.abort:
        boxes = [p["box"] for p in patches]
        dbg_img = draw_boxes(img_bgr, boxes)
        debug.show(
            dbg_img,
            "Step: Final accepted patches (orig image)",
            [f"kept={len(patches)}", f"mode={params.mode}", f"top_k={params.top_k}"],
        )

    return patches


def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for (x0, y0, x1, y1) in boxes:
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
    return out
