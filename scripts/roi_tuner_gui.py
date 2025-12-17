from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np


# ---------- Utils ----------
def iter_image_files(input_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in exts else []
    out = []
    for p in sorted(input_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def resize_max_dim(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    s = max_dim / float(m)
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out, s


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


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
    # sort by score desc
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


# ---------- ROI scoring (rotation-invariant “parallel edges” via anisotropy) ----------
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      score_map (0..1 float32)
      heatmap_bgr (uint8) for display
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)

    # edges
    edges = cv2.Canny(g, canny_low, canny_high)
    edges01 = edges.astype(np.float32) / 255.0

    # corners (Harris)
    g32 = (g.astype(np.float32) / 255.0)
    harris = cv2.cornerHarris(g32, blockSize=2, ksize=3, k=harris_k)
    harris = cv2.dilate(harris, None)
    thr = harris_thr_rel * float(harris.max()) if float(harris.max()) > 0 else 0.0
    corners01 = (harris > thr).astype(np.float32)

    # gradient magnitude
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    mag01 = normalize01(mag)

    # structure-tensor anisotropy (rotation invariant)
    # parallel edges -> strong orientation -> high anisotropy
    jxx = dx * dx
    jyy = dy * dy
    jxy = dx * dy
    # smooth tensor components
    ksize = int(max(3, round(st_sigma * 6))) | 1
    jxx = cv2.GaussianBlur(jxx, (ksize, ksize), st_sigma)
    jyy = cv2.GaussianBlur(jyy, (ksize, ksize), st_sigma)
    jxy = cv2.GaussianBlur(jxy, (ksize, ksize), st_sigma)

    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    root = np.sqrt(disc)
    l1 = 0.5 * (tr + root)
    l2 = 0.5 * (tr - root)
    aniso = (l1 - l2) / (l1 + l2 + 1e-6)   # 0..1
    aniso01 = normalize01(aniso)

    score = (
        w_edges * edges01 +
        w_corners * corners01 +
        w_grad * mag01 +
        w_aniso * aniso01
    )
    score = normalize01(score)

    heat = (score * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return score, heat_bgr


def integral_image(x: np.ndarray) -> np.ndarray:
    return cv2.integral(x)


def window_mean(integ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    # integral has shape (h+1, w+1)
    s = float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
    area = float((x1 - x0) * (y1 - y0))
    return s / (area + 1e-9)


def scan_windows(
    integ: np.ndarray,
    win: int,
    stride: int,
    score_thr: float,
    top_k: int,
) -> List[Dict]:
    h1, w1 = integ.shape[0] - 1, integ.shape[1] - 1
    half = win // 2
    cands: List[Dict] = []

    for cy in range(half, h1 - half, stride):
        for cx in range(half, w1 - half, stride):
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + win, y0 + win
            m = window_mean(integ, x0, y0, x1, y1)
            if m >= score_thr:
                cands.append({
                    "cx_s": cx, "cy_s": cy,
                    "score": float(m),
                    "win": int(win),
                    "box_s": (x0, y0, x1, y1),
                })

    # keep top_k by score
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

    # 1) coarse scan
    coarse_cands: List[Dict] = []
    for win in win_sizes:
        cc = scan_windows(integ, win, stride_coarse, score_thr, top_k_coarse)
        coarse_cands.extend(cc)

    # optional NMS already on coarse
    coarse_cands = nms(coarse_cands, iou_thresh, top_k=max(top_k_coarse, 200))

    # 2) fine refine around each coarse candidate
    h, w = score_map.shape[:2]
    for c in coarse_cands:
        cx0, cy0 = c["cx_s"], c["cy_s"]
        win = c["win"]
        half = win // 2

        x_min = max(half, cx0 - refine_radius)
        x_max = min(w - half - 1, cx0 + refine_radius)
        y_min = max(half, cy0 - refine_radius)
        y_max = min(h - half - 1, cy0 + refine_radius)

        local: List[Dict] = []
        for cy in range(y_min, y_max + 1, stride_fine):
            for cx in range(x_min, x_max + 1, stride_fine):
                x0, y0 = cx - half, cy - half
                x1, y1 = x0 + win, y0 + win
                m = window_mean(integ, x0, y0, x1, y1)
                if m >= score_thr:
                    local.append({
                        "cx_s": cx, "cy_s": cy,
                        "score": float(m),
                        "win": int(win),
                        "box_s": (x0, y0, x1, y1),
                    })
        local.sort(key=lambda d: d["score"], reverse=True)
        all_cands.extend(local[:top_k_fine_each])

    # merge + NMS
    all_cands.sort(key=lambda d: d["score"], reverse=True)
    all_cands = nms(all_cands, iou_thresh, top_k=top_k_final)

    return all_cands


# ---------- GUI state ----------
@dataclass
class GuiParams:
    # patch output size (fixed box you later crop)
    patch_size: int = 265

    # scoring resize
    score_max_dim: int = 1600

    # multi-scale window sizes (in scoring space)
    win_min: int = 128
    win_max: int = 320
    n_scales: int = 3

    # coarse-to-fine
    stride_coarse: int = 96
    stride_fine: int = 24
    refine_radius: int = 80
    top_k_coarse: int = 60
    top_k_fine_each: int = 20
    top_k_final: int = 120
    iou_thresh: float = 0.25

    # threshold on window mean score (0..1)
    score_thr: float = 0.35

    # scoring weights
    w_edges: float = 1.00
    w_corners: float = 0.70
    w_grad: float = 0.60
    w_aniso: float = 0.80

    # canny
    canny_low: int = 50
    canny_high: int = 150

    # harris
    harris_k: float = 0.04
    harris_thr_rel: float = 0.01

    # structure tensor smoothing
    st_sigma: float = 2.0


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def make_win_sizes(win_min: int, win_max: int, n_scales: int) -> List[int]:
    win_min = clamp_int(win_min, 32, 512)
    win_max = clamp_int(win_max, 32, 768)
    if win_min > win_max:
        win_min, win_max = win_max, win_min
    n_scales = clamp_int(n_scales, 1, 6)
    if n_scales == 1:
        return [int(win_max)]
    vals = np.linspace(win_min, win_max, n_scales)
    sizes = sorted({int(round(v)) for v in vals})
    return [max(32, s) for s in sizes]


def draw_overlay(
    img_bgr: np.ndarray,
    *,
    cands_orig: List[Dict],
    patch_size: int,
    title_lines: List[str],
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    half = patch_size // 2

    # draw boxes: best -> green, rest -> yellow
    for i, c in enumerate(cands_orig):
        cx, cy = c["cx"], c["cy"]
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + patch_size, y0 + patch_size
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)

        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"{c['score']:.3f}", (x0 + 4, max(15, y0 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # text box
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    th = 1
    lines = title_lines + ["[n]=next  [p]=prev  [w]=write cfg  [q]=quit"]
    sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = (len(lines) * (sizes[0][1] + 8)) + 2 * pad
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    y = pad + sizes[0][1]
    for t in lines:
        cv2.putText(out, t, (pad, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += sizes[0][1] + 8

    return out


# ---------- Trackbar helpers ----------
def tb(name: str, win: str, init: int, maxv: int):
    cv2.createTrackbar(name, win, init, maxv, lambda _x: None)


def get_tb(name: str, win: str) -> int:
    return cv2.getTrackbarPos(name, win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Folder or single image")
    ap.add_argument("--display-max-dim", type=int, default=1200)
    ap.add_argument("--save-config", type=str, default="configs/roi_tuner_params.json")
    ap.add_argument("--load-config", type=str, default="", help="Optional JSON to override defaults at startup.")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    cfg_arg = args.load_config.strip()
    cfg_in = Path(cfg_arg)
    if cfg_arg:
        # wenn relativ: relativ zum repo_root statt cwd
        if not cfg_in.is_absolute():
            cfg_in = repo_root / cfg_in


    in_path = Path(args.input)
    files = iter_image_files(in_path)
    if not files:
        raise SystemExit(f"No images found in: {in_path}")

    cfg_path = Path(args.save_config)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # windows
    win_main = "ROI Tuner"
    win_heat = "Score Heatmap"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_heat, cv2.WINDOW_NORMAL)

    # init params
    P = GuiParams()
    # optional: load overrides
    if cfg_arg:
        if cfg_in.exists():
            data = json.loads(cfg_in.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(P, k):
                    setattr(P, k, v)
            print(f"[OK] loaded config overrides: {cfg_in}")
        else:
            print(f"[WARN] config not found: {cfg_in} (using defaults)")



    # trackbars (ints only)
    tb("score_max_dim", win_main, P.score_max_dim, 4000)
    tb("win_min",       win_main, P.win_min, 512)
    tb("win_max",       win_main, P.win_max, 768)
    tb("n_scales",      win_main, P.n_scales, 6)

    tb("stride_coarse", win_main, P.stride_coarse, 256)
    tb("stride_fine",   win_main, P.stride_fine, 64)
    tb("refine_radius", win_main, P.refine_radius, 400)

    tb("top_k_coarse",  win_main, P.top_k_coarse, 300)
    tb("top_k_fine",    win_main, P.top_k_fine_each, 80)
    tb("top_k_final",   win_main, P.top_k_final, 400)

    tb("iou_x100",      win_main, int(P.iou_thresh * 100), 100)
    tb("thr_x100",      win_main, int(P.score_thr * 100), 100)

    tb("w_edges_x100",  win_main, int(P.w_edges * 100), 300)
    tb("w_corn_x100",   win_main, int(P.w_corners * 100), 300)
    tb("w_grad_x100",   win_main, int(P.w_grad * 100), 300)
    tb("w_aniso_x100",  win_main, int(P.w_aniso * 100), 300)

    tb("canny_low",     win_main, P.canny_low, 255)
    tb("canny_high",    win_main, P.canny_high, 255)

    tb("h_thr_x1000",   win_main, int(P.harris_thr_rel * 1000), 50)  # 0..0.05
    tb("st_sigma_x10",  win_main, int(P.st_sigma * 10), 80)          # 0..8.0

    idx = 0
    last_hash = None
    cached = None

    while True:
        img_path = files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] unreadable: {img_path}")
            idx = (idx + 1) % len(files)
            continue

        # read params from UI
        P.score_max_dim = clamp_int(get_tb("score_max_dim", win_main), 256, 4000)
        P.win_min = clamp_int(get_tb("win_min", win_main), 32, 512)
        P.win_max = clamp_int(get_tb("win_max", win_main), 32, 768)
        P.n_scales = clamp_int(get_tb("n_scales", win_main), 1, 6)

        P.stride_coarse = clamp_int(get_tb("stride_coarse", win_main), 8, 512)
        P.stride_fine = clamp_int(get_tb("stride_fine", win_main), 1, 128)
        P.refine_radius = clamp_int(get_tb("refine_radius", win_main), 0, 9999)

        P.top_k_coarse = clamp_int(get_tb("top_k_coarse", win_main), 1, 9999)
        P.top_k_fine_each = clamp_int(get_tb("top_k_fine", win_main), 1, 9999)
        P.top_k_final = clamp_int(get_tb("top_k_final", win_main), 1, 9999)

        P.iou_thresh = float(clamp_int(get_tb("iou_x100", win_main), 0, 100)) / 100.0
        P.score_thr = float(clamp_int(get_tb("thr_x100", win_main), 0, 100)) / 100.0

        P.w_edges = float(get_tb("w_edges_x100", win_main)) / 100.0
        P.w_corners = float(get_tb("w_corn_x100", win_main)) / 100.0
        P.w_grad = float(get_tb("w_grad_x100", win_main)) / 100.0
        P.w_aniso = float(get_tb("w_aniso_x100", win_main)) / 100.0

        P.canny_low = clamp_int(get_tb("canny_low", win_main), 0, 255)
        P.canny_high = clamp_int(get_tb("canny_high", win_main), 0, 255)
        if P.canny_high < P.canny_low:
            P.canny_high = P.canny_low

        P.harris_thr_rel = float(get_tb("h_thr_x1000", win_main)) / 1000.0
        P.st_sigma = float(get_tb("st_sigma_x10", win_main)) / 10.0
        if P.st_sigma < 0.1:
            P.st_sigma = 0.1

        # cache key: recompute only if params or image changed
        key = (idx, tuple(sorted(asdict(P).items())))
        key_hash = hash(key)

        if key_hash != last_hash:
            # scoring image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_s, s = resize_max_dim(gray, P.score_max_dim)

            score_map, heat = compute_score_map(
                gray_s,
                canny_low=P.canny_low,
                canny_high=P.canny_high,
                harris_k=0.04,
                harris_thr_rel=P.harris_thr_rel,
                st_sigma=P.st_sigma,
                w_edges=P.w_edges,
                w_corners=P.w_corners,
                w_grad=P.w_grad,
                w_aniso=P.w_aniso,
            )

            win_sizes = make_win_sizes(P.win_min, P.win_max, P.n_scales)

            cands_s = coarse_to_fine(
                score_map=score_map,
                win_sizes=win_sizes,
                stride_coarse=P.stride_coarse,
                stride_fine=P.stride_fine,
                refine_radius=P.refine_radius,
                score_thr=P.score_thr,
                top_k_coarse=P.top_k_coarse,
                top_k_fine_each=P.top_k_fine_each,
                iou_thresh=P.iou_thresh,
                top_k_final=P.top_k_final,
            )

            # map candidates to original coordinates (center only)
            cands_o: List[Dict] = []
            for c in cands_s:
                cx = int(round(c["cx_s"] / s))
                cy = int(round(c["cy_s"] / s))
                cands_o.append({"cx": cx, "cy": cy, "score": c["score"], "win": c["win"]})

            cached = (cands_o, heat, win_sizes, s)
            last_hash = key_hash

        cands_o, heat, win_sizes, s = cached

        # display
        disp, disp_scale = resize_max_dim(img, args.display_max_dim)
        # scale candidate centers to display size
        cands_disp = []
        for c in cands_o:
            cands_disp.append({
                "cx": int(round(c["cx"] * disp_scale)),
                "cy": int(round(c["cy"] * disp_scale)),
                "score": c["score"],
                "win": c["win"],
            })

        title_lines = [
            f"{img_path.name}  [{idx+1}/{len(files)}]",
            f"thr={P.score_thr:.2f} iou={P.iou_thresh:.2f} scales={win_sizes}",
            f"coarse stride={P.stride_coarse} fine stride={P.stride_fine} radius={P.refine_radius}",
            f"weights: edges={P.w_edges:.2f} corners={P.w_corners:.2f} grad={P.w_grad:.2f} aniso={P.w_aniso:.2f}",
            f"cands_shown={min(len(cands_disp), P.top_k_final)} (cap={P.top_k_final})",
        ]

        overlay = draw_overlay(disp, cands_orig=cands_disp, patch_size=int(round(P.patch_size * disp_scale)), title_lines=title_lines)
        cv2.imshow(win_main, overlay)

        heat_disp, _ = resize_max_dim(heat, args.display_max_dim)
        cv2.imshow(win_heat, heat_disp)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("n"):
            idx = (idx + 1) % len(files)
            last_hash = None
        if k == ord("p"):
            idx = (idx - 1) % len(files)
            last_hash = None
        if k == ord("w"):
            # write config
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(P), f, indent=2)
            print(f"[OK] wrote config: {cfg_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
