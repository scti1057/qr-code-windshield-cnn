from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


# ----------------- basics -----------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def integral_image(x: np.ndarray) -> np.ndarray:
    return cv2.integral(x)


def window_mean(integ: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    s = float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
    area = float((x1 - x0) * (y1 - y0))
    return s / (area + 1e-9)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ----------------- scoring (same idea as GUI) -----------------
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
      heatmap_bgr (uint8)
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
    jxx = dx * dx
    jyy = dy * dy
    jxy = dx * dy
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
    aniso = (l1 - l2) / (l1 + l2 + 1e-6)
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


# ----------------- config -----------------
@dataclass
class Params:
    patch_size: int = 265
    score_max_dim: int = 1600

    win_min: int = 128
    win_max: int = 320
    n_scales: int = 3

    # scoring params
    w_edges: float = 1.0
    w_corners: float = 0.7
    w_grad: float = 0.6
    w_aniso: float = 0.8

    canny_low: int = 50
    canny_high: int = 150
    harris_k: float = 0.04
    harris_thr_rel: float = 0.01
    st_sigma: float = 2.0


def win_sizes(p: Params) -> List[int]:
    a = int(clamp(p.win_min, 24, 512))
    b = int(clamp(p.win_max, 24, 768))
    if a > b:
        a, b = b, a
    n = int(clamp(p.n_scales, 1, 6))
    if n == 1:
        return [b]
    vals = np.linspace(a, b, n)
    sizes = sorted({int(round(v)) for v in vals})
    return [max(24, s) for s in sizes]


def load_params(path: Path) -> Params:
    p = Params()
    if not path.exists():
        return p
    data = json.loads(path.read_text(encoding="utf-8"))
    for k, v in data.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def save_params(path: Path, p: Params):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(p), indent=2), encoding="utf-8")


# ----------------- optimizer -----------------
def objective_at_click(
    gray_s: np.ndarray,
    click_xy_s: Tuple[int, int],
    p: Params,
    *,
    crop_pad: int,
) -> Tuple[float, Dict]:
    """
    Evaluate objective on a LOCAL crop around click for speed.
    Objective = max mean(score_map) over window sizes at the click center.
    """
    cx, cy = click_xy_s
    sizes = win_sizes(p)
    max_win = max(sizes)

    pad = max(crop_pad, max_win // 2 + 10)
    h, w = gray_s.shape[:2]

    x0 = clamp(cx - pad, 0, w - 1)
    x1 = clamp(cx + pad, 0, w - 1)
    y0 = clamp(cy - pad, 0, h - 1)
    y1 = clamp(cy + pad, 0, h - 1)

    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return 0.0, {"best_win": None}

    crop = gray_s[y0:y1, x0:x1]
    local_cx = cx - x0
    local_cy = cy - y0

    score_map, heat = compute_score_map(
        crop,
        canny_low=int(p.canny_low),
        canny_high=int(p.canny_high),
        harris_k=float(p.harris_k),
        harris_thr_rel=float(p.harris_thr_rel),
        st_sigma=float(p.st_sigma),
        w_edges=float(p.w_edges),
        w_corners=float(p.w_corners),
        w_grad=float(p.w_grad),
        w_aniso=float(p.w_aniso),
    )

    integ = integral_image(score_map)

    best = -1.0
    best_win = None
    for win in sizes:
        half = win // 2
        if local_cx - half < 0 or local_cy - half < 0:
            continue
        if local_cx + half >= crop.shape[1] or local_cy + half >= crop.shape[0]:
            continue
        wx0 = local_cx - half
        wy0 = local_cy - half
        wx1 = wx0 + win
        wy1 = wy0 + win
        m = window_mean(integ, wx0, wy0, wx1, wy1)
        if m > best:
            best = m
            best_win = win

    meta = {
        "best_win": best_win,
        "crop_box_s": (x0, y0, x1, y1),
        "heat": heat,
        "score_map": score_map,
        "local_center": (int(local_cx), int(local_cy)),
    }
    return float(best if best > 0 else 0.0), meta


def random_search_optimize(
    gray_s: np.ndarray,
    click_xy_s: Tuple[int, int],
    p0: Params,
    *,
    iters: int,
    crop_pad: int,
    rng: np.random.Generator,
) -> Tuple[Params, float]:
    """
    Random search around current params (bounded).
    Optimizes scoring parameters only.
    """
    best_p = Params(**asdict(p0))
    best_val, _ = objective_at_click(gray_s, click_xy_s, best_p, crop_pad=crop_pad)

    # ranges
    def sample_from(base: Params) -> Params:
        p = Params(**asdict(base))

        # weights (0..3)
        p.w_edges = float(clamp(rng.normal(p.w_edges, 0.25), 0.0, 3.0))
        p.w_corners = float(clamp(rng.normal(p.w_corners, 0.35), 0.0, 3.0))
        p.w_grad = float(clamp(rng.normal(p.w_grad, 0.25), 0.0, 3.0))
        p.w_aniso = float(clamp(rng.normal(p.w_aniso, 0.35), 0.0, 3.0))

        # canny thresholds
        low = int(clamp(round(rng.normal(p.canny_low, 10)), 0, 200))
        high = int(clamp(round(rng.normal(p.canny_high, 12)), 10, 255))
        if high < low + 1:
            high = min(255, low + 1)
        p.canny_low = low
        p.canny_high = high

        # harris thr rel (0.001..0.02) log-ish
        ht = float(clamp(rng.normal(p.harris_thr_rel, 0.002), 0.001, 0.02))
        p.harris_thr_rel = ht

        # st_sigma (0.5..4.0)
        p.st_sigma = float(clamp(rng.normal(p.st_sigma, 0.3), 0.5, 4.0))

        return p

    for _ in range(iters):
        cand = sample_from(best_p)
        val, _ = objective_at_click(gray_s, click_xy_s, cand, crop_pad=crop_pad)
        if val > best_val:
            best_val = val
            best_p = cand

    return best_p, float(best_val)


# ----------------- GUI -----------------
class App:
    def __init__(self, files: List[Path], params: Params, cfg_in: Path, cfg_out: Path, display_max_dim: int):
        self.files = files
        self.idx = 0
        self.params0 = Params(**asdict(params))
        self.params = params
        self.cfg_in = cfg_in
        self.cfg_out = cfg_out
        self.display_max_dim = display_max_dim

        self.win_main = "ROI AutoTune (click to optimize)"
        self.win_heat = "Heatmap (scoring)"
        cv2.namedWindow(self.win_main, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.win_heat, cv2.WINDOW_NORMAL)

        self.last_click_s: Optional[Tuple[int, int]] = None
        self.last_val_before: Optional[float] = None
        self.last_val_after: Optional[float] = None
        self.last_best_win: Optional[int] = None

        self.rng = np.random.default_rng(123)

        cv2.setMouseCallback(self.win_main, self.on_mouse)

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        img = cv2.imread(str(self.files[self.idx]), cv2.IMREAD_COLOR)
        if img is None:
            return

        disp, disp_scale = resize_max_dim(img, self.display_max_dim)

        # map display click -> original
        ox = int(round(x / disp_scale))
        oy = int(round(y / disp_scale))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_s, s = resize_max_dim(gray, self.params.score_max_dim)
        cx_s = int(round(ox * s))
        cy_s = int(round(oy * s))
        cx_s = int(clamp(cx_s, 0, gray_s.shape[1] - 1))
        cy_s = int(clamp(cy_s, 0, gray_s.shape[0] - 1))
        self.last_click_s = (cx_s, cy_s)

        # objective before
        val_before, meta = objective_at_click(gray_s, self.last_click_s, self.params, crop_pad=260)
        self.last_val_before = val_before
        self.last_best_win = meta.get("best_win", None)

        # optimize (auto)
        print(f"[AutoTune] click at scoring=(x={cx_s}, y={cy_s}) val_before={val_before:.4f} best_win={self.last_best_win}")
        new_p, val_after = random_search_optimize(
            gray_s,
            self.last_click_s,
            self.params,
            iters=80,
            crop_pad=260,
            rng=self.rng,
        )
        self.params = new_p
        self.last_val_after = val_after

        # recompute best win after
        val_after2, meta2 = objective_at_click(gray_s, self.last_click_s, self.params, crop_pad=260)
        self.last_val_after = val_after2
        self.last_best_win = meta2.get("best_win", self.last_best_win)

        print(f"[AutoTune] val_after={self.last_val_after:.4f} best_win={self.last_best_win}")
        print(f"[AutoTune] updated params: w_edges={self.params.w_edges:.2f} w_corners={self.params.w_corners:.2f} w_grad={self.params.w_grad:.2f} w_aniso={self.params.w_aniso:.2f} "
              f"canny={self.params.canny_low}/{self.params.canny_high} h_thr={self.params.harris_thr_rel:.4f} st_sigma={self.params.st_sigma:.2f}")

    def draw(self):
        img_path = self.files[self.idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return

        disp, disp_scale = resize_max_dim(img, self.display_max_dim)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_s, s = resize_max_dim(gray, self.params.score_max_dim)

        score_map, heat = compute_score_map(
            gray_s,
            canny_low=int(self.params.canny_low),
            canny_high=int(self.params.canny_high),
            harris_k=float(self.params.harris_k),
            harris_thr_rel=float(self.params.harris_thr_rel),
            st_sigma=float(self.params.st_sigma),
            w_edges=float(self.params.w_edges),
            w_corners=float(self.params.w_corners),
            w_grad=float(self.params.w_grad),
            w_aniso=float(self.params.w_aniso),
        )

        # show click marker + window box on display
        if self.last_click_s is not None:
            cx_s, cy_s = self.last_click_s
            ox = int(round(cx_s / s))
            oy = int(round(cy_s / s))
            dx = int(round(ox * disp_scale))
            dy = int(round(oy * disp_scale))

            cv2.drawMarker(disp, (dx, dy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # draw window box (best_win) in display coords
            if self.last_best_win is not None:
                half_s = int(round(self.last_best_win / 2))
                x0s = cx_s - half_s
                y0s = cy_s - half_s
                x1s = cx_s + half_s
                y1s = cy_s + half_s

                # map to original->display
                x0o = int(round(x0s / s)); y0o = int(round(y0s / s))
                x1o = int(round(x1s / s)); y1o = int(round(y1s / s))

                x0d = int(round(x0o * disp_scale)); y0d = int(round(y0o * disp_scale))
                x1d = int(round(x1o * disp_scale)); y1d = int(round(y1o * disp_scale))

                cv2.rectangle(disp, (x0d, y0d), (x1d, y1d), (0, 255, 255), 2)

        # overlay text
        lines = [
            f"{img_path.name} [{self.idx+1}/{len(self.files)}]",
            f"Loaded: {self.cfg_in}",
            f"Save to: {self.cfg_out}",
            f"score_max_dim={self.params.score_max_dim} win={win_sizes(self.params)}",
            f"w: edges={self.params.w_edges:.2f} corners={self.params.w_corners:.2f} grad={self.params.w_grad:.2f} aniso={self.params.w_aniso:.2f}",
            f"canny={self.params.canny_low}/{self.params.canny_high} h_thr={self.params.harris_thr_rel:.4f} st_sigma={self.params.st_sigma:.2f}",
        ]
        if self.last_val_before is not None:
            lines.append(f"click score: before={self.last_val_before:.4f} after={self.last_val_after:.4f} best_win={self.last_best_win}")

        pad = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        th = 1
        sizes = [cv2.getTextSize(t, font, fs, th)[0] for t in (lines + ["[n]=next [p]=prev [r]=reset [w]=write cfg [q]=quit"])]
        box_w = max(s[0] for s in sizes) + 2 * pad
        box_h = (len(lines) + 1) * (sizes[0][1] + 8) + 2 * pad
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        disp = cv2.addWeighted(overlay, 0.55, disp, 0.45, 0)

        y = pad + sizes[0][1]
        for t in lines:
            cv2.putText(disp, t, (pad, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            y += sizes[0][1] + 8
        cv2.putText(disp, "[n]=next [p]=prev [r]=reset [w]=write cfg [q]=quit", (pad, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)

        # show
        cv2.imshow(self.win_main, disp)

        heat_disp, _ = resize_max_dim(heat, self.display_max_dim)
        cv2.imshow(self.win_heat, heat_disp)

    def run(self):
        while True:
            self.draw()
            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("n"):
                self.idx = (self.idx + 1) % len(self.files)
                self.last_click_s = None
                self.last_val_before = None
                self.last_val_after = None
                self.last_best_win = None
            if k == ord("p"):
                self.idx = (self.idx - 1) % len(self.files)
                self.last_click_s = None
                self.last_val_before = None
                self.last_val_after = None
                self.last_best_win = None
            if k == ord("r"):
                self.params = Params(**asdict(self.params0))
                print("[Reset] params reset to loaded config")
            if k == ord("w"):
                save_params(self.cfg_out, self.params)
                print(f"[OK] wrote config: {self.cfg_out}")

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Folder or single image")
    ap.add_argument("--load-config", type=str, required=True, help="JSON config to load as start values")
    ap.add_argument("--save-config", type=str, default="configs/roi_autotuned.json", help="Where to write tuned config")
    ap.add_argument("--display-max-dim", type=int, default=1200)
    args = ap.parse_args()

    files = iter_image_files(Path(args.input))
    if not files:
        raise SystemExit(f"No images found: {args.input}")

    cfg_in = Path(args.load_config)
    if not cfg_in.is_absolute():
        cfg_in = repo_root() / cfg_in

    cfg_out = Path(args.save_config)
    if not cfg_out.is_absolute():
        cfg_out = repo_root() / cfg_out

    p = load_params(cfg_in)
    print(f"[OK] loaded start config: {cfg_in}")

    app = App(files, p, cfg_in=cfg_in, cfg_out=cfg_out, display_max_dim=args.display_max_dim)
    app.run()


if __name__ == "__main__":
    main()
