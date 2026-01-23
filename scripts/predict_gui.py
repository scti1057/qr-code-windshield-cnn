from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# src-layout import
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from qr_cnn.data.roi_c2f import (  # type: ignore
    RoiC2FConfig,
    iter_image_files,
    propose_centers_c2f,
    pad_and_crop,
)

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore


"""
GUI: ROI -> CNN prediction viewer (3 steps) + model picker + threshold slider

Steps:
1) show original image
2) show ROI boxes (yellow)
3) show ROI boxes colored by CNN prediction (green=QR, red=no_qr) + probability text

Controls:
- SPACE / n : next step
- b         : previous step
- 1/2/3     : jump to step
- j / RIGHT : next image
- k / LEFT  : previous image
- r         : recompute current image (ROI + CNN) for current model
- mouse LMB : click a box to open patch preview (+prob)
- q / ESC   : quit

Model selection:
- Trackbar "model": selects a checkpoint found under runs/cnn_scratch (best.pt / last.pt)

Threshold:
- Trackbar "thr(%)": sets threshold in percent. thr = value / 100.0
- Image is QR if ANY patch prob >= thr

Example usage:
python3 scripts/predict_gui.py \
  --input data/raw \
  --roi-config configs/roi_tuner_params.json \
  --runs-dir runs/cnn_scratch \
  --device mps
"""


# ------------------ tiny model definition (must match training) ------------------
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
        layers: List[nn.Module] = [
            nn.Conv2d(cin, cout, kernel_size=k, padding=padding, bias=not bn)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(make_activation(act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TinyQRNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        blocks_per_stage: List[int] = [2, 2, 2, 2],
        kernel_size: int = 3,
        activation: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.2,
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
        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if (global_pool or "avg").lower() == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(cin, int(num_outputs))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.head(x)


# ------------------ config/model loading ------------------
def parse_args():
    p = argparse.ArgumentParser(description="GUI viewer: ROI->CNN predictions + model picker + threshold slider.")
    p.add_argument("--input", type=str, required=True, help="Folder with images (or single image).")
    p.add_argument("--roi-config", type=str, required=True, help="ROI config JSON (from ROI tuner GUI).")

    p.add_argument("--runs-dir", type=str, default="runs/cnn_scratch",
                   help="Where to search for models (default: runs/cnn_scratch).")
    p.add_argument("--model", type=str, default="",
                   help="Optional: add a specific model path (best.pt/last.pt/torchscript).")
    p.add_argument("--device", type=str, default="", help="cuda | cpu | mps (default: auto)")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-patches", type=int, default=0, help="0 => cfg.top_k_final, else cap patches per image.")

    # normalization (defaults match our training pipeline: only /255.0)
    p.add_argument("--mean", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="RGB mean (default 0,0,0).")
    p.add_argument("--std", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="RGB std (default 1,1,1).")

    # visualization
    p.add_argument("--max-dim", type=int, default=1400, help="Max display size for window (keeps aspect).")
    p.add_argument("--font-scale", type=float, default=0.65)
    p.add_argument("--start-step", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--threshold", type=float, default=0.5, help="Initial threshold (will be controlled by slider).")

    # model caching
    p.add_argument("--max-loaded-models", type=int, default=3,
                   help="How many models to keep loaded in memory (default: 3).")
    return p.parse_args()


def pick_device(device_str: str):
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    if device_str:
        d = device_str.lower()
        if d == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if d == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(d)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config_any(path: Path) -> Dict:
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
        if k == "top_k_fine":  # backward compat
            cfg.top_k_fine_each = v
    return cfg


def find_run_config_near_model(model_path: Path) -> Optional[Path]:
    """
    Prefer config_used.yaml/json living next to best.pt/last.pt in the run directory.
    """
    run_dir = model_path.parent
    for name in ["config_used.yaml", "config_used.yml", "config_used.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None


def load_model_any(model_path: Path, device):
    """
    Supports:
    - TorchScript via torch.jit.load
    - Checkpoint dict with keys: model_state or state_dict (+ uses config_used.yaml next to model if available)
    """
    assert torch is not None and nn is not None

    # 1) try torchscript
    try:
        m = torch.jit.load(str(model_path), map_location=device)
        m.eval()
        return m, {"type": "torchscript", "path": str(model_path)}
    except Exception:
        pass

    # 2) checkpoint dict
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
        ckpt.eval()
        return ckpt.to(device), {"type": "torchscript", "path": str(model_path)}

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported model file: {model_path}")

    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise RuntimeError(f"Checkpoint has no 'model_state'/'state_dict': {model_path}")

    # Try to load model config from config_used.yaml/json next to the model
    model_cfg = {}
    cfg_used = find_run_config_near_model(model_path)
    if cfg_used and cfg_used.exists():
        try:
            full = load_config_any(cfg_used)
            if isinstance(full, dict):
                model_cfg = full.get("model") or {}
        except Exception:
            model_cfg = {}

    # fallback: checkpoint config_path (if any)
    if not model_cfg:
        cfg_path = ckpt.get("config_path")
        if cfg_path:
            p = Path(str(cfg_path))
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists():
                try:
                    full = load_config_any(p)
                    if isinstance(full, dict):
                        model_cfg = full.get("model") or {}
                except Exception:
                    model_cfg = {}

    def get(k, default):
        return model_cfg.get(k, default)

    net = TinyQRNet(
        in_channels=int(get("in_channels", 3)),
        base_channels=int(get("base_channels", 32)),
        blocks_per_stage=list(get("blocks_per_stage", [2, 2, 2, 2])),
        kernel_size=int(get("kernel_size", 3)),
        activation=str(get("activation", "relu")),
        batch_norm=bool(get("batch_norm", True)),
        dropout=float(get("dropout", 0.2)),
        num_outputs=int(get("num_outputs", 1)),
        global_pool=str(get("global_pool", "avg")),
    )

    missing, unexpected = net.load_state_dict(state, strict=False)
    net = net.to(device)
    net.eval()

    info = {
        "type": "checkpoint",
        "path": str(model_path),
        "config_used_nearby": str(cfg_used) if cfg_used else "",
        "missing_keys": list(missing) if isinstance(missing, (list, tuple)) else [],
        "unexpected_keys": list(unexpected) if isinstance(unexpected, (list, tuple)) else [],
        "model_cfg_used": model_cfg if model_cfg else {},
    }
    return net, info


# ------------------ tensor/prob utils ------------------
def patches_to_tensor(patches_bgr: List[np.ndarray], mean: List[float], std: List[float], device):
    assert torch is not None
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
    assert torch is not None
    if isinstance(y, (list, tuple)):
        y = y[0]
    if isinstance(y, dict):
        for k in ["logits", "output", "pred", "y"]:
            if k in y:
                y = y[k]
                break

    if y.ndim == 1:
        return torch.sigmoid(y).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 1:
        return torch.sigmoid(y[:, 0]).detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] == 2:
        p = torch.softmax(y, dim=1)[:, 1]
        return p.detach().cpu().numpy()

    yy = y.reshape(y.shape[0], -1)[:, 0]
    return torch.sigmoid(yy).detach().cpu().numpy()


# ------------------ viz utils ------------------
def resize_for_display(img_bgr: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img_bgr, 1.0
    s = max_dim / float(m)
    out = cv2.resize(img_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


def overlay_text_block(img_bgr: np.ndarray, lines: List[str], font_scale: float = 0.65) -> np.ndarray:
    out = img_bgr.copy()
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2

    text_lines = lines + [
        "Keys: SPACE/n next | b back | j/k img | 1/2/3 step | r recompute | click box => patch | q quit"
    ]
    sizes = [cv2.getTextSize(t, font, font_scale, thick)[0] for t in text_lines]
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = (len(text_lines) * (sizes[0][1] + 10)) + 2 * pad

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    y = pad + sizes[0][1]
    for t in text_lines:
        cv2.putText(out, t, (pad, y), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += sizes[0][1] + 10
    return out


def compute_boxes(centers: List[Dict], patch_size: int, img_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    h, w = img_shape[:2]
    half = patch_size // 2
    boxes = []
    for c in centers:
        cx, cy = int(c["cx"]), int(c["cy"])
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + patch_size, y0 + patch_size
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        boxes.append((x0, y0, x1, y1))
    return boxes


def draw_boxes_plain(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_bgr.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)  # yellow
    return out


def draw_boxes_pred(
    img_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    probs: List[float],
    thr: float,
) -> np.ndarray:
    out = img_bgr.copy()
    for (x0, y0, x1, y1), p in zip(boxes, probs):
        ok = float(p) >= float(thr)
        color = (0, 255, 0) if ok else (0, 0, 255)  # green/red (BGR)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out, f"{float(p):.2f}",
            (x0 + 4, max(18, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
    return out


def find_box_at_point(pt: Tuple[int, int], boxes: List[Tuple[int, int, int, int]]) -> Optional[int]:
    x, y = pt
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return i
    return None


def discover_models(runs_dir: Path) -> List[Dict]:
    """
    Returns list of {path: Path, label: str}.
    We prioritize best.pt and also include last.pt if present.
    """
    items: List[Dict] = []
    if not runs_dir.exists():
        return items

    bests = sorted(runs_dir.rglob("best.pt"))
    for p in bests:
        run_name = p.parent.name
        label = f"{run_name}/best.pt"
        items.append({"path": p, "label": label})

        last = p.parent / "last.pt"
        if last.exists():
            items.append({"path": last, "label": f"{run_name}/last.pt"})

    # also add torchscript if user stored them
    for p in sorted(runs_dir.rglob("*.ts")):
        items.append({"path": p, "label": f"{p.parent.name}/{p.name}"})

    # keep stable order
    return items


# ------------------ GUI ------------------
GUI_INSTANCE = None  # used by OpenCV trackbar callbacks


class PredictorGUI:
    def __init__(self, args, roi_cfg: RoiC2FConfig, device):
        assert torch is not None

        self.args = args
        self.roi_cfg = roi_cfg
        self.device = device

        self.img_paths = list(iter_image_files(Path(args.input)))
        if not self.img_paths:
            raise SystemExit(f"No images found in: {args.input}")

        runs_dir = Path(args.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = REPO_ROOT / runs_dir

        self.models = discover_models(runs_dir)

        # optionally add specific model path (even if outside runs_dir)
        if args.model:
            mp = Path(args.model)
            if not mp.is_absolute():
                mp = REPO_ROOT / mp
            if mp.exists():
                self.models.insert(0, {"path": mp, "label": f"manual/{mp.name}"})

        if not self.models:
            raise SystemExit(f"No models found. Check --runs-dir {runs_dir} or pass --model <path>.")

        self.model_index = 0
        self.threshold = float(args.threshold)
        self.threshold = float(np.clip(self.threshold, 0.0, 1.0))

        self.idx = 0
        self.step = int(args.start_step)

        # caches
        self.roi_cache: Dict[int, Dict] = {}  # per image: img, centers, boxes
        self.pred_cache: Dict[Tuple[int, int], List[float]] = {}  # (model_index, img_index) -> probs

        # model LRU cache (load/unload)
        self.loaded_models: "OrderedDict[int, Dict]" = OrderedDict()  # model_index -> {"model":..., "info":...}
        self.max_loaded_models = max(1, int(args.max_loaded_models))

        self.window = "ROI->CNN GUI"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)

        # trackbars
        cv2.createTrackbar("model", self.window, 0, max(0, len(self.models) - 1), PredictorGUI._on_model_change)
        cv2.createTrackbar("thr(%)", self.window, int(round(self.threshold * 100)), 100, PredictorGUI._on_thr_change)

        self.last_display_scale = 1.0
        self.last_boxes: List[Tuple[int, int, int, int]] = []
        self.last_probs: List[float] = []

        # initial model load
        self.set_model(0)

    # --- trackbar callbacks ---
    @staticmethod
    def _on_model_change(pos: int):
        if GUI_INSTANCE is None:
            return
        GUI_INSTANCE.set_model(int(pos))

    @staticmethod
    def _on_thr_change(pos: int):
        if GUI_INSTANCE is None:
            return
        GUI_INSTANCE.threshold = float(np.clip(pos / 100.0, 0.0, 1.0))

    # --- model loading ---
    def set_model(self, model_index: int):
        model_index = int(np.clip(model_index, 0, len(self.models) - 1))
        self.model_index = model_index

        # keep trackbar synced (if changed via keyboard in future)
        try:
            cv2.setTrackbarPos("model", self.window, self.model_index)
        except Exception:
            pass

        if model_index in self.loaded_models:
            # refresh LRU order
            self.loaded_models.move_to_end(model_index)
            return

        # load model
        mpath = self.models[model_index]["path"]
        model, info = load_model_any(mpath, self.device)

        self.loaded_models[model_index] = {"model": model, "info": info}
        self.loaded_models.move_to_end(model_index)

        # enforce LRU limit
        while len(self.loaded_models) > self.max_loaded_models:
            old_idx, _ = self.loaded_models.popitem(last=False)
            # keep preds cache; only unload model weights

    def get_model(self):
        self.set_model(self.model_index)
        return self.loaded_models[self.model_index]["model"], self.loaded_models[self.model_index]["info"]

    # --- ROI / preds ---
    def ensure_roi(self, img_index: int):
        if img_index in self.roi_cache:
            return

        img_path = self.img_paths[img_index]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            self.roi_cache[img_index] = {"img": None, "centers": [], "boxes": []}
            return

        centers = propose_centers_c2f(img, self.roi_cfg)
        max_patches = self.args.max_patches if self.args.max_patches and self.args.max_patches > 0 else int(self.roi_cfg.top_k_final)
        centers = centers[:max_patches]

        boxes = compute_boxes(centers, int(self.roi_cfg.patch_size), img.shape)
        self.roi_cache[img_index] = {"img": img, "centers": centers, "boxes": boxes}

    def ensure_preds(self, img_index: int, model_index: int, force: bool = False):
        key = (model_index, img_index)
        if (not force) and key in self.pred_cache:
            return

        self.ensure_roi(img_index)
        st = self.roi_cache[img_index]
        img = st["img"]
        centers = st["centers"]
        if img is None or not centers:
            self.pred_cache[key] = []
            return

        model, _info = self.get_model()

        patch_size = int(self.roi_cfg.patch_size)
        patches: List[np.ndarray] = []
        for c in centers:
            cx, cy = int(c["cx"]), int(c["cy"])
            patches.append(pad_and_crop(img, cx, cy, patch_size))

        bs = max(1, int(self.args.batch_size))
        probs_all: List[float] = []

        # inference
        with torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad():
            for start in range(0, len(patches), bs):
                batch = patches[start:start + bs]
                x = patches_to_tensor(batch, self.args.mean, self.args.std, self.device)
                y = model(x)
                p = probs_from_model_output(y).tolist()
                probs_all.extend([float(v) for v in p])

        self.pred_cache[key] = probs_all

    def recompute_current(self):
        # clear caches for current image+model and recompute
        if self.idx in self.roi_cache:
            del self.roi_cache[self.idx]
        key = (self.model_index, self.idx)
        if key in self.pred_cache:
            del self.pred_cache[key]
        self.ensure_roi(self.idx)
        self.ensure_preds(self.idx, self.model_index, force=True)

    # --- rendering ---
    def render(self) -> np.ndarray:
        self.ensure_roi(self.idx)
        self.ensure_preds(self.idx, self.model_index)

        img_path = self.img_paths[self.idx]
        st = self.roi_cache[self.idx]
        img = st["img"]
        centers = st["centers"]
        boxes = st["boxes"]

        probs = self.pred_cache.get((self.model_index, self.idx), [])

        model_label = self.models[self.model_index]["label"]

        if img is None:
            vis = np.zeros((600, 1000, 3), dtype=np.uint8)
            vis = overlay_text_block(
                vis,
                [f"[{self.idx+1}/{len(self.img_paths)}] {img_path.name}",
                 f"model={model_label}",
                 "ERROR: could not read image."],
                font_scale=self.args.font_scale,
            )
            self.last_display_scale = 1.0
            self.last_boxes = []
            self.last_probs = []
            return vis

        thr = float(self.threshold)
        max_prob = max(probs) if probs else 0.0
        is_pos = (max_prob >= thr) if probs else False

        if self.step == 1:
            base = img
            step_txt = "Step 1: Original image"
        elif self.step == 2:
            base = draw_boxes_plain(img, boxes)
            step_txt = "Step 2: ROI boxes (yellow)"
        else:
            base = draw_boxes_pred(img, boxes, probs, thr)
            step_txt = "Step 3: CNN prediction per ROI (green>=thr, red<thr)"

        base_disp, s = resize_for_display(base, int(self.args.max_dim))
        self.last_display_scale = s
        self.last_boxes = boxes
        self.last_probs = probs

        header = [
            f"[{self.idx+1}/{len(self.img_paths)}] {img_path.name} | step={self.step}",
            f"model={model_label}",
            f"patches={len(centers)} | thr={thr:.2f} | max_prob={max_prob:.3f} | image_pred={'QR' if is_pos else 'NO_QR'}",
            step_txt,
        ]
        return overlay_text_block(base_disp, header, font_scale=self.args.font_scale)

    # --- patch preview ---
    def show_patch_preview(self, patch: np.ndarray, title: str):
        wname = "Patch preview"
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        vis, _ = resize_for_display(patch, 700)
        vis = overlay_text_block(vis, [title], font_scale=0.7)
        cv2.imshow(wname, vis)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.ensure_roi(self.idx)
        self.ensure_preds(self.idx, self.model_index)

        st = self.roi_cache[self.idx]
        img = st["img"]
        centers = st["centers"]
        boxes = st["boxes"]
        probs = self.pred_cache.get((self.model_index, self.idx), [])

        if img is None or not boxes:
            return

        s = float(self.last_display_scale) if self.last_display_scale > 0 else 1.0
        ox = int(round(x / s))
        oy = int(round(y / s))

        i = find_box_at_point((ox, oy), boxes)
        if i is None or i >= len(centers):
            return

        c = centers[i]
        cx, cy = int(c["cx"]), int(c["cy"])
        patch = pad_and_crop(img, cx, cy, int(self.roi_cfg.patch_size))
        p = float(probs[i]) if i < len(probs) else -1.0
        title = f"idx={i}  prob={p:.3f}  thr={float(self.threshold):.2f}  cx={cx} cy={cy}"
        self.show_patch_preview(patch, title)

    # --- navigation ---
    def next_image(self):
        self.idx = (self.idx + 1) % len(self.img_paths)

    def prev_image(self):
        self.idx = (self.idx - 1) % len(self.img_paths)

    def next_step(self):
        self.step = 1 if self.step >= 3 else (self.step + 1)

    def prev_step(self):
        self.step = 3 if self.step <= 1 else (self.step - 1)

    def run(self):
        while True:
            vis = self.render()
            cv2.imshow(self.window, vis)

            k = cv2.waitKey(20) & 0xFF

            if k in (ord("q"), 27):  # q or ESC
                break

            if k in (32, ord("n")):  # SPACE / n
                self.next_step()

            if k == ord("b"):
                self.prev_step()

            if k == ord("1"):
                self.step = 1
            if k == ord("2"):
                self.step = 2
            if k == ord("3"):
                self.step = 3

            if k in (ord("j"), 83):  # j or right arrow
                self.next_image()
            if k in (ord("k"), 81):  # k or left arrow
                self.prev_image()

            if k == ord("r"):
                self.recompute_current()

        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for this script. Please install torch.")

    roi_cfg_path = Path(args.roi_config)
    if not roi_cfg_path.is_absolute():
        roi_cfg_path = REPO_ROOT / roi_cfg_path
    cfg = load_roi_config(roi_cfg_path)

    device = pick_device(args.device)
    print("[INFO] device:", device)
    print("[INFO] ROI cfg:", {k: v for k, v in asdict(cfg).items() if k in ["patch_size", "top_k_final", "n_scales", "win_min", "win_max"]})

    global GUI_INSTANCE
    GUI_INSTANCE = PredictorGUI(args, cfg, device)

    # show discovered models once
    print(f"[INFO] discovered models: {len(GUI_INSTANCE.models)}")
    for i, it in enumerate(GUI_INSTANCE.models[:30]):
        print(f"  [{i:02d}] {it['label']}  ->  {it['path']}")
    if len(GUI_INSTANCE.models) > 30:
        print("  ...")

    GUI_INSTANCE.run()


if __name__ == "__main__":
    main()
