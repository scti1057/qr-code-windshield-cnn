from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]


"""
Train a CNN from scratch for QR/no_qr patch classification.

Example:
python3 scripts/train_cnn_from_scratch.py --config configs/cnn/baseline.yaml

Correct (base + overrides):
python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/exp_lr_3e-4.yaml
"""


# -------- config loading ----------
def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: pyyaml. Install via: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# -------- dict utils ----------
def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# -------- reproducibility ----------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------- dataset ----------
class PatchDataset:
    """
    Reads split JSON created by make_patch_splits.py.
    Applies OpenCV augmentations on the fly.
    Returns numpy arrays (CHW float32) + int label.
    """
    def __init__(self, items: List[Dict[str, Any]], img_size: int, augment_cfg: Dict[str, Any], is_train: bool):
        self.items = items
        self.img_size = int(img_size)
        self.aug = augment_cfg or {}
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.items)

    def _rand(self) -> float:
        return random.random()

    def _maybe_hflip(self, img: np.ndarray) -> np.ndarray:
        p = float(self.aug.get("hflip_p", 0.0))
        if self._rand() < p:
            img = cv2.flip(img, 1)
        return img

    def _rotate(self, img: np.ndarray, deg: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def _maybe_rotate(self, img: np.ndarray) -> np.ndarray:
        deg_max = float(self.aug.get("rotate_deg", 0.0))
        if deg_max <= 0:
            return img
        deg = random.uniform(-deg_max, deg_max)
        return self._rotate(img, deg)

    def _maybe_perspective(self, img: np.ndarray) -> np.ndarray:
        p = float(self.aug.get("perspective_p", 0.0))
        if self._rand() >= p:
            return img
        h, w = img.shape[:2]
        max_j = max(2.0, 0.06 * min(h, w))
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32([[random.uniform(-max_j, max_j), random.uniform(-max_j, max_j)] for _ in range(4)])
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def _maybe_blur(self, img: np.ndarray) -> np.ndarray:
        p = float(self.aug.get("blur_p", 0.0))
        if self._rand() >= p:
            return img
        k = random.choice([3, 5])
        if random.random() < 0.5:
            return cv2.GaussianBlur(img, (k, k), 0)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)

    def _maybe_bc(self, img: np.ndarray) -> np.ndarray:
        b = float(self.aug.get("brightness", 0.0))
        c = float(self.aug.get("contrast", 0.0))
        if b <= 0 and c <= 0:
            return img
        alpha = 1.0 + random.uniform(-c, c)
        beta = 255.0 * random.uniform(-b, b)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _maybe_jpeg(self, img: np.ndarray) -> np.ndarray:
        p = float(self.aug.get("jpeg_p", 0.0))
        if self._rand() >= p:
            return img
        q = random.randint(30, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        ok, enc = cv2.imencode(".jpg", img, encode_param)
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img

    def _augment(self, img: np.ndarray) -> np.ndarray:
        img = self._maybe_hflip(img)
        img = self._maybe_rotate(img)
        img = self._maybe_perspective(img)
        img = self._maybe_blur(img)
        img = self._maybe_bc(img)
        img = self._maybe_jpeg(img)
        return img

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = REPO_ROOT / Path(item["path"])
        y = int(item["y"])

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.is_train and bool(self.aug.get("enabled", True)):
            img = self._augment(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))  # CHW
        return x, y


# -------- model (scratch CNN) ----------
def make_activation(name: str) -> nn.Module:
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
    """
    Simple staged CNN from scratch:
      stage i: [ConvBlock]*blocks + Downsample(MaxPool)
      head: GlobalAvgPool -> Dropout -> Linear(1 logit)
    """
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
        return self.head(x)  # logits


# -------- losses ----------
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    # targets: (N,) in {0,1}
    targets = targets.float()
    prob = torch.sigmoid(logits.squeeze(1))
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-6))
    return loss.mean()


# -------- metrics ----------
def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def prf_from_conf(tn, fp, fn, tp) -> Tuple[float, float, float]:
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(prec), float(rec), float(f1)


# -------- plotting ----------
def plot_curves(metrics_csv: Path, out_png_stub: Path):
    import matplotlib.pyplot as plt

    rows = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    if not rows:
        return

    epoch = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    plt.figure()
    plt.plot(epoch, train_loss, label="train_loss")
    plt.plot(epoch, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_stub.with_name("curves_loss.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_stub.with_name("curves_acc.png"), dpi=160)
    plt.close()


def plot_confusion(tn, fp, fn, tp, out_png: Path, title: str):
    import matplotlib.pyplot as plt

    mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    plt.figure()
    plt.imshow(mat)
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    for (i, j), v in np.ndenumerate(mat):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["no_qr", "qr"])
    plt.yticks([0, 1], ["no_qr", "qr"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------- training helpers ----------
def batch_to_device(x, y, device):
    if isinstance(x, torch.Tensor):
        x_t = x.to(device=device, dtype=torch.float32)
    else:
        x_t = torch.from_numpy(np.asarray(x)).to(device=device, dtype=torch.float32)

    if isinstance(y, torch.Tensor):
        y_i = y.to(device=device, dtype=torch.int64).view(-1)
    else:
        y_i = torch.tensor(y, device=device, dtype=torch.int64).view(-1)

    return x_t, y_i


def train_one_epoch(model, loader, optimizer, loss_name: str, device, label_smoothing: float = 0.0):
    model.train()
    total_loss = 0.0
    n = 0

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x_t, y_i = batch_to_device(x, y, device)
        y_f = y_i.float().view(-1, 1)

        if label_smoothing > 0:
            y_f = y_f * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        logits = model(x_t)

        if loss_name == "bce_logits":
            loss = nn.BCEWithLogitsLoss()(logits, y_f)
        elif loss_name == "focal":
            loss = focal_loss_with_logits(logits, y_i)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x_t.shape[0]
        n += x_t.shape[0]

        with torch.no_grad():
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            pred = (prob >= 0.5).astype(np.int64)
            y_cpu = y_i.detach().cpu().numpy().astype(np.int64)
            y_true_all.append(y_cpu)
            y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-9))
    return total_loss / max(1, n), acc, (tn, fp, fn, tp)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_name: str, device):
    model.eval()
    total_loss = 0.0
    n = 0

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x_t, y_i = batch_to_device(x, y, device)
        y_f = y_i.float().view(-1, 1)

        logits = model(x_t)

        if loss_name == "bce_logits":
            loss = nn.BCEWithLogitsLoss()(logits, y_f)
        elif loss_name == "focal":
            loss = focal_loss_with_logits(logits, y_i)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        total_loss += float(loss.item()) * x_t.shape[0]
        n += x_t.shape[0]

        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        pred = (prob >= 0.5).astype(np.int64)
        y_cpu = y_i.detach().cpu().numpy().astype(np.int64)
        y_true_all.append(y_cpu)
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-9))
    return total_loss / max(1, n), acc, (tn, fp, fn, tp)


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(cfg_train: Dict[str, Any], optimizer):
    sch = (cfg_train.get("scheduler") or {}).copy()
    typ = (sch.get("type") or "none").lower()
    if typ == "none":
        return None
    if typ == "cosine":
        epochs = int(cfg_train.get("epochs", 50))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if typ == "step":
        step_size = int(sch.get("step_size", 15))
        gamma = float(sch.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {typ}")


def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="Train a scratch CNN for QR/no_qr patch classification (config-driven).")
    ap.add_argument("--base", type=str, required=False, help="Base YAML/JSON config (e.g. baseline.yaml).")
    ap.add_argument("--override", type=str, action="append", default=[],
                    help="Override YAML/JSON config(s). Can be used multiple times.")
    ap.add_argument("--config", type=str, required=False,
                    help="DEPRECATED: single config (same as --base).")
    args = ap.parse_args()

    # Determine base config (support old --config)
    base_path_str = args.base or args.config
    if not base_path_str:
        raise SystemExit("Provide --base <path> (or legacy --config <path>).")

    base_path = Path(base_path_str)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    cfg = load_config(base_path)

    override_paths: List[Path] = []
    for ov_str in (args.override or []):
        ov = Path(ov_str)
        if not ov.is_absolute():
            ov = REPO_ROOT / ov
        cfg = deep_update(cfg, load_config(ov))
        override_paths.append(ov)

    config_sources = {"base": str(base_path), "overrides": [str(p) for p in override_paths]}

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    # device
    cfg_train = (cfg.get("train") or {})
    device_str = str(cfg_train.get("device", "cuda")).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

    # Data config
    cfg_data = (cfg.get("data") or {})
    split_cache = Path(cfg_data.get("split_cache", "data/splits/patches_split_seed42.json"))
    if not split_cache.is_absolute():
        split_cache = REPO_ROOT / split_cache
    if not split_cache.exists():
        raise SystemExit(
            f"Split cache not found: {split_cache}\n"
            "Create it first with: python3 scripts/make_patch_splits.py ..."
        )

    split_json = json.loads(split_cache.read_text(encoding="utf-8"))
    splits = split_json["splits"]
    img_size = int(cfg_data.get("img_size", 265))

    aug_cfg = cfg.get("augment") or {}

    train_ds = PatchDataset(splits["train"], img_size=img_size, augment_cfg=aug_cfg, is_train=True)
    val_ds = PatchDataset(splits["val"], img_size=img_size, augment_cfg=aug_cfg, is_train=False)
    test_ds = PatchDataset(splits["test"], img_size=img_size, augment_cfg=aug_cfg, is_train=False)

    bs = int(cfg_train.get("batch_size", 64))
    num_workers = int(cfg_train.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                             pin_memory=(device.type == "cuda"))

    # Model config
    cfg_model = (cfg.get("model") or {})
    model = TinyQRNet(
        in_channels=int(cfg_model.get("in_channels", 3)),
        base_channels=int(cfg_model.get("base_channels", 32)),
        blocks_per_stage=list(cfg_model.get("blocks_per_stage", [2, 2, 2, 2])),
        kernel_size=int(cfg_model.get("kernel_size", 3)),
        activation=str(cfg_model.get("activation", "relu")),
        batch_norm=bool(cfg_model.get("batch_norm", True)),
        dropout=float(cfg_model.get("dropout", 0.2)),
        num_outputs=int(cfg_model.get("num_outputs", 1)),
        global_pool=str(cfg_model.get("global_pool", "avg")),
    ).to(device)

    # Optimizer / scheduler / loss
    lr = float(cfg_train.get("lr", 1e-3))
    wd = float(cfg_train.get("weight_decay", 1e-4))
    opt_name = str(cfg_train.get("optimizer", "adam"))
    optimizer = make_optimizer(opt_name, model.parameters(), lr=lr, weight_decay=wd)
    scheduler = make_scheduler(cfg_train, optimizer)

    loss_name = str(cfg_train.get("loss", "bce_logits")).lower()
    label_smoothing = float(cfg_train.get("label_smoothing", 0.0))

    # Run dir + logging
    cfg_log = (cfg.get("logging") or {})
    runs_dir = Path(cfg_log.get("runs_dir", "runs/cnn_scratch"))
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg_log.get("run_name", "run"))
    run_dir = runs_dir / f"{now_tag()}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # store merged config + sources + split
    try:
        import yaml  # type: ignore
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        (run_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (run_dir / "config_sources.json").write_text(json.dumps(config_sources, indent=2), encoding="utf-8")
    shutil.copy2(split_cache, run_dir / "split_used.json")

    # notes
    (run_dir / "notes.md").write_text(
        "\n".join([
            f"# Run: {run_name}",
            "",
            "Key hyperparams:",
            f"- epochs={cfg_train.get('epochs')}, batch_size={bs}, lr={lr}, optimizer={opt_name}, loss={loss_name}",
            f"- activation={cfg_model.get('activation')}, base_channels={cfg_model.get('base_channels')}, blocks={cfg_model.get('blocks_per_stage')}",
            f"- early_stopping={cfg_train.get('early_stopping', {})}",
            f"- device={device.type}",
            "",
            "Config sources:",
            json.dumps(config_sources, indent=2),
            "",
        ]),
        encoding="utf-8"
    )

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "epoch", "lr",
            "train_loss", "train_acc",
            "val_loss", "val_acc",
            "val_prec", "val_rec", "val_f1",
        ])

    # Early stopping
    es_cfg = (cfg_train.get("early_stopping") or {})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_monitor = str(es_cfg.get("monitor", "val_loss")).lower()
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))

    best_metric = math.inf if es_monitor == "val_loss" else -math.inf
    best_epoch = -1
    bad_epochs = 0

    save_best = bool(cfg_log.get("save_best", True))
    save_last = bool(cfg_log.get("save_last", True))
    save_every = int(cfg_log.get("save_every_epochs", 0))

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    epochs = int(cfg_train.get("epochs", 50))

    for epoch in range(1, epochs + 1):
        cur_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, optimizer, loss_name, device, label_smoothing=label_smoothing
        )
        val_loss, val_acc, (tn, fp, fn, tp) = eval_one_epoch(model, val_loader, loss_name, device)
        val_prec, val_rec, val_f1 = prf_from_conf(tn, fp, fn, tp)

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([epoch, f"{cur_lr:.8f}", f"{train_loss:.6f}", f"{train_acc:.6f}",
                         f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_prec:.6f}", f"{val_rec:.6f}", f"{val_f1:.6f}"])

        if scheduler is not None:
            scheduler.step()

        current = val_loss if es_monitor == "val_loss" else val_acc
        improved = (current < (best_metric - es_min_delta)) if es_monitor == "val_loss" else (current > (best_metric + es_min_delta))

        if improved:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            if save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "split_path": str(split_cache),
                        "best_monitor": es_monitor,
                        "best_metric": float(best_metric),
                        "device": device.type,
                        "config_sources": config_sources,
                    },
                    best_path,
                )
        else:
            bad_epochs += 1

        if save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "split_path": str(split_cache),
                    "device": device.type,
                    "config_sources": config_sources,
                },
                last_path,
            )

        if save_every and (epoch % save_every == 0):
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, run_dir / f"epoch_{epoch:03d}.pt")

        print(
            f"[Epoch {epoch:03d}] lr={cur_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f} "
            f"(best {es_monitor}={best_metric:.4f} @ {best_epoch})"
        )

        if es_enabled and bad_epochs >= es_patience:
            print(f"[EarlyStopping] stop at epoch {epoch} (no improvement for {es_patience} epochs).")
            break

    # plots + confusion matrices
    plot_curves(metrics_csv, run_dir / "curves.png")

    val_loss, val_acc, (tn, fp, fn, tp) = eval_one_epoch(model, val_loader, loss_name, device)
    plot_confusion(tn, fp, fn, tp, run_dir / "confusion_val.png", "Confusion (val)")

    test_loss, test_acc, (tn, fp, fn, tp) = eval_one_epoch(model, test_loader, loss_name, device)
    plot_confusion(tn, fp, fn, tp, run_dir / "confusion_test.png", "Confusion (test)")

    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_monitor": es_monitor,
                "best_metric": float(best_metric),
                "final_val": {"loss": float(val_loss), "acc": float(val_acc)},
                "final_test": {"loss": float(test_loss), "acc": float(test_acc)},
                "device": device.type,
                "config_sources": config_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone.")
    print("Run dir:", run_dir)
    print("Best checkpoint:", best_path if best_path.exists() else "(not saved)")
    print("Last checkpoint:", last_path if last_path.exists() else "(not saved)")


if __name__ == "__main__":
    main()
