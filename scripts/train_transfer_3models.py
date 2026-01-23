from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


"""
Train 3 transfer-learning models (ImageNet pretrained) for QR/no_qr patch classification.

Trains sequentially:
  - resnet18
  - efficientnet_b0
  - mobilenet_v3_large

Saves each run under runs/cnn_scratch/<timestamp>_transfer_<modelname>/ with:
  - metrics.csv
  - curves_loss.png / curves_acc.png
  - confusion_val.png / confusion_test.png
  - best.pt / last.pt
  - final_metrics.json
  - config_used.yaml (merged)
  - config_sources.json

Example usage:

# Use a base config + optional overrides (recommended):
python3 scripts/train_transfer_3models.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/transfer_defaults.yaml

# Explicit model list:
python3 scripts/train_transfer_3models.py \
  --base configs/cnn/baseline.yaml \
  --models resnet18 efficientnet_b0 mobilenet_v3_large

Notes:
- For transfer learning, you SHOULD normalize like ImageNet:
  mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Set data.img_size to 224 (common) or keep 265 if you prefer.
"""


# ---------------- config loading ----------------
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


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------------- reproducibility ----------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- dataset ----------------
class PatchDataset(Dataset):
    """
    Reads split JSON created by make_patch_splits.py.
    Applies OpenCV augmentations on the fly.
    Returns torch tensors (CHW float32) + int label.

    Normalization is configurable; for pretrained ImageNet weights use:
      mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
    """
    def __init__(
        self,
        items: List[Dict[str, Any]],
        img_size: int,
        augment_cfg: Dict[str, Any],
        is_train: bool,
        mean: List[float],
        std: List[float],
    ):
        self.items = items
        self.img_size = int(img_size)
        self.aug = augment_cfg or {}
        self.is_train = bool(is_train)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

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

        # BGR -> RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # normalize (HWC)
        rgb = (rgb - self.mean) / (self.std + 1e-12)

        # to torch CHW
        x = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
        return x, torch.tensor(y, dtype=torch.int64)


# ---------------- metrics ----------------
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


# ---------------- plotting ----------------
def plot_curves(metrics_csv: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    rows = []
    if not metrics_csv.exists():
        return
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
    plt.savefig(out_dir / "curves_loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epoch, train_acc, label="train_acc")
    plt.plot(epoch, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "curves_acc.png", dpi=160)
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


# ---------------- losses ----------------
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    targets = targets.float()
    prob = torch.sigmoid(logits.squeeze(1))
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * (1 - p_t).pow(gamma) * torch.log(p_t.clamp(min=1e-6))
    return loss.mean()


# ---------------- optim / sched ----------------
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


# ---------------- torchvision model build ----------------
def build_torchvision_model(
    name: str,
    pretrained: bool,
    dropout: float,
    num_outputs: int = 1,
) -> nn.Module:
    try:
        import torchvision.models as M
    except Exception as e:
        raise RuntimeError("torchvision is required for transfer learning. Install torchvision.") from e

    n = name.lower().strip()
    weights = "DEFAULT" if pretrained else None

    if n == "resnet18":
        model = M.resnet18(weights=weights)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_f, int(num_outputs)),
        )
        return model

    if n == "efficientnet_b0":
        model = M.efficientnet_b0(weights=weights)
        # classifier is Sequential(Dropout, Linear)
        if isinstance(model.classifier, nn.Sequential):
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, int(num_outputs))
            # try to set dropout
            for m in model.classifier:
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)
        else:
            # fallback
            in_f = getattr(model.classifier, "in_features", 1280)
            model.classifier = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_f, int(num_outputs)))
        return model

    if n == "mobilenet_v3_large":
        model = M.mobilenet_v3_large(weights=weights)
        # classifier: Sequential(Linear, Hardswish, Dropout, Linear)
        if isinstance(model.classifier, nn.Sequential):
            # last layer is Linear
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, int(num_outputs))
            for m in model.classifier:
                if isinstance(m, nn.Dropout):
                    m.p = float(dropout)
        else:
            in_f = getattr(model.classifier, "in_features", 1280)
            model.classifier = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_f, int(num_outputs)))
        return model

    raise ValueError(f"Unknown transfer model: {name}. Supported: resnet18, efficientnet_b0, mobilenet_v3_large")


def set_backbone_trainable(model: nn.Module, trainable: bool, model_name: str):
    """
    Freeze/unfreeze everything except classification head.
    """
    n = model_name.lower().strip()

    # first freeze all
    for p in model.parameters():
        p.requires_grad = trainable

    # then ensure head is trainable
    if n == "resnet18":
        for p in model.fc.parameters():
            p.requires_grad = True
        return
    if n in {"efficientnet_b0", "mobilenet_v3_large"}:
        for p in model.classifier.parameters():
            p.requires_grad = True
        return


def get_head_params(model: nn.Module, model_name: str):
    n = model_name.lower().strip()
    if n == "resnet18":
        return model.fc.parameters()
    return model.classifier.parameters()


def get_backbone_params(model: nn.Module, model_name: str):
    n = model_name.lower().strip()
    head = set(get_head_params(model, model_name))
    for p in model.parameters():
        if p not in head:
            yield p


# ---------------- training loops ----------------
def train_one_epoch(model, loader, optimizer, loss_name: str, device, label_smoothing: float = 0.0):
    model.train()
    total_loss = 0.0
    n = 0

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.int64).view(-1)
        y_f = y.float().view(-1, 1)

        if label_smoothing > 0:
            y_f = y_f * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()
        logits = model(x)

        if loss_name == "bce_logits":
            loss = nn.BCEWithLogitsLoss()(logits, y_f)
        elif loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.shape[0]
        n += x.shape[0]

        with torch.no_grad():
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            pred = (prob >= 0.5).astype(np.int64)
            y_true_all.append(y.detach().cpu().numpy().astype(np.int64))
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
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.int64).view(-1)
        y_f = y.float().view(-1, 1)

        logits = model(x)

        if loss_name == "bce_logits":
            loss = nn.BCEWithLogitsLoss()(logits, y_f)
        elif loss_name == "focal":
            loss = focal_loss_with_logits(logits, y)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        total_loss += float(loss.item()) * x.shape[0]
        n += x.shape[0]

        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        pred = (prob >= 0.5).astype(np.int64)

        y_true_all.append(y.detach().cpu().numpy().astype(np.int64))
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.zeros((0,), dtype=np.int64)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-9))
    return total_loss / max(1, n), acc, (tn, fp, fn, tp)


# ---------------- utils ----------------
def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def resolve_repo_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def pick_device(cfg_train: Dict[str, Any]) -> torch.device:
    device_str = str(cfg_train.get("device", "cuda")).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    return torch.device(device_str)


def run_one_model(
    model_name: str,
    cfg: Dict[str, Any],
    base_path: Path,
    override_paths: List[Path],
) -> Path:
    # apply optional per-model overrides
    cfg_local = deepcopy(cfg)
    per_model = (((cfg_local.get("transfer") or {}).get("model_overrides")) or {}).get(model_name, {})
    if isinstance(per_model, dict) and per_model:
        cfg_local = deep_update(cfg_local, deepcopy(per_model))

    seed = int(cfg_local.get("seed", 42))
    seed_everything(seed)

    cfg_train = (cfg_local.get("train") or {})
    device = pick_device(cfg_train)

    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

    # data
    cfg_data = (cfg_local.get("data") or {})
    split_cache = resolve_repo_path(str(cfg_data.get("split_cache", "data/splits/patches_split_seed42.json")))
    if not split_cache.exists():
        raise SystemExit(
            f"Split cache not found: {split_cache}\n"
            "Create it first with: python3 scripts/make_patch_splits.py ..."
        )
    split_json = json.loads(split_cache.read_text(encoding="utf-8"))
    splits = split_json["splits"]

    img_size = int(cfg_data.get("img_size", 224))

    # augment + normalization
    aug_cfg = cfg_local.get("augment") or {}
    cfg_pp = cfg_local.get("preprocess") or {}
    mean = list(cfg_pp.get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_pp.get("std", [0.229, 0.224, 0.225]))

    train_ds = PatchDataset(splits["train"], img_size=img_size, augment_cfg=aug_cfg, is_train=True, mean=mean, std=std)
    val_ds = PatchDataset(splits["val"], img_size=img_size, augment_cfg=aug_cfg, is_train=False, mean=mean, std=std)
    test_ds = PatchDataset(splits["test"], img_size=img_size, augment_cfg=aug_cfg, is_train=False, mean=mean, std=std)

    bs = int(cfg_train.get("batch_size", 64))
    num_workers = int(cfg_train.get("num_workers", 0))
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # transfer config
    cfg_t = (cfg_local.get("transfer") or {})
    pretrained = bool(cfg_t.get("pretrained", True))
    freeze_backbone = bool(cfg_t.get("freeze_backbone", True))
    warmup_epochs = int(cfg_t.get("warmup_epochs", 2))
    backbone_lr_mult = float(cfg_t.get("backbone_lr_mult", 0.1))

    # head config
    cfg_head = (cfg_local.get("model_head") or {})
    dropout = float(cfg_head.get("dropout", 0.2))
    num_outputs = int(cfg_head.get("num_outputs", 1))

    model = build_torchvision_model(model_name, pretrained=pretrained, dropout=dropout, num_outputs=num_outputs).to(device)

    # freeze backbone for warmup (optional)
    if freeze_backbone:
        set_backbone_trainable(model, trainable=False, model_name=model_name)

    # optimizer / scheduler / loss
    lr = float(cfg_train.get("lr", 1e-4))
    wd = float(cfg_train.get("weight_decay", 1e-4))
    opt_name = str(cfg_train.get("optimizer", "adamw"))

    # param groups: head lr = lr, backbone lr = lr*mult (if backbone trainable)
    head_params = list(get_head_params(model, model_name))
    back_params = list(get_backbone_params(model, model_name))

    param_groups = [{"params": head_params, "lr": lr, "weight_decay": wd}]
    if (not freeze_backbone) or warmup_epochs == 0:
        if back_params:
            param_groups.append({"params": back_params, "lr": lr * backbone_lr_mult, "weight_decay": wd})

    optimizer = make_optimizer(opt_name, param_groups, lr=lr, weight_decay=wd)
    scheduler = make_scheduler(cfg_train, optimizer)

    loss_name = str(cfg_train.get("loss", "bce_logits")).lower()
    label_smoothing = float(cfg_train.get("label_smoothing", 0.0))

    # logging
    cfg_log = (cfg_local.get("logging") or {})
    runs_dir = resolve_repo_path(str(cfg_log.get("runs_dir", "runs/cnn_scratch")))
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg_log.get("run_name", "transfer"))
    run_dir = runs_dir / f"{now_tag()}_{run_name}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # store config used
    try:
        import yaml  # type: ignore
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg_local, sort_keys=False), encoding="utf-8")
    except Exception:
        (run_dir / "config_used.json").write_text(json.dumps(cfg_local, indent=2), encoding="utf-8")

    (run_dir / "config_sources.json").write_text(
        json.dumps(
            {"base": str(base_path), "overrides": [str(p) for p in override_paths], "model_name": model_name},
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(split_cache, run_dir / "split_used.json")

    (run_dir / "notes.md").write_text(
        "\n".join([
            f"# Transfer Run: {run_name}_{model_name}",
            "",
            "Key settings:",
            f"- model={model_name}, pretrained={pretrained}",
            f"- img_size={img_size}, mean={mean}, std={std}",
            f"- epochs={cfg_train.get('epochs')}, batch_size={bs}, lr={lr}, optimizer={opt_name}, loss={loss_name}",
            f"- freeze_backbone={freeze_backbone}, warmup_epochs={warmup_epochs}, backbone_lr_mult={backbone_lr_mult}",
            f"- early_stopping={cfg_train.get('early_stopping', {})}",
            f"- device={device.type}",
            "",
        ]),
        encoding="utf-8",
    )

    metrics_csv = run_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "epoch", "lr_head", "lr_backbone",
            "train_loss", "train_acc",
            "val_loss", "val_acc",
            "val_prec", "val_rec", "val_f1",
        ])

    # early stopping
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

    epochs = int(cfg_train.get("epochs", 30))

    for epoch in range(1, epochs + 1):
        # unfreeze after warmup
        if freeze_backbone and warmup_epochs > 0 and epoch == warmup_epochs + 1:
            set_backbone_trainable(model, trainable=True, model_name=model_name)
            # add backbone group if not present yet
            if len(optimizer.param_groups) == 1 and back_params:
                optimizer.add_param_group({"params": back_params, "lr": lr * backbone_lr_mult, "weight_decay": wd})

        lr_head = float(optimizer.param_groups[0]["lr"])
        lr_back = float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else 0.0

        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, optimizer, loss_name, device, label_smoothing=label_smoothing
        )
        val_loss, val_acc, (tn, fp, fn, tp) = eval_one_epoch(model, val_loader, loss_name, device)
        val_prec, val_rec, val_f1 = prf_from_conf(tn, fp, fn, tp)

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([
                epoch, f"{lr_head:.8f}", f"{lr_back:.8f}",
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
                f"{val_prec:.6f}", f"{val_rec:.6f}", f"{val_f1:.6f}",
            ])

        if scheduler is not None:
            scheduler.step()

        current = val_loss if es_monitor == "val_loss" else val_acc
        improved = (current < (best_metric - es_min_delta)) if es_monitor == "val_loss" else (current > (best_metric + es_min_delta))

        if improved:
            best_metric = float(current)
            best_epoch = epoch
            bad_epochs = 0
            if save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": model_name,
                        "pretrained": pretrained,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "split_path": str(split_cache),
                        "best_monitor": es_monitor,
                        "best_metric": float(best_metric),
                        "device": device.type,
                    },
                    best_path,
                )
        else:
            bad_epochs += 1

        if save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "split_path": str(split_cache),
                    "device": device.type,
                },
                last_path,
            )

        if save_every and (epoch % save_every == 0):
            torch.save(
                {"epoch": epoch, "model_name": model_name, "model_state": model.state_dict()},
                run_dir / f"epoch_{epoch:03d}.pt",
            )

        print(
            f"[{model_name}][Epoch {epoch:03d}] "
            f"lr_head={lr_head:.2e} lr_back={lr_back:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f} "
            f"(best {es_monitor}={best_metric:.4f} @ {best_epoch})"
        )

        if es_enabled and bad_epochs >= es_patience:
            print(f"[{model_name}][EarlyStopping] stop at epoch {epoch} (no improvement for {es_patience} epochs).")
            break

    # plots + confusion
    plot_curves(metrics_csv, run_dir)

    val_loss, val_acc, (tn, fp, fn, tp) = eval_one_epoch(model, val_loader, loss_name, device)
    plot_confusion(tn, fp, fn, tp, run_dir / "confusion_val.png", "Confusion (val)")

    test_loss, test_acc, (tn2, fp2, fn2, tp2) = eval_one_epoch(model, test_loader, loss_name, device)
    plot_confusion(tn2, fp2, fn2, tp2, run_dir / "confusion_test.png", "Confusion (test)")

    (run_dir / "final_metrics.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "best_epoch": best_epoch,
                "best_monitor": es_monitor,
                "best_metric": float(best_metric),
                "final_val": {"loss": float(val_loss), "acc": float(val_acc)},
                "final_test": {"loss": float(test_loss), "acc": float(test_acc)},
                "device": device.type,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[{model_name}] Done. Run dir: {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train 3 transfer-learning models sequentially (config-driven).")

    p.add_argument("--base", type=str, required=False, help="Base YAML/JSON config (e.g. baseline.yaml).")
    p.add_argument("--override", type=str, action="append", default=[],
                   help="Override YAML/JSON config(s). Can be used multiple times.")

    # Backward compatible:
    p.add_argument("--config", type=str, required=False, help="DEPRECATED: single config (same as --base).")

    p.add_argument("--models", nargs="*", default=[],
                   help="Optional explicit model list (overrides config.transfer.models). "
                        "Supported: resnet18 efficientnet_b0 mobilenet_v3_large")

    return p.parse_args()


def main():
    args = parse_args()

    base_path_str = args.base or args.config
    if not base_path_str:
        raise SystemExit("Provide --base <path> (or legacy --config <path>).")

    base_path = resolve_repo_path(base_path_str)
    cfg = load_config(base_path)

    override_paths: List[Path] = []
    for ov_str in (args.override or []):
        ov = resolve_repo_path(ov_str)
        cfg = deep_update(cfg, load_config(ov))
        override_paths.append(ov)

    # default model list
    cfg_t = (cfg.get("transfer") or {})
    models_cfg = cfg_t.get("models", ["resnet18", "efficientnet_b0", "mobilenet_v3_large"])
    models = args.models if args.models else list(models_cfg)

    # run sequentially
    for m in models:
        run_one_model(m, cfg, base_path, override_paths)

    print("\nAll transfer runs finished.")


if __name__ == "__main__":
    main()
