#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exportiert alle Validierungs-Patches aus patches_split_seedXX.json
in neu erstellte Ordner:
  <splits_dir>/validierung/qr
  <splits_dir>/validierung/no_qr

Standard-Annahmen:
- Script liegt in <repo_root>/scripts/
- Split-JSON liegt in <repo_root>/data/splits/patches_split_seed42.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def safe_copy(src: Path, dst: Path) -> Path:
    """Copy file to dst; if dst exists, append _1, _2, ..."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem, suffix = dst.stem, dst.suffix
    for i in range(1, 10_000):
        cand = dst.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand

    raise RuntimeError(f"Zu viele Namenskollisionen für: {dst.name}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]  # <repo_root>/scripts/<this.py>
    default_splits_dir = repo_root / "data" / "splits"
    default_split_json = default_splits_dir / "patches_split_seed42.json"
    default_out_dir = default_splits_dir / "validierung"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split_json",
        type=Path,
        default=default_split_json,
        help="Pfad zur patches_split_seedXX.json",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=default_out_dir,
        help="Zielordner (es werden out_dir/qr und out_dir/no_qr erstellt)",
    )
    ap.add_argument(
        "--repo_root",
        type=Path,
        default=None,
        help="Optional: Repo-Root überschreiben (sonst nimmt er repo_root aus JSON oder aus Script-Location).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Nur anzeigen, nichts kopieren.")
    args = ap.parse_args()

    split_json_path = args.split_json
    if not split_json_path.exists():
        print(f"[ERROR] Split-JSON nicht gefunden: {split_json_path}", file=sys.stderr)
        return 2

    with split_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # repo_root bestimmen: CLI > JSON > Script-Location
    repo_root_from_json = None
    if isinstance(data, dict) and "repo_root" in data:
        try:
            repo_root_from_json = Path(data["repo_root"])
        except Exception:
            repo_root_from_json = None

    repo_root_final = args.repo_root or repo_root_from_json or repo_root

    # Val-Items holen
    try:
        val_items = data["splits"]["val"]
    except Exception:
        print("[ERROR] Konnte data['splits']['val'] nicht finden. Prüfe JSON-Struktur.", file=sys.stderr)
        return 3

    # Klassen-Mapping (Fallback: y==1 => qr)
    classes = data.get("classes", ["no_qr", "qr"])
    class_from_y = {0: "no_qr", 1: "qr"}
    if isinstance(classes, list) and len(classes) >= 2:
        # Wenn classes wie erwartet ["no_qr","qr"] ist, passt das.
        # Falls anders sortiert, bleibt trotzdem y-basiert sicher.
        pass

    out_dir = args.out_dir
    out_qr = out_dir / "qr"
    out_no = out_dir / "no_qr"
    out_qr.mkdir(parents=True, exist_ok=True)
    out_no.mkdir(parents=True, exist_ok=True)

    copied = {"qr": 0, "no_qr": 0}
    missing = 0

    for item in val_items:
        if not isinstance(item, dict):
            continue
        rel_path = item.get("path")
        y = item.get("y")

        if rel_path is None or y is None:
            continue

        src = Path(rel_path)
        if not src.is_absolute():
            src = repo_root_final / src

        cls = class_from_y.get(int(y), "qr" if int(y) == 1 else "no_qr")
        dst_dir = out_qr if cls == "qr" else out_no
        dst = dst_dir / src.name

        if not src.exists():
            missing += 1
            print(f"[WARN] Datei fehlt: {src}", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"[DRY] {src} -> {dst}")
        else:
            safe_copy(src, dst)
        copied[cls] += 1

    print("\n=== Fertig ===")
    print(f"Split JSON: {split_json_path}")
    print(f"Repo root:  {repo_root_final}")
    print(f"Output:     {out_dir}")
    print(f"Kopiert qr:     {copied['qr']}")
    print(f"Kopiert no_qr:  {copied['no_qr']}")
    if missing:
        print(f"Fehlende Dateien: {missing} (siehe WARN in stderr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
