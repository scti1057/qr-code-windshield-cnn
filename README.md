# QR-Code Windshield CNN - Umfassende Projektdokumentation

## 📋 Projektüberblick

### Titel
**QR-Code-Erkennung in einer Waschstraße mittels Künstlicher Intelligenz**

### Autoren
- Paul Glaser, B. Eng (glpa1013)
- Tim Schäfer, B. Eng (scti1057)

### Institution
- **Universität**: Hochschule Karlsruhe
- **Fakultät**: Fakultät für Maschinenbau und Mechatronik
- **Betreuer**: Prof. Dr.-Ing. habil. Catherina Burghart
- **Zeitraum**: Wintersemester 25/26

---

## 🎯 Problemstellung und Motivation

### Anwendungsszenario
In modernen Waschanlagen soll die Programmauswahl und Abrechnung zunehmend automatisiert werden. Das System soll:
- QR-Codes erkennen, die vom Kunden auf einem Zettel hinter der Windschutzscheibe eines Fahrzeugs platziert werden
- Automatisch das gewählte Waschprogramm identifizieren
- Die Abrechnung automatisieren

### Herausforderungen
Das System muss unter schwierigen Bedingungen arbeiten:
- **Nasse, seifige oder verschmutzte Scheiben** - optische Verzerrungen und Verschmutzungen
- **Spiegelungen und Reflexionen** - durch Wasser und Lichteinstrahlung
- **Unterschiedliche Lichtverhältnisse** - von dunkel bis sehr hell
- **Störobjekte** - weitere Zettel oder Objekte ohne QR-Code hinter der Scheibe
- **Entfernung** - Erkennung aus bis zu 3 Metern Entfernung erforderlich
- **Kamerawinkel und Perspektive** - der QR-Code kann aus verschiedenen Winkeln sichtbar sein

### Zielstellung
Entwicklung eines **robusten Systems zur optischen Erkennung und Analyse von QR-Codes** in dem beschriebenen Szenario mit zwei Hauptbereichen:
1. **QR-Code Lokalisierung** - Finden und Lokalisieren des QR-Codes im Bild
2. **Lesbarkeitsprüfung** - Validierung und Analyse der Lesbarkeit des erkannten QR-Codes

---

## 🏗️ Systemarchitektur

### Pipeline-Überblick
```
Eingabebild
    ↓
[ROI-Vorverarbeitung] → Interessengebiet extrahieren
    ↓
[Patch-Extraktion] → 265×265 Pixel Patches generieren
    ↓
[CNN-Klassifizierung] → QR vs. No-QR klassifizieren
    ↓
[Lokalisierung] → Exakte Position bestimmen
    ↓
[Lesbarkeitsprüfung] → Scanner-Lesbarkeit validieren
    ↓
Erkannte QR-Code + Perspektivanalyse (rot_x, rot_y)
```

### Zwei Erkennungsmethoden

#### **Methode A: OpenCV QR-Code Detektor**
- Standard-OpenCV-Funktion zur direkten QR-Code-Erkennung
- **Vorteil**: Schnell und zuverlässig bei klaren Bildern
- **Nachteil**: Scheitert bei schlechter Bildqualität, Spiegelungen, Verschmutzung

#### **Methode B: Robuste Line-basierte Erkennung**
- Alternative für schwierige Bedingungen
- **Prozess**:
  1. Kantendetektion (Canny Edge Detection)
  2. Morphologische Operationen (Closing, Opening)
  3. Hough-Linien-Transformation zur Linienerkennung
  4. K-Means Clustering zur Liniengruppierung
  5. Extremale Linien auswählen
  6. Schnittpunkte berechnen (Quad-Ecken)
- **Vorteil**: Robuster gegen Bildverzerrungen
- **Nachteil**: Rechenintensiver, benötigt mehr Parameter-Tuning

---

## 🔧 Kernkomponenten

### 1. ROI (Region of Interest) - Autotuning System

**Dateien**: 
- `scripts/roi_tuner_gui.py` - Interaktive GUI zum Kalibrieren
- `src/qr_cnn/data/roi_c2f.py` - ROI-Detektionslogik
- `configs/roi_tuner_params.json` - ROI-Parameter
- `configs/roi_autotuned.json` - Optimierte Konfiguration

**Aufgabe**: 
- Automatische Detektion des Windscheibenbereichs
- Rotation-invariante Scoremap basierend auf "parallelen Kanten" (Anisotropie)
- Kandidatenselektion durch NMS (Non-Maximum Suppression)

**Funktionsweise**:
```
Original-Bild
    ↓
Gradient-Berechnung (Sobel)
    ↓
Anisotropie-Berechnung (dominante Richtung)
    ↓
Scoremap generieren (QR-typische Strukturen)
    ↓
Kandidaten mit NMS filtern
    ↓
ROI-Box bestimmen
```

**Verwendung**:
```bash
python3 scripts/roi_tuner_gui.py \
  --input data/raw \
  --load-config configs/roi_tuner_params.json
```

### 2. Patch-Extraktion und Datensatz

**Dateien**:
- `scripts/extract_rois_to_disk.py` - ROIs extrahieren
- `scripts/make_patch_splits.py` - Train/Val/Test Split erstellen
- `scripts/augment_flip_patches.py` - Datensatz vergrößern

**Datensatzstruktur**:
```
data/patches/all/
├── qr/          # QR-Code Patches (265×265)
│   ├── train/
│   ├── val/
│   └── test/
└── no_qr/       # Keine QR-Code Patches
    ├── train/
    ├── val/
    └── test/
```

**Patch-Eigenschaften**:
- **Größe**: 265×265 Pixel
- **Klassen**: 2 (qr, no_qr)
- **Split**: 80% Training, 10% Validierung, 10% Test
- **Seed**: 42 (reproduzierbar)

**Augmentierung**:
- Horizontal Flip (20% Wahrscheinlichkeit)
- Rotation (0-360°, wichtig für gedrehte QR-Codes)
- Perspektive Transformation (25% Wahrscheinlichkeit)
- Motion Blur (25% Wahrscheinlichkeit)
- Helligkeit-Anpassung (±20%)
- Kontrast-Anpassung (±20%)

### 3. CNN-Training-System

#### Training von Grund auf (Scratch)

**Datei**: `scripts/train_cnn_from_scratch.py`

**Verwendung**:
```bash
# Basis-Konfiguration
python3 scripts/train_cnn_from_scratch.py \
  --config configs/cnn/baseline.yaml

# Mit Überrides (Basis + Experiment)
python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/exp_lr_3e-4.yaml
```

**Verfügbare Experimente**:
- `baseline.yaml` - Basis-Konfiguration
- `exp_lr_3e-4.yaml` - Learning Rate 3e-4
- `exp_lr_3e-3.yaml` - Learning Rate 3e-3
- `exp_bs_32.yaml` - Batch Size 32
- `exp_act_leakyrelu.yaml` - LeakyReLU Aktivierung
- `exp_loss_focal.yaml` - Focal Loss
- `exp_opt_adamw.yaml` - AdamW Optimizer
- `exp_opt_sgd.yaml` - SGD Optimizer
- `exp_width_64.yaml` - Kanal-Breite 64
- `exp_deeper.yaml` - Tieferes Netzwerk

**Ausgaben eines Trainings**:
```
runs/cnn_scratch/<timestamp>_<exp_name>/
├── best.pt              # Beste Gewichte
├── last.pt              # Letzte Gewichte
├── config_used.yaml     # Verwendete Konfiguration
├── config_sources.json  # Konfigurations-Herkunft
├── final_metrics.json   # Finale Metriken
├── metrics.csv          # Trainings-Log
├── curves_loss.png      # Loss-Kurven
├── curves_acc.png       # Accuracy-Kurven
├── confusion_val.png    # Confusion Matrix Validierung
└── confusion_test.png   # Confusion Matrix Test
```

#### Transfer Learning

**Datei**: `scripts/train_transfer_3models.py`

**Trainiert sequenziell 3 ImageNet-vortrainierte Modelle**:
1. ResNet-18
2. EfficientNet-B0
3. MobileNet-V3-Large

**Verwendung**:
```bash
python3 scripts/train_transfer_3models.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/transfer_defaults.yaml
```

**Vorteil**:
- Transfer Learning nutzt ImageNet-Vortraining
- Schnelleres Training mit besserer Konvergenz
- Ideal für kleinere Datensätze

#### Modell-Architektur

**Baseline CNN** (TinyQRNet):
```
Input (3×265×265)
  ↓
Conv2d (3→32, 3×3)
  ↓ BatchNorm + ReLU + MaxPool (2×2)
Conv2d (32→64, 3×3)
  ↓ BatchNorm + ReLU + MaxPool (2×2)
Conv2d (64→128, 3×3)
  ↓ BatchNorm + ReLU + MaxPool (2×2)
GlobalAvgPool
  ↓
Linear (128→64)
  ↓ Dropout(0.5)
Linear (64→2)  [softmax]
  ↓
Output: [no_qr_prob, qr_prob]
```

### 4. Inferenz und GUI-Systeme

#### QR-Rektifizierungs-GUI

**Datei**: `scripts/qr_rectify_gui.py` (Hauptversion, mit Plan B Improvements)
**Alt**: `scripts/qr_rectify_gui_v1.py` (ältere Version)

**Funktionalität**:
- **Echtzeit-Bildanzeige** mit interaktiven Views
- **Dual-Methoden-Erkennung**:
  - Plan A: OpenCV QR-Detektor
  - Plan B: Line-basierte Erkennung (Fallback)
- **Perspektiv-Analyse**:
  - rot_x: Vertikale Perspektive (oben vs. unten)
  - rot_y: Horizontale Perspektive (links vs. rechts)
- **Warping zu Top-Down-Ansicht** (Birdseye, 420×420)
- **Interaktive Ansichten**:
  1. Vollständig erkanntes Bild
  2. Gewarpter QR-Code (Draufsicht)
  3. Cropped Patches + Quad-Overlay

**Verwendung**:
```bash
python3 scripts/qr_rectify_gui.py \
  --roi-config configs/roi_autotuned.json \
  --model-path models/best.pt \
  --batch-size 64 \
  --warp-size 420
```

**Tastatursteuerung**:
- `[Pfeile]`: Durch Bilder navigieren
- `[SPACE]`: Views durchschalten
- `[q/ESC]`: Beenden

#### ROI-Tuner GUI

**Datei**: `scripts/roi_tuner_gui.py`

**Funktionalität**:
- Visuelle Tuning der ROI-Detektionsparameter
- Live-Feedback für Parameter-Anpassungen
- Speichern optimierter Konfigurationen

**Parameter**:
- Morph Close Kernel Size / Iterationen
- Morph Open Kernel Size / Iterationen
- Min Area Fraction
- NMS IoU Threshold
- Top-K Kandidaten

### 5. Analyse und Vergleich

**Datei**: `scripts/compare_runs.py`

**Funktionalität**:
- Vergleich mehrerer Trainings-Läufe
- Metrik-Aggregation
- CSV-Zusammenfassung
- Markdown-Report

**Ausgabe**:
```
runs/cnn_scratch/_compare/
├── summary.csv   # Tabellendaten
└── summary.md    # Markdown-Report
```

---

## 📊 Konfigurationssystem

### Konfiguration Laden und Überschreiben

Alle YAML/JSON-Dateien werden hierarchisch geladen:

**Basis + Override-Muster**:
```python
config = load_config("configs/cnn/baseline.yaml")
override = load_config("configs/cnn/exp_lr_3e-4.yaml")
merged = deep_update(config, override)
```

### Struktur einer Baseline-Konfiguration

```yaml
seed: 42

data:
  root_all: "data/patches/all"
  classes: ["no_qr", "qr"]
  split:
    train: 0.80
    val: 0.10
    test: 0.10
  split_cache: "data/splits/patches_split_seed42.json"
  img_size: 265

augment:
  enabled: true
  hflip_p: 0.2
  rotate_deg: 360
  perspective_p: 0.25
  blur_p: 0.25
  brightness: 0.2
  contrast: 0.2

model:
  name: "tiny_qr_net"
  in_channels: 3
  initial_width: 32

train:
  epochs: 50
  batch_size: 64
  lr: 1.0e-3
  optimizer: "adam"
  loss: "cross_entropy"
  
logging:
  log_interval: 10
  save_interval: 5
```

---

## 🚀 Workflow und Anwendungsszenarien

### Szenario 1: Neues Projekt Starten

```bash
# 1. Rohbilder vorbereiten
# data/raw/ ← Waschanlage-Bilder

# 2. ROI-Parameter tunen
python3 scripts/roi_tuner_gui.py --input data/raw --output configs/roi_autotuned.json

# 3. Patches extrahieren
python3 scripts/extract_rois_to_disk.py \
  --roi-config configs/roi_autotuned.json \
  --input data/raw \
  --output data/patches/all

# 4. Train/Val/Test Split erstellen
python3 scripts/make_patch_splits.py \
  --input data/patches/all \
  --seed 42

# 5. Modell trainieren
python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml

# 6. Test-Inferenz durchführen
python3 scripts/qr_rectify_gui.py \
  --roi-config configs/roi_autotuned.json \
  --model-path runs/cnn_scratch/<latest>_baseline/best.pt
```

### Szenario 2: Hyperparameter-Experimente

```bash
# Multiple Experimente in Reihe
python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/exp_lr_3e-4.yaml

python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/exp_lr_3e-3.yaml

python3 scripts/train_cnn_from_scratch.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/exp_bs_32.yaml

# Alle Ergebnisse vergleichen
python3 scripts/compare_runs.py runs/cnn_scratch
```

### Szenario 3: Transfer Learning Pipeline

```bash
# Alle 3 vortrainierten Modelle sequenziell trainieren
python3 scripts/train_transfer_3models.py \
  --base configs/cnn/baseline.yaml \
  --override configs/cnn/transfer_defaults.yaml

# Ergebnisse in runs/cnn_scratch/<timestamp>_transfer_*/ verfügbar
```

### Szenario 4: Modell-Monitoring während Training

```bash
# Metriken live überwachen
python3 scripts/watch_metrics_keypress.py runs/cnn_scratch/<run_name>/
```

---

## 📁 Verzeichnisstruktur im Detail

```
qr-code-windshield-cnn/
├── README.md                          # Kurze Einführung
├── PROJECT_DOCUMENTATION.md           # Diese Datei
├── requirements.txt                   # Python-Abhängigkeiten
│
├── configs/
│   ├── roi_autotuned.json            # Optimierte ROI-Konfiguration
│   ├── roi_tuner_params.json         # ROI-Tuner Startparameter
│   └── cnn/
│       ├── baseline.yaml              # Basis-Konfiguration
│       ├── transfer_defaults.yaml     # Transfer Learning Standard
│       └── exp_*.yaml                 # Verschiedene Experimente
│
├── data/
│   ├── raw/                          # Eingabe: Rohbilder
│   ├── patches/
│   │   └── all/
│   │       ├── qr/
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── test/
│   │       └── no_qr/
│   │           ├── train/
│   │           ├── val/
│   │           └── test/
│   └── splits/
│       └── patches_split_seed42.json  # Train/Val/Test Zuordnung
│
├── models/
│   └── [trainierte Modelle außerhalb von runs/]
│
├── scripts/
│   ├── train_cnn_from_scratch.py     # ⭐ Haupttraining
│   ├── train_transfer_3models.py     # ⭐ Transfer Learning
│   ├── roi_tuner_gui.py               # ⭐ ROI-Tuning GUI
│   ├── qr_rectify_gui.py              # ⭐ Inferenz GUI (Version 2)
│   ├── qr_rectify_gui_v1.py           # Inferenz GUI (ältere Version)
│   ├── extract_rois_to_disk.py       # ROI Extraktion
│   ├── make_patch_splits.py          # Datensatz-Split
│   ├── augment_flip_patches.py       # Datensatz-Augmentierung
│   ├── compare_runs.py                # Trainings-Vergleich
│   ├── watch_metrics_keypress.py     # Live-Monitoring
│   ├── predict_folder_patches.py     # Batch-Inferenz
│   ├── predict_gui.py                 # Einfache Inferenz GUI
│   ├── run_sweep_sequential.py       # Parameter-Sweep
│   ├── overlapping_test.py           # Test-Utility
│   └── roi_autotune_click.py         # Automatisches ROI-Tuning
│
├── src/qr_cnn/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── roi_c2f.py                # ROI-Detektion
│   │   └── [andere Data-Module]
│   ├── models/
│   │   ├── __init__.py
│   │   └── [Model Definitions]
│   ├── train/
│   │   ├── __init__.py
│   │   └── [Trainings-Utilities]
│   └── infer/
│       ├── __init__.py
│       └── [Inferenz-Utilities]
│
├── runs/
│   ├── cnn_scratch/
│   │   ├── _compare/
│   │   │   ├── summary.csv           # Vergleichstabelle
│   │   │   └── summary.md            # Vergleichsbericht
│   │   ├── 2026-01-05_111844_lr_3e-4/
│   │   │   ├── best.pt               # Beste Gewichte
│   │   │   ├── last.pt               # Letzte Gewichte
│   │   │   ├── config_used.yaml      # Finale Konfiguration
│   │   │   ├── final_metrics.json    # Finale Metriken
│   │   │   ├── metrics.csv           # Trainings-Log
│   │   │   ├── curves_loss.png       # Loss-Kurven
│   │   │   ├── curves_acc.png        # Accuracy-Kurven
│   │   │   ├── confusion_val.png     # Val Confusion Matrix
│   │   │   └── confusion_test.png    # Test Confusion Matrix
│   │   └── [weitere Runs...]
│   └── sweeps/
│       └── [Parameter-Sweep Ergebnisse]
│
├── reports/
│   └── figures/                       # Bericht-Grafiken
│
└── docs/
    └── LaTeX/
        ├── thesis.tex                # Hauptdatei
        ├── bibliography.bib          # Referenzen
        ├── TeXFiles/
        │   ├── 000_Settings.tex      # Einstellungen
        │   ├── 001_Titlepage.tex     # Titelseite
        │   ├── 010_Einleitung.tex    # Einleitung
        │   ├── 020_Datensatz.tex     # Datensatz
        │   ├── 030_CNN_Eigenentwicklung.tex
        │   ├── 040_TransferLearning.tex
        │   ├── 050_Lokalisierung_Lesbarkeit.tex
        │   ├── 060_Fazit.tex
        │   └── [weitere Kapitel...]
        └── Figures/
            ├── jpg/
            ├── pdf/
            ├── png/
            ├── svg/
            ├── tikz/
            └── [Bilder für Thesis]
```

---

## 💾 Abhängigkeiten und Installation

### Python-Abhängigkeiten

```txt
numpy              # Numerische Berechnungen
opencv-python      # Computer Vision
torch              # Deep Learning
torchvision        # Vision Utilities für PyTorch
tqdm               # Progress Bars
matplotlib         # Visualisierung
scikit-learn       # ML Utilities
pyyaml             # YAML Konfiguration
pillow             # Bildverarbeitung
```

### Installation

```bash
# Python Virtual Environment (empfohlen)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

---

## 📈 Metriken und Evaluierung

### Trainings-Metriken

Das System verfolgt folgende Metriken pro Epoche:

- **Loss** (Training, Validierung): Cross-Entropy oder Focal Loss
- **Accuracy** (Training, Validierung, Test): % korrekt klassifiziert
- **Precision, Recall, F1-Score**: Pro Klasse
- **Confusion Matrix**: Visualisierung Verwechslungen
- **ROC-AUC**: Area Under Curve

### CSV-Log Format

```
epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc
0,0.523,0.761,0.489,0.798,0.512,0.785
1,0.412,0.823,0.398,0.831,0.401,0.829
...
```

### Ausgabe-Visualisierungen

- `curves_loss.png` - Loss-Verlauf (Train/Val)
- `curves_acc.png` - Accuracy-Verlauf (Train/Val/Test)
- `confusion_val.png` - Confusion Matrix Validierung
- `confusion_test.png` - Confusion Matrix Test

---

## 🔍 Troubleshooting und Best Practices

### Problem: Niedriges Datensatz-Accuracy

**Mögliche Ursachen**:
1. Zu kleine ROI-Box → QR-Code wird abgeschnitten
2. Schlechte Patch-Extraktion → Fehler im `extract_rois_to_disk.py`
3. Unausgeglichene Klassen → Zu viele "no_qr" Patches
4. **Lösung**: ROI-Parameter in `roi_tuner_gui.py` neu tunen

### Problem: Modell trainiert nicht (Loss sinkt nicht)

**Mögliche Ursachen**:
1. Learning Rate zu hoch → Verwenden Sie `exp_lr_3e-4.yaml`
2. Korrupte Datensatz-Split-Datei → Löschen Sie `data/splits/patches_split_seed42.json`
3. Zu wenig Training-Daten → Nutzen Sie Augmentierung in `augment_flip_patches.py`

### Problem: GUI friert ein bei großen Bildern

**Lösung**:
- Max-Dimension reduzieren in `qr_rectify_gui.py`
- Batch-Size anpassen mit `--batch-size 32`

### Best Practices

1. **Reproduzierbarkeit**: Seed immer auf 42 setzen
2. **Experiment-Tracking**: Konfigurationen mit sprechenden Namen speichern
3. **Datensatz-Balance**: Min. 50% QR / 50% No-QR
4. **Augmentierung**: Aktivieren für < 1000 Samples pro Klasse
5. **Transfer Learning**: Für kleine Datensätze (< 5000 Samples) verwenden

---

## 🎓 Wissenschaftlicher Kontext

### Thesis-Kapitel

1. **Einleitung** (`010_Einleitung.tex`)
   - Motivation: Waschanlage-Szenario
   - Anforderungen und Herausforderungen
   - Aufgabenstellung

2. **Datensatz** (`020_Datensatz.tex`)
   - Datenakquisition
   - Patch-Extraktion und Labels

3. **CNN-Eigenentwicklung** (`030_CNN_Eigenentwicklung.tex`)
   - Architektur-Design
   - Trainings-Strategie
   - Experimente und Ergebnisse

4. **Transfer Learning** (`040_TransferLearning.tex`)
   - ResNet-18, EfficientNet-B0, MobileNet-V3
   - Performance-Vergleiche

5. **QR-Code Lokalisierung und Lesbarkeit** (`050_Lokalisierung_Lesbarkeit.tex`)
   - Positionsbestimmung (Methode A vs. B)
   - Perspektiv-Analyse (rot_x, rot_y)
   - Lesbarkeitsprüfung

6. **Fazit und Ausblick** (`060_Fazit.tex`)

---

## 📞 Kontakt und Support

Für Fragen zum Projekt:
- **Paul Glaser**: glpa1013
- **Tim Schäfer**: scti1057

---

## 📝 Changelog und Versioning

### Version 2.0 (Aktuell)
- ✅ Robuste Line-basierte Erkennung (Plan B)
- ✅ Perspektiv-Analyse (rot_x, rot_y)
- ✅ Verbesserte ROI-Tuning GUI
- ✅ Transfer Learning mit 3 Modellen
- ✅ Umfassende Experiment-Suite

### Version 1.0 (Initial)
- Basis-CNN-Architektur
- OpenCV QR-Detektor
- ROI-Extraktion

---

**Zuletzt aktualisiert**: 2. Februar 2026  
**Lizenz**: [Projekt der Hochschule Karlsruhe]
