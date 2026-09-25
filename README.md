# fight_detection

Local Ubuntu/Python 3.13 reproduction of an MSAR-style violence-detection pipeline:

**YOLOv9 → DeepSORT → MediaPipe PoseLandmarker → multi-person skeleton sequence → Fusion BiLSTM-GRU**

This repository is designed to reproduce the public MSAR 2025 pipeline as closely as possible with public tooling. It is **not** the authors' original source code: the exact YOLOv9 sub-variant, DeepSORT/ReID settings, person-padding details, and exact recurrent layer layout are not fully public. The defaults here are the reviewed reconstruction used in this project.

## Target environment

- Ubuntu x86-64
- Python **3.13**
- NVIDIA GPU recommended
- MediaPipe **0.10.35** Tasks API
- Ultralytics **8.4.149**
- deep-sort-realtime **1.3.2**
- TensorFlow **2.21.0**

MediaPipe 0.10.35 uses the Tasks API here; the old `mp.solutions.pose` API is not used.

## 1. Clone and create the Python 3.13 environment

```bash
git clone https://github.com/samoonz/fight_detection.git
cd fight_detection
bash scripts/setup_ubuntu_313.sh
source .venv/bin/activate
```

Check the environment:

```bash
python -m fight_detection.check_env
```

## 2. Download and validate AIRTLab

```bash
python -m fight_detection.dataset --config configs/msar_local.json --download
```

Expected: 350 videos = 120 non-violent + 230 violent.

## 3. Extract features

Smoke test:

```bash
python -m fight_detection.feature_extractor \
  --config configs/msar_local.json \
  --video data/AIRTLab/violence-detection-dataset/non-violent/cam1/1.mp4
```

Full extraction:

```bash
python -m fight_detection.feature_extractor \
  --config configs/msar_local.json \
  --all
```

Pipeline per frame:

```text
YOLOv9 person detection
 ↓
DeepSORT identity tracking
 ↓
MediaPipe PoseLandmarker on each person crop
 ↓
33 × (x, y, visibility) = 99 values/person
 ↓
up to 4 tracked people
 ↓
396 values/frame
```

Features are cached under `cache/msar_features/` with a configuration fingerprint.

## 4. Train Fusion BiLSTM-GRU

Paper-like random video split:

```bash
python -m fight_detection.train \
  --config configs/msar_local.json \
  --split paper_like_random
```

Stricter grouped split that keeps cam1/cam2 of the same staged event together:

```bash
python -m fight_detection.train \
  --config configs/msar_local.json \
  --split pair_group_strict
```

One seed only:

```bash
python -m fight_detection.train \
  --config configs/msar_local.json \
  --split pair_group_strict \
  --seeds 11
```

Outputs are written under `runs/<split>/`. The held-out test split is not used as Keras validation data.

## 5. Inference

```bash
python -m fight_detection.infer \
  --config configs/msar_local.json \
  --model runs/pair_group_strict/Fusion_BiLSTM_GRU_seed11/final.keras \
  --metadata runs/pair_group_strict/metadata.json \
  --video /path/to/video.mp4
```

Long videos use overlapping temporal windows instead of silently truncating the tail.

## Reproduction fidelity

Confirmed at pipeline level by the published MSAR work: human detection, DeepSORT tracking, MediaPipe Pose skeleton extraction, person-count padding, and LSTM/BiLSTM-GRU classifier families.

Reconstruction assumptions because exact public source/config is unavailable include the exact YOLOv9 sub-variant, DeepSORT/ReID settings, fixed `max_people=4`, recurrent layer sizes/counts, and crop-normalized coordinate representation.
