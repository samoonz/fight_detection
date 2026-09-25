from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import load_config
from .feature_extractor import MSARFeatureExtractor


def predict_windows(
    sequence: np.ndarray,
    model: keras.Model,
    window_len: int,
    batch_size: int,
) -> list[tuple[int, int, float]]:
    if len(sequence) <= window_len:
        batch = pad_sequences(
            [sequence],
            maxlen=window_len,
            dtype="float32",
            padding="post",
            truncating="post",
            value=0.0,
        )
        probability = float(
            model.predict(batch, verbose=0)[0, 1]
        )
        return [(0, len(sequence), probability)]

    stride = max(1, window_len // 2)

    starts = list(
        range(
            0,
            len(sequence) - window_len + 1,
            stride,
        )
    )

    last = len(sequence) - window_len
    if starts[-1] != last:
        starts.append(last)

    windows = np.stack(
        [
            sequence[start : start + window_len]
            for start in starts
        ],
        axis=0,
    ).astype(np.float32)

    probabilities = model.predict(
        windows,
        batch_size=min(batch_size, len(windows)),
        verbose=0,
    )[:, 1]

    return [
        (
            int(start),
            int(start + window_len),
            float(prob),
        )
        for start, prob in zip(starts, probabilities)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer fight probability."
    )
    parser.add_argument(
        "--config",
        default="configs/msar_local.json",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    with Path(args.metadata).open(
        "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    if int(metadata["frame_dim"]) != int(cfg["frame_dim"]):
        raise RuntimeError(
            f"Feature dimension mismatch: "
            f"model={metadata['frame_dim']}, "
            f"config={cfg['frame_dim']}"
        )

    if (
        metadata["coordinate_mode"]
        != cfg["coordinate_mode"]
    ):
        raise RuntimeError(
            "coordinate_mode mismatch between "
            "trained model metadata and config."
        )

    model = keras.models.load_model(args.model)

    with MSARFeatureExtractor(cfg) as extractor:
        sequence, _, stats = extractor.extract_video(
            args.video,
            label=0,
            cache_path=None,
            force=True,
        )

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not fps or fps <= 0:
        fps = 30.0

    windows = predict_windows(
        sequence=sequence,
        model=model,
        window_len=int(metadata["max_timesteps"]),
        batch_size=int(cfg["batch_size"]),
    )

    probs = np.asarray(
        [prob for _, _, prob in windows],
        dtype=np.float32,
    )

    p_mean = float(probs.mean())
    p_max = float(probs.max())

    print(f"Video: {Path(args.video).resolve()}")
    print(f"Feature shape: {sequence.shape}")
    print(f"Pose stats: {stats}")
    print(f"Number of windows: {len(windows)}")
    print(f"Mean P(violent): {p_mean:.4f}")
    print(f"Max  P(violent): {p_max:.4f}")

    ranked = sorted(
        windows,
        key=lambda item: item[2],
        reverse=True,
    )[:10]

    print("\nTop windows:")

    for start, end, probability in ranked:
        t0 = (
            start
            * int(cfg["frame_stride"])
            / fps
        )
        t1 = (
            end
            * int(cfg["frame_stride"])
            / fps
        )

        print(
            f"{t0:8.2f}s -> {t1:8.2f}s | "
            f"P(violent)={probability:.4f}"
        )

    decision_probability = (
        p_mean
        if len(windows) == 1
        else p_max
    )

    label = (
        "VIOLENT"
        if decision_probability >= args.threshold
        else "NON-VIOLENT"
    )

    print(f"\nPrediction: {label}")
    print(
        f"Decision P(violent): "
        f"{decision_probability:.4f}"
    )
    print(
        f"Threshold: "
        f"{args.threshold:.4f}"
    )


if __name__ == "__main__":
    main()
