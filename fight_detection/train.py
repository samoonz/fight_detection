from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .config import cache_root, load_config, runs_root
from .models import MODEL_BUILDERS


def load_feature_dataset(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    manifest_path = cache_root(cfg).parent / "msar_feature_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found: {manifest_path}. "
            "Run feature extraction first."
        )

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["ok"].astype(bool)].reset_index(drop=True)

    if not bool(cfg["allow_partial_extraction"]) and len(manifest) != 350:
        raise RuntimeError(
            f"Expected 350 successful feature files, found {len(manifest)}"
        )

    seqs: list[np.ndarray] = []
    labels: list[int] = []

    for row in manifest.to_dict("records"):
        with np.load(Path(row["cache"]), allow_pickle=False) as data:
            seqs.append(data["x"].astype(np.float32))
            labels.append(int(data["y"]))

    y = np.asarray(labels, dtype=np.int64)
    max_timesteps = max(x.shape[0] for x in seqs)

    x = pad_sequences(
        seqs,
        maxlen=max_timesteps,
        dtype="float32",
        padding="post",
        truncating="post",
        value=0.0,
    )

    print(f"X: {x.shape} ({x.nbytes / 1024**2:.1f} MB)")
    print(f"y: {y.shape}, counts={np.bincount(y)}")
    print(f"MAX_TIMESTEPS: {max_timesteps}")
    return x, y, manifest, max_timesteps


def make_split(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    split_mode: str,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(manifest))

    if split_mode == "paper_like_random":
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        return np.asarray(train_idx), np.asarray(test_idx)

    if split_mode == "pair_group_strict":
        groups = (
            manifest[["event_group", "label"]]
            .drop_duplicates("event_group")
            .reset_index(drop=True)
        )

        train_groups, test_groups = train_test_split(
            groups["event_group"].to_numpy(),
            test_size=test_size,
            random_state=seed,
            stratify=groups["label"].to_numpy(),
        )

        train_set = set(train_groups.tolist())
        test_set = set(test_groups.tolist())

        train_idx = np.asarray(
            [i for i, g in enumerate(manifest["event_group"]) if g in train_set]
        )
        test_idx = np.asarray(
            [i for i, g in enumerate(manifest["event_group"]) if g in test_set]
        )

        overlap = set(manifest.iloc[train_idx]["event_group"]) & set(
            manifest.iloc[test_idx]["event_group"]
        )
        if overlap:
            raise RuntimeError(f"Grouped split leakage detected: {overlap}")

        return train_idx, test_idx

    raise ValueError(f"Unknown split_mode={split_mode}")


def train_one(
    cfg: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    manifest: pd.DataFrame,
    max_timesteps: int,
    split_mode: str,
    model_key: str,
    seed: int,
) -> dict[str, Any]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    train_idx, test_idx = make_split(
        manifest=manifest,
        labels=y,
        split_mode=split_mode,
        test_size=float(cfg["test_size"]),
        seed=seed,
    )

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = MODEL_BUILDERS[model_key](
        timesteps=max_timesteps,
        frame_dim=int(cfg["frame_dim"]),
        cfg=cfg,
    )

    output_dir = runs_root(cfg) / split_mode / f"{model.name}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Held-out test data is never used as Keras validation data.
    history = model.fit(
        x_train,
        tf.keras.utils.to_categorical(y_train, 2),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        verbose=2,
    )

    model_path = output_dir / "final.keras"
    model.save(model_path)

    probs = model.predict(
        x_test,
        batch_size=int(cfg["batch_size"]),
        verbose=0,
    )
    pred = probs.argmax(axis=1)

    metrics: dict[str, Any] = {
        "model": model.name,
        "model_key": model_key,
        "seed": int(seed),
        "split_mode": split_mode,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "precision_violent": float(
            precision_score(y_test, pred, zero_division=0)
        ),
        "recall_violent": float(
            recall_score(y_test, pred, zero_division=0)
        ),
        "f1_violent": float(
            f1_score(y_test, pred, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, pred, average="macro", zero_division=0)
        ),
        "model_path": str(model_path.resolve()),
    }

    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "test_indices.npy", test_idx)

    pd.DataFrame(history.history).to_csv(
        output_dir / "history.csv",
        index=False,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        classification_report(
            y_test,
            pred,
            target_names=["non-violent", "violent"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print(metrics)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MSAR recurrent classifier."
    )
    parser.add_argument(
        "--config",
        default="configs/msar_local.json",
    )
    parser.add_argument(
        "--split",
        choices=["paper_like_random", "pair_group_strict"],
        default="paper_like_random",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_BUILDERS),
        default="bilstm_gru",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="Override seeds from config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    x, y, manifest, max_timesteps = load_feature_dataset(cfg)

    seeds = (
        args.seeds
        if args.seeds
        else [int(s) for s in cfg["seeds"]]
    )

    output_root = runs_root(cfg) / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": cfg["_config_path"],
        "split_mode": args.split,
        "model_key": args.model,
        "max_timesteps": int(max_timesteps),
        "frame_dim": int(cfg["frame_dim"]),
        "frame_stride": int(cfg["frame_stride"]),
        "coordinate_mode": cfg["coordinate_mode"],
        "seeds": seeds,
    }

    with (output_root / "metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    results = []

    for seed in seeds:
        results.append(
            train_one(
                cfg=cfg,
                x=x,
                y=y,
                manifest=manifest,
                max_timesteps=max_timesteps,
                split_mode=args.split,
                model_key=args.model,
                seed=int(seed),
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        output_root / "results.csv",
        index=False,
    )

    print(results_df.to_string(index=False))
    print(
        f"Mean accuracy: "
        f"{100.0 * results_df['accuracy'].mean():.2f}%"
    )
    print(
        f"Max accuracy:  "
        f"{100.0 * results_df['accuracy'].max():.2f}%"
    )
    print(
        f"Results: "
        f"{(output_root / 'results.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
