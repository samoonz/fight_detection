from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .config import dataset_repo_dir, dataset_root, load_config, resolve_path

EXPECTED = {
    ("non-violent", "cam1"): 60,
    ("non-violent", "cam2"): 60,
    ("violent", "cam1"): 115,
    ("violent", "cam2"): 115,
}


def download_dataset(cfg: dict[str, Any]) -> Path:
    repo_dir = dataset_repo_dir(cfg)
    if repo_dir.exists():
        print(f"Dataset repository already exists: {repo_dir}")
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            cfg["dataset_repo_url"],
            str(repo_dir),
        ],
        check=True,
    )
    return repo_dir


def build_manifest(cfg: dict[str, Any], validate: bool = True) -> list[dict[str, Any]]:
    root = dataset_root(cfg)
    if not root.exists():
        raise FileNotFoundError(
            f"AIRTLab dataset not found at {root}. "
            "Run: python -m fight_detection.dataset --download"
        )

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.mp4")):
        parts = [x.lower() for x in path.parts]

        if "non-violent" in parts:
            label = 0
            class_name = "non-violent"
        elif "violent" in parts:
            label = 1
            class_name = "violent"
        else:
            raise ValueError(f"Unexpected dataset path: {path}")

        if "cam1" in parts:
            camera = "cam1"
        elif "cam2" in parts:
            camera = "cam2"
        else:
            raise ValueError(f"Camera folder missing from path: {path}")

        rows.append(
            {
                "path": str(path.resolve()),
                "label": label,
                "class": class_name,
                "camera": camera,
                "clip_id": path.stem,
                "event_group": f"{class_name}:{path.stem}",
            }
        )

    if validate:
        validate_manifest(rows)
    return rows


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 350:
        raise RuntimeError(f"Expected 350 AIRTLab videos, found {len(rows)}")

    counts = Counter((r["class"], r["camera"]) for r in rows)
    if dict(counts) != EXPECTED:
        raise RuntimeError(f"Unexpected AIRTLab distribution: {dict(counts)}")

    first = Path(rows[0]["path"])
    cap = cv2.VideoCapture(str(first))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"OpenCV cannot decode the first dataset video: {first}")


def save_manifest(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "class", "camera", "clip_id", "event_group"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/validate AIRTLab.")
    parser.add_argument("--config", default="configs/msar_local.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--manifest", default="cache/airtlab_manifest.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.download:
        download_dataset(cfg)

    rows = build_manifest(cfg, validate=True)
    manifest_path = resolve_path(args.manifest)
    save_manifest(rows, manifest_path)

    counts = Counter(r["class"] for r in rows)
    print(f"Dataset root: {dataset_root(cfg)}")
    print(f"Total videos: {len(rows)}")
    print(f"Non-violent: {counts['non-violent']}")
    print(f"Violent: {counts['violent']}")
    print(f"Unique staged events: {len(set(r['event_group'] for r in rows))}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
