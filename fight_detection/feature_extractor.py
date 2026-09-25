from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.tracker import Tracker
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm.auto import tqdm
from ultralytics import YOLO

from .config import cache_root, dataset_root, load_config, weights_root
from .dataset import build_manifest


class MSARFeatureExtractor:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device: int | str = 0 if torch.cuda.is_available() else "cpu"

        self.pose_model_path = self._ensure_pose_model()
        self.detector = YOLO(cfg["yolo_weight"])

        # Construct DeepSort only once so its ReID embedder is loaded once.
        self.deepsort = DeepSort(
            max_iou_distance=float(cfg["deepsort_max_iou_distance"]),
            max_age=int(cfg["deepsort_max_age"]),
            n_init=int(cfg["deepsort_n_init"]),
            max_cosine_distance=float(cfg["deepsort_max_cosine_distance"]),
            embedder="mobilenet",
            half=torch.cuda.is_available(),
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
        )

        self.pose_model = self._make_pose()
        self.cfg_hash = self._config_hash()

        print(f"YOLO device: {self.device}")
        print(f"Pose model: {self.pose_model_path}")
        print(f"Feature cache hash: {self.cfg_hash}")

    def close(self) -> None:
        if getattr(self, "pose_model", None) is not None:
            self.pose_model.close()
            self.pose_model = None

    def __enter__(self) -> "MSARFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_pose_model(self) -> Path:
        root = weights_root(self.cfg)
        root.mkdir(parents=True, exist_ok=True)
        path = root / self.cfg["pose_model_filename"]

        if path.exists() and path.stat().st_size > 1_000_000:
            return path

        tmp = path.with_suffix(path.suffix + ".part")
        if tmp.exists():
            tmp.unlink()

        print("Downloading MediaPipe PoseLandmarker Full...")
        urllib.request.urlretrieve(self.cfg["pose_model_url"], tmp)

        if not tmp.exists() or tmp.stat().st_size <= 1_000_000:
            raise RuntimeError("MediaPipe pose model download failed.")

        tmp.replace(path)
        return path

    def _make_pose(self) -> vision.PoseLandmarker:
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.pose_model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=float(
                self.cfg["pose_min_detection_confidence"]
            ),
            min_pose_presence_confidence=float(
                self.cfg["pose_min_presence_confidence"]
            ),
            min_tracking_confidence=float(
                self.cfg["pose_min_tracking_confidence"]
            ),
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _reset_deepsort_tracks(self) -> None:
        """Reset tracks between videos while reusing the loaded ReID embedder."""
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            float(self.cfg["deepsort_max_cosine_distance"]),
            None,
        )
        self.deepsort.tracker = Tracker(
            metric,
            max_iou_distance=float(self.cfg["deepsort_max_iou_distance"]),
            max_age=int(self.cfg["deepsort_max_age"]),
            n_init=int(self.cfg["deepsort_n_init"]),
            gating_only_position=False,
        )

    def _config_hash(self) -> str:
        keys = [
            "cache_schema_version",
            "yolo_weight",
            "det_conf",
            "det_iou",
            "det_imgsz",
            "deepsort_max_iou_distance",
            "deepsort_max_age",
            "deepsort_n_init",
            "deepsort_max_cosine_distance",
            "pose_model_filename",
            "pose_min_detection_confidence",
            "pose_min_presence_confidence",
            "pose_min_tracking_confidence",
            "max_people",
            "n_landmarks",
            "features_per_landmark",
            "coordinate_mode",
            "frame_stride",
            "crop_padding",
        ]
        payload = {key: self.cfg[key] for key in keys}
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _pose_feature(
        self,
        frame: np.ndarray,
        bbox: np.ndarray | list[float],
    ) -> tuple[np.ndarray, bool]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad = float(self.cfg["crop_padding"])
        px = int(pad * bw)
        py = int(pad * bh)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(width, x2 + px)
        y2 = min(height, y2 + py)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.cfg["person_dim"], np.float32), False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.cfg["person_dim"], np.float32), False

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.pose_model.detect(mp_image)

        if not result.pose_landmarks:
            return np.zeros(self.cfg["person_dim"], np.float32), False

        landmarks = result.pose_landmarks[0]
        n_landmarks = int(self.cfg["n_landmarks"])
        if len(landmarks) < n_landmarks:
            return np.zeros(self.cfg["person_dim"], np.float32), False

        crop_h, crop_w = crop.shape[:2]
        feat: list[float] = []

        for lm in landmarks[:n_landmarks]:
            if self.cfg["coordinate_mode"] == "crop_normalized":
                x = float(lm.x)
                y = float(lm.y)
            elif self.cfg["coordinate_mode"] == "global_normalized":
                x = (x1 + float(lm.x) * crop_w) / max(width, 1)
                y = (y1 + float(lm.y) * crop_h) / max(height, 1)
            else:
                raise ValueError(
                    f"Unknown coordinate_mode={self.cfg['coordinate_mode']}"
                )

            visibility = float(getattr(lm, "visibility", 0.0))
            feat.extend(
                [
                    float(np.clip(x, 0.0, 1.0)),
                    float(np.clip(y, 0.0, 1.0)),
                    float(np.clip(visibility, 0.0, 1.0)),
                ]
            )

        arr = np.asarray(feat, dtype=np.float32)
        if arr.size != int(self.cfg["person_dim"]):
            raise RuntimeError(
                f"Unexpected pose feature size: {arr.size} "
                f"!= {self.cfg['person_dim']}"
            )
        return arr, True

    def _read_cache(self, path: Path) -> tuple[np.ndarray, int] | None:
        if not path.exists():
            return None

        with np.load(path, allow_pickle=False) as data:
            if "cfg_hash" not in data.files:
                return None
            if str(data["cfg_hash"].item()) != self.cfg_hash:
                return None
            return data["x"].astype(np.float32), int(data["y"])

    def extract_video(
        self,
        video_path: str | Path,
        label: int = 0,
        cache_path: str | Path | None = None,
        force: bool = False,
    ) -> tuple[np.ndarray, int, dict[str, float | int]]:
        video_path = Path(video_path)
        cache_path = Path(cache_path) if cache_path is not None else None

        if cache_path is not None and not force:
            cached = self._read_cache(cache_path)
            if cached is not None:
                x, y = cached
                with np.load(cache_path, allow_pickle=False) as data:
                    attempts = int(data["pose_attempts"]) if "pose_attempts" in data.files else -1
                    success = int(data["pose_success"]) if "pose_success" in data.files else -1
                stats = {
                    "pose_attempts": attempts,
                    "pose_success": success,
                    "pose_success_rate": success / attempts if attempts > 0 else 0.0,
                }
                return x, y, stats

        self._reset_deepsort_tracks()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        track_to_slot: dict[str, int] = {}
        track_last_seen: dict[str, int] = {}
        frame_features: list[np.ndarray] = []

        frame_idx = -1
        processed_idx = -1
        pose_attempts = 0
        pose_success = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                if frame_idx % int(self.cfg["frame_stride"]) != 0:
                    continue
                processed_idx += 1

                results = self.detector.predict(
                    frame,
                    classes=[0],
                    conf=float(self.cfg["det_conf"]),
                    iou=float(self.cfg["det_iou"]),
                    imgsz=int(self.cfg["det_imgsz"]),
                    device=self.device,
                    verbose=False,
                )

                ds_dets: list[tuple[list[float], float, str]] = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.detach().cpu().numpy()
                    confs = results[0].boxes.conf.detach().cpu().numpy()

                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = box.tolist()
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        ds_dets.append(
                            (
                                [float(x1), float(y1), float(w), float(h)],
                                float(conf),
                                "person",
                            )
                        )

                tracks = self.deepsort.update_tracks(ds_dets, frame=frame)

                stale_ids = [
                    tid
                    for tid, last_seen in track_last_seen.items()
                    if processed_idx - last_seen > int(self.cfg["deepsort_max_age"])
                ]
                for tid in stale_ids:
                    track_to_slot.pop(tid, None)
                    track_last_seen.pop(tid, None)

                current = np.zeros(
                    (int(self.cfg["max_people"]), int(self.cfg["person_dim"])),
                    dtype=np.float32,
                )

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue

                    tid = str(track.track_id)
                    ltrb = track.to_ltrb(orig=True, orig_strict=True)
                    if ltrb is None:
                        continue

                    if tid not in track_to_slot:
                        used = set(track_to_slot.values())
                        free_slots = [
                            slot
                            for slot in range(int(self.cfg["max_people"]))
                            if slot not in used
                        ]
                        if not free_slots:
                            continue
                        track_to_slot[tid] = free_slots[0]

                    slot = track_to_slot[tid]
                    track_last_seen[tid] = processed_idx

                    pose_attempts += 1
                    feat, found = self._pose_feature(frame, ltrb)
                    if found:
                        pose_success += 1
                    current[slot] = feat

                frame_features.append(current.reshape(-1))

        finally:
            cap.release()

        if not frame_features:
            raise RuntimeError(f"No frames processed from {video_path}")

        x = np.stack(frame_features).astype(np.float32)
        stats: dict[str, float | int] = {
            "pose_attempts": pose_attempts,
            "pose_success": pose_success,
            "pose_success_rate": pose_success / pose_attempts if pose_attempts > 0 else 0.0,
        }

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                x=x,
                y=np.int64(label),
                cfg_hash=np.asarray(self.cfg_hash),
                pose_attempts=np.int64(pose_attempts),
                pose_success=np.int64(pose_success),
                path=np.asarray(str(video_path.resolve())),
            )

        return x, int(label), stats


def extract_all(cfg: dict[str, Any], force: bool = False) -> pd.DataFrame:
    rows = build_manifest(cfg, validate=True)
    root = dataset_root(cfg)
    cache = cache_root(cfg)
    records: list[dict[str, Any]] = []

    with MSARFeatureExtractor(cfg) as extractor:
        for row in tqdm(rows, desc="MSAR feature extraction"):
            path = Path(row["path"])
            relative = path.relative_to(root)
            cache_path = cache / relative.with_suffix(".npz")

            try:
                x, _, stats = extractor.extract_video(
                    path,
                    label=int(row["label"]),
                    cache_path=cache_path,
                    force=force,
                )
                records.append(
                    {
                        **row,
                        "cache": str(cache_path.resolve()),
                        "n_frames": int(x.shape[0]),
                        "feature_dim": int(x.shape[1]),
                        **stats,
                        "ok": True,
                        "error": "",
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        **row,
                        "cache": str(cache_path.resolve()),
                        "n_frames": 0,
                        "feature_dim": int(cfg["frame_dim"]),
                        "pose_attempts": 0,
                        "pose_success": 0,
                        "pose_success_rate": 0.0,
                        "ok": False,
                        "error": repr(exc),
                    }
                )
                print(f"\nERROR: {path}: {exc}")

    manifest = pd.DataFrame(records)
    manifest_path = cache.parent / "msar_feature_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    successful = int(manifest["ok"].sum())
    print(f"Successful: {successful}/{len(manifest)}")
    print(f"Manifest: {manifest_path}")

    if not bool(cfg["allow_partial_extraction"]) and successful != 350:
        failed = manifest.loc[~manifest["ok"], ["path", "error"]]
        print(failed.to_string(index=False))
        raise RuntimeError(
            "Feature extraction did not succeed on all 350 videos. "
            "Fix failed videos before training a reproduction run."
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MSAR skeleton features.")
    parser.add_argument("--config", default="configs/msar_local.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--video")
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.all:
        extract_all(cfg, force=args.force)
        return

    with MSARFeatureExtractor(cfg) as extractor:
        x, _, stats = extractor.extract_video(
            args.video,
            label=args.label,
            cache_path=None,
            force=True,
        )
        print(f"Video: {Path(args.video).resolve()}")
        print(f"Features: {x.shape}")
        print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
