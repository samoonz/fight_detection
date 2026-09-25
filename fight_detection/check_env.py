from __future__ import annotations

import shutil
import sys


def main() -> None:
    print("Python:", sys.version)
    if sys.version_info[:2] != (3, 13):
        raise SystemExit(
            f"Python 3.13 is required; current runtime is "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )

    import cv2
    import mediapipe as mp
    import tensorflow as tf
    import torch
    import ultralytics

    print("OpenCV:", cv2.__version__)
    print("MediaPipe:", getattr(mp, "__version__", "unknown"))
    print("TensorFlow:", tf.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("PyTorch GPU:", torch.cuda.get_device_name(0))

    print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))

    if not hasattr(mp, "tasks"):
        raise SystemExit("MediaPipe Tasks API is unavailable.")

    for binary in ["git", "ffmpeg"]:
        print(f"{binary}:", shutil.which(binary) or "NOT FOUND")

    print("Environment check: OK")


if __name__ == "__main__":
    main()
