#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "ERROR: python3.13 was not found in PATH."
  echo "Install Python 3.13 first, then rerun this script."
  exit 1
fi

PY_VER="$(python3.13 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "3.13" ]]; then
  echo "ERROR: expected Python 3.13, got $PY_VER"
  exit 1
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3.13-venv git ffmpeg libgl1 libglib2.0-0
fi

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo
python -m fight_detection.check_env
