#!/bin/bash
set -e

echo "==[ Hunter StegAnalyzer Install Script ]=="
sudo apt update && sudo apt install -y \
  python3-pip python3-venv python3-dev build-essential \
  libmagic-dev libopencv-dev exiftool binwalk foremost \
  steghide outguess ffmpeg sox tesseract-ocr zsteg \
  git curl unzip libsqlite3-dev graphviz imagemagick \
  libsm6 libxext6 libxrender-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

for d in models wordlists logs reports data static templates; do
    [ -d "$d" ] || mkdir "$d"
done

echo "[*] Installation complete."
echo "To run a scan:"
echo "  source .venv/bin/activate"
echo "  python steg_main.py --file suspicious.png"
echo "  python steg_main.py --dir /data/batch/"
echo "Dashboard: http://127.0.0.1:8080"
