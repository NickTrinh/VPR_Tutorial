#!/usr/bin/env bash
#
# One-shot setup for the IEEE RAL submission reproducer.
# Installs dependencies and downloads the 6 datasets used in the paper.
#
# Expected flow:
#   bash setup.sh
#   python -m experiments.extract_dinov2_salad_all   # GPU required
#   python -m experiments.final_all_datasets         # CPU OK, ~few minutes
#
# Run from the repository root. Activate your Python env first
# (conda create -n vprtutorial python=3.11 && conda activate vprtutorial).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo " VPR RAL Reproducer — Setup"
echo "=========================================="
echo "Repo: $REPO_ROOT"
echo

# ---------- 1. Python deps ----------
echo "[1/3] Installing Python dependencies..."
if ! command -v pip >/dev/null 2>&1; then
    echo "ERROR: pip not found. Activate a Python 3.11 env first:"
    echo "  conda create -n vprtutorial python=3.11 && conda activate vprtutorial"
    exit 1
fi
pip install -r requirements.txt
echo

# ---------- 2. Datasets ----------
echo "[2/3] Downloading datasets into images/..."
mkdir -p images
cd images

download_zip () {
    # $1 = url, $2 = expected top-level directory after unzip
    local url="$1"
    local target="$2"
    local zipname="$(basename "$url")"
    if [ -d "$target" ]; then
        echo "  [skip] $target already exists"
        return
    fi
    echo "  [get]  $url"
    wget -q --show-progress "$url"
    unzip -q "$zipname"
    rm -f "$zipname"
}

# --- Bonn (1.2 GB) ---
download_zip "http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/bonn_example.zip" "bonn_example"

# --- Freiburg (738 MB) ---
download_zip "http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/freiburg_example.zip" "freiburg_example"

# --- GardensPoint (32 MB) — from TU-Chemnitz mirror used by stschubert/VPR_Tutorial ---
if [ ! -d "GardensPoint" ]; then
    echo "  [get]  GardensPoint_Walking.zip"
    wget -q --show-progress "https://www.tu-chemnitz.de/etit/proaut/datasets/GardensPoint_Walking.zip"
    unzip -q GardensPoint_Walking.zip
    rm -f GardensPoint_Walking.zip
else
    echo "  [skip] GardensPoint/ already exists"
fi

# --- SFU Mountain (72 MB) ---
if [ ! -d "SFU" ]; then
    echo "  [get]  SFU.zip"
    wget -q --show-progress "https://www.tu-chemnitz.de/etit/proaut/datasets/SFU.zip"
    unzip -q SFU.zip
    rm -f SFU.zip
else
    echo "  [skip] SFU/ already exists"
fi

# --- ESSEX3IN1 (1.5 GB) — git clone ---
if [ ! -d "ESSEX3IN1" ]; then
    echo "  [get]  ESSEX3IN1 (git clone)"
    git clone --depth 1 https://github.com/MubarizZaffar/ESSEX3IN1-Dataset ESSEX3IN1
    echo "  NOTE: verify images/ESSEX3IN1/{reference_combined,query_combined}/ exist."
    echo "        If the repo layout differs, re-organize to match the expected path."
else
    echo "  [skip] ESSEX3IN1/ already exists"
fi

cd "$REPO_ROOT"
echo

# ---------- 3. Nordland (manual) ----------
echo "[3/3] Nordland-500 — manual step required"
echo "  The 500-image subset is not auto-downloaded (HuggingFace gated/LFS)."
echo "  Download from: https://huggingface.co/datasets/Somayeh-h/Nordland"
echo "  Place the first 500 winter and summer frames (1 fps) as:"
echo "    images/Nordland_Mini/winter/*.jpg"
echo "    images/Nordland_Mini/summer/*.jpg"
echo

echo "=========================================="
echo " Setup done. Next steps:"
echo "   python -m experiments.extract_dinov2_salad_all"
echo "   python -m experiments.final_all_datasets"
echo "=========================================="
