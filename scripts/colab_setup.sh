#!/usr/bin/env bash
# ============================================================================
# One-Command Google Colab Pro Setup Script for Kashmiri w2v-BERT 2.0 ASR
# ============================================================================
set -euo pipefail

echo "============================================================"
echo "  Setting up Kashmiri Wav2Vec2-BERT 2.0 Environment (Colab)"
echo "============================================================"

pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .

python3 -c '
import torch
print("------------------------------------------------------------")
print("PyTorch version:", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:     ", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU VRAM:        {vram:.2f} GB")
    print("Compute Cap:    ", torch.cuda.get_device_capability(0))
print("------------------------------------------------------------")
'
echo "[✓] Setup complete!"

