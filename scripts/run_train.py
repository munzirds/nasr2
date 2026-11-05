#!/usr/bin/env python3
"""
Standalone launcher for Kashmiri Wav2Vec2-BERT 2.0 CTC Fine-Tuning.
===================================================================
Example (Colab Pro):
    python scripts/run_train.py --config configs/w2v_bert_kas_news.yaml \
        --output_dir /content/drive/MyDrive/nasr2_kashmiri_w2v_bert

Automatically resumes from the latest valid checkpoint in `<output_dir>/checkpoints`
if interrupted or restarted.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kas_w2v_bert.cli_train import main

if __name__ == "__main__":
    main()

