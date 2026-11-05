"""
CLI Entry Point for Wav2Vec2-BERT 2.0 Kashmiri ASR Fine-Tuning.
"""

from typing import Optional
from kas_w2v_bert.config import load_config_from_cli
from kas_w2v_bert.trainer import run_training_pipeline


def main(argv: Optional[list] = None) -> None:
    cfg = load_config_from_cli(argv)
    run_training_pipeline(cfg)


if __name__ == "__main__":
    main()

