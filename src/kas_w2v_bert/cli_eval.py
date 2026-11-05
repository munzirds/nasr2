"""
CLI Entry Point for Resumable Wav2Vec2-BERT 2.0 Kashmiri ASR Evaluation.
"""

import dataclasses
from typing import Optional

from kas_w2v_bert.config import TrainingConfig, build_arg_parser
from kas_w2v_bert.evaluator import run_resumable_evaluation


def main(argv: Optional[list] = None) -> None:
    parser = build_arg_parser()
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "validation", "train"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint directory (defaults to <output_dir>/best_model or latest checkpoint).",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Ignore cached eval_predictions_<split>.jsonl and re-run evaluation from scratch.",
    )
    args = parser.parse_args(argv)

    if args.config:
        cfg = TrainingConfig.from_yaml(args.config)
    else:
        cfg = TrainingConfig()

    for field in dataclasses.fields(TrainingConfig):
        cli_val = getattr(args, field.name, None)
        if cli_val is not None:
            setattr(cfg, field.name, cli_val)

    run_resumable_evaluation(
        cfg=cfg,
        split=args.eval_split,
        checkpoint_path=args.checkpoint_path,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()

