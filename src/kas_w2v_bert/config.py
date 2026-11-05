"""
Configuration Schema & Hardware Auto-Tuner for Wav2Vec2-BERT 2.0 Kashmiri ASR.
==============================================================================
Supports loading from a YAML file and overriding any parameter from the CLI.
Automatically adapts precision (BF16/FP16/TF32) and batch sizes to Colab Pro GPUs (A100, L4, V100, T4).
"""

import argparse
import dataclasses
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TrainingConfig:
    # Model & Dataset Source ("auto", "huggingface", or "gdrive")
    model_name_or_path: str = "facebook/w2v-bert-2.0"
    dataset_source: str = "auto"  # "auto", "huggingface", or "gdrive"
    dataset_name: str = "munzirahangar/kas-news-dataset"
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    drive_dataset_dir: Optional[str] = None  # e.g., /content/drive/MyDrive/nasr_kashmiri_asr/voicehub_dataset
    shard_glob: str = "batch_*"
    audio_column: str = "audio"
    text_column: str = "sentence"
    sampling_rate: int = 16000

    # Deterministic Train / Val / Test Splitting (cached to output_dir/artifacts/splits.json)
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42

    # Text & Audio Filtering / Preprocessing
    remove_diacritics: bool = False
    strip_punctuation: bool = True
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    cache_features_to_disk: bool = False

    # Model Architecture & Regularization (w2v-BERT 2.0 specific)
    add_adapter: bool = True
    freeze_base_model: bool = False
    freeze_conformer_layers: int = 0  # Freeze bottom N out of 24 Conformer layers (0 = train all)
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.05
    mask_feature_length: int = 10
    layerdrop: float = 0.05
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True

    # Persistent Paths (Point output_dir to Google Drive in Colab for fault-tolerance)
    output_dir: str = "/content/drive/MyDrive/nasr2_kashmiri_w2v_bert"
    hf_cache_dir: Optional[str] = None

    # Optimization & Schedule
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    num_train_epochs: float = 10.0
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Batching & Hardware Acceleration (None = auto-detect from GPU VRAM)
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    gradient_checkpointing: bool = True
    bf16: Optional[bool] = None
    fp16: Optional[bool] = None
    tf32: Optional[bool] = None
    dataloader_num_workers: int = 2
    group_by_length: bool = True

    # Checkpointing, Resilience & Hub Backup
    save_strategy: str = "steps"
    save_steps: int = 200
    eval_strategy: str = "steps"
    eval_steps: int = 200
    logging_steps: int = 25
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "da_wer"
    early_stopping_patience: int = 8
    auto_resume: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True

    def resolve_hardware_defaults(self) -> Dict[str, Any]:
        """
        Inspects available CUDA GPU (A100, L4, V100, T4, or CPU) and fills in optimal
        precision and batch size settings if not explicitly overridden by the user.
        """
        hw_info: Dict[str, Any] = {
            "device": "cpu",
            "gpu_name": "CPU",
            "vram_gb": 0.0,
            "compute_capability": (0, 0),
        }
        try:
            import torch
            if torch.cuda.is_available():
                hw_info["device"] = "cuda"
                hw_info["gpu_name"] = torch.cuda.get_device_name(0)
                hw_info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
                )
                hw_info["compute_capability"] = torch.cuda.get_device_capability(0)
        except Exception:
            pass

        major_cap = hw_info["compute_capability"][0]
        vram_gb = hw_info["vram_gb"]
        supports_bf16 = major_cap >= 8  # Ampere (A100, L4) or Hopper (H100)

        if self.bf16 is None:
            self.bf16 = bool(supports_bf16)
        if self.fp16 is None:
            self.fp16 = bool(hw_info["device"] == "cuda" and not self.bf16)
        if self.tf32 is None:
            self.tf32 = bool(major_cap >= 8)

        # Target effective batch size = 32 utterances (~8 minutes of speech per step)
        if self.per_device_train_batch_size is None:
            if vram_gb >= 70:      # A100 80GB / H100
                self.per_device_train_batch_size = 16
            elif vram_gb >= 35:    # A100 40GB
                self.per_device_train_batch_size = 8
            elif vram_gb >= 20:    # L4 24GB
                self.per_device_train_batch_size = 8
            else:                  # T4 / V100 16GB or smaller
                self.per_device_train_batch_size = 4

        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size

        if self.gradient_accumulation_steps is None:
            target_effective_batch = 32
            self.gradient_accumulation_steps = max(
                1, target_effective_batch // self.per_device_train_batch_size
            )

        return hw_info

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.output_dir, "artifacts")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    @property
    def best_model_dir(self) -> str:
        return os.path.join(self.output_dir, "best_model")

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.output_dir, "evaluation")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")

    def ensure_dirs(self) -> None:
        for d in [
            self.output_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.best_model_dir,
            self.eval_dir,
            self.logs_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")) and yaml is not None:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        return cls(**filtered)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate facebook/w2v-bert-2.0 for Kashmiri ASR (Colab Pro fault-tolerant)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/w2v_bert_kas_news.yaml).",
    )
    for field in dataclasses.fields(TrainingConfig):
        flag = f"--{field.name}"
        ftype = field.type
        if ftype is bool or ftype == Optional[bool]:
            parser.add_argument(
                flag,
                type=lambda v: str(v).lower() in ("true", "1", "yes", "y"),
                default=None,
                help=f"Override {field.name} (default: {field.default})",
            )
        elif ftype in (int, Optional[int]):
            parser.add_argument(flag, type=int, default=None, help=f"Override {field.name} (default: {field.default})")
        elif ftype in (float, Optional[float]):
            parser.add_argument(flag, type=float, default=None, help=f"Override {field.name} (default: {field.default})")
        else:
            parser.add_argument(flag, type=str, default=None, help=f"Override {field.name} (default: {field.default})")
    return parser


def load_config_from_cli(argv: Optional[list] = None) -> TrainingConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config and os.path.exists(args.config):
        cfg = TrainingConfig.from_yaml(args.config)
    else:
        cfg = TrainingConfig()

    for field in dataclasses.fields(TrainingConfig):
        cli_val = getattr(args, field.name, None)
        if cli_val is not None:
            setattr(cfg, field.name, cli_val)

    return cfg

