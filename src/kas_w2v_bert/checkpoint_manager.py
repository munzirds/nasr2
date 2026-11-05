"""
Fault-Tolerant Checkpoint Manager & Colab Resilience Callbacks.
===============================================================
Protects against Google Colab runtime disconnections, browser timeouts, and Google Drive
partial writes:
1. Validates `checkpoint-*` folders before resuming (`is_valid_checkpoint`).
2. Quarantines incomplete/corrupted checkpoints caused by mid-save VM disconnects and falls back
   to the most recent intact checkpoint (`find_latest_valid_checkpoint`).
3. Continuously flushes step/eval metrics to `<output_dir>/logs/training_metrics.jsonl` (`os.fsync`)
   and updates `<output_dir>/run_status.json`.
4. Immediately exports the best model + processor to `<output_dir>/best_model` whenever validation
   `da_wer` improves, so the best model is always ready on Google Drive even if training is stopped early.
"""

from datetime import datetime, timezone
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:
    TrainerCallback = object  # type: ignore
    TrainerControl = Any      # type: ignore
    TrainerState = Any        # type: ignore
    TrainingArguments = Any   # type: ignore


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _is_valid_json_file(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict)
    except Exception:
        return False


def is_valid_checkpoint(ckpt_dir: str, require_optimizer: bool = True) -> bool:
    """
    Verifies that a Hugging Face Trainer checkpoint directory is complete and not truncated
    by a Colab disconnect during Google Drive file sync.
    """
    if not os.path.isdir(ckpt_dir):
        return False

    trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
    config_path = os.path.join(ckpt_dir, "config.json")
    if not _is_valid_json_file(trainer_state_path) or not _is_valid_json_file(config_path):
        return False

    # Check model weights exist and are non-empty
    weight_candidates = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    has_weights = any(
        os.path.isfile(os.path.join(ckpt_dir, w))
        and os.path.getsize(os.path.join(ckpt_dir, w)) > 0
        for w in weight_candidates
    )
    if not has_weights:
        return False

    if require_optimizer:
        opt_path = os.path.join(ckpt_dir, "optimizer.pt")
        sched_path = os.path.join(ckpt_dir, "scheduler.pt")
        if not (os.path.isfile(opt_path) and os.path.getsize(opt_path) > 0):
            return False
        if not (os.path.isfile(sched_path) and os.path.getsize(sched_path) > 0):
            return False

    return True


def list_checkpoint_dirs_sorted(checkpoints_dir: str) -> List[str]:
    """Returns `checkpoint-<N>` paths sorted from highest step `N` to lowest."""
    if not os.path.isdir(checkpoints_dir):
        return []
    found = []
    for entry in os.listdir(checkpoints_dir):
        m = CHECKPOINT_PATTERN.match(entry)
        if m:
            full_path = os.path.join(checkpoints_dir, entry)
            if os.path.isdir(full_path):
                found.append((int(m.group(1)), full_path))
    found.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in found]


def find_latest_valid_checkpoint(
    checkpoints_dir: str,
    require_optimizer: bool = True,
    quarantine_corrupted: bool = True,
) -> Optional[str]:
    """
    Scans `checkpoints_dir` for the most recent valid checkpoint.
    If the newest checkpoint was interrupted mid-save (incomplete files), quarantines it
    so Hugging Face Trainer does not crash, and returns the previous intact checkpoint.
    """
    candidates = list_checkpoint_dirs_sorted(checkpoints_dir)
    for ckpt_path in candidates:
        if is_valid_checkpoint(ckpt_path, require_optimizer=require_optimizer):
            return ckpt_path

        print(f"[!] Detected incomplete/corrupted checkpoint: {ckpt_path}")
        if quarantine_corrupted:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base = os.path.basename(ckpt_path)
            quarantine_dir = os.path.join(checkpoints_dir, f".corrupted_{base}_{ts}")
            try:
                shutil.move(ckpt_path, quarantine_dir)
                print(f"[!] Quarantined corrupted checkpoint to: {quarantine_dir}")
            except Exception as e:
                print(f"[!] Could not move corrupted checkpoint ({e}); attempting removal...")
                shutil.rmtree(ckpt_path, ignore_errors=True)

    return None


def resolve_best_model_or_checkpoint(output_dir: str) -> str:
    """
    Resolves the best available model directory inside `output_dir` for evaluation/inference:
    1. `<output_dir>/best_model` (if it contains valid weights + config)
    2. Latest valid checkpoint in `<output_dir>/checkpoints`
    3. `output_dir` itself (if passed directly as a checkpoint directory)
    """
    best_dir = os.path.join(output_dir, "best_model")
    if is_valid_checkpoint(best_dir, require_optimizer=False) or (
        os.path.isfile(os.path.join(best_dir, "config.json"))
        and (
            os.path.isfile(os.path.join(best_dir, "model.safetensors"))
            or os.path.isfile(os.path.join(best_dir, "pytorch_model.bin"))
        )
    ):
        return best_dir

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    latest_ckpt = find_latest_valid_checkpoint(
        checkpoints_dir, require_optimizer=False, quarantine_corrupted=False
    )
    if latest_ckpt:
        return latest_ckpt

    # Check if output_dir itself has checkpoints or is a checkpoint
    direct_ckpt = find_latest_valid_checkpoint(
        output_dir, require_optimizer=False, quarantine_corrupted=False
    )
    if direct_ckpt:
        return direct_ckpt

    if os.path.isfile(os.path.join(output_dir, "config.json")):
        return output_dir

    raise FileNotFoundError(
        f"No trained model or valid checkpoint found under '{output_dir}'. "
        f"Checked '{best_dir}' and '{checkpoints_dir}'."
    )


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Writes JSON atomically via a temporary file + fsync + replace to survive disconnects."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class ColabResilienceCallback(TrainerCallback):
    """
    TrainerCallback that:
    1. Streams every training/evaluation log to `<logs_dir>/training_metrics.jsonl` with `os.fsync`.
    2. Updates `<output_dir>/run_status.json` atomically after saves/evals.
    3. Saves a standalone copy of the model & processor to `<output_dir>/best_model` whenever
       `eval_da_wer` reaches a new best score.
    """

    def __init__(
        self,
        output_dir: str,
        logs_dir: str,
        best_model_dir: str,
        processor: Any,
    ) -> None:
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.best_model_dir = best_model_dir
        self.processor = processor
        self.metrics_jsonl_path = os.path.join(logs_dir, "training_metrics.jsonl")
        self.status_json_path = os.path.join(output_dir, "run_status.json")
        self.best_da_wer: float = float("inf")

        # Restore previously recorded best_da_wer if resuming across Colab sessions
        if os.path.isfile(self.status_json_path):
            try:
                with open(self.status_json_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("best_da_wer") is not None:
                    self.best_da_wer = float(prev["best_da_wer"])
            except Exception:
                pass

    def _update_status(self, state: TrainerState, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "status": status,
            "global_step": state.global_step,
            "epoch": round(float(state.epoch or 0.0), 4),
            "max_steps": state.max_steps,
            "best_metric": state.best_metric,
            "best_da_wer": None if self.best_da_wer == float("inf") else round(self.best_da_wer, 2),
            "best_model_checkpoint": state.best_model_checkpoint,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        atomic_write_json(self.status_json_path, payload)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._update_status(state, status="running")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        os.makedirs(self.logs_dir, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": state.global_step,
            "epoch": round(float(state.epoch or 0.0), 4),
            **logs,
        }
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # Also save processor inside each checkpoint so any checkpoint is self-contained
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir) and self.processor is not None:
            try:
                self.processor.save_pretrained(ckpt_dir)
            except Exception:
                pass
        self._update_status(
            state,
            status="running",
            extra={"latest_saved_checkpoint": ckpt_dir},
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if not metrics:
            return
        current_da_wer = metrics.get("eval_da_wer")
        if current_da_wer is not None and float(current_da_wer) < self.best_da_wer:
            self.best_da_wer = float(current_da_wer)
            if model is not None:
                os.makedirs(self.best_model_dir, exist_ok=True)
                model.save_pretrained(self.best_model_dir)
                if self.processor is not None:
                    self.processor.save_pretrained(self.best_model_dir)
                atomic_write_json(
                    os.path.join(self.best_model_dir, "best_metrics.json"),
                    {
                        "global_step": state.global_step,
                        "epoch": round(float(state.epoch or 0.0), 4),
                        "metrics": metrics,
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    f"\n[★] New best Validation DA-WER: {self.best_da_wer:.2f}% "
                    f"(saved standalone model to {self.best_model_dir})"
                )
        self._update_status(state, status="running", extra={"last_eval_metrics": metrics})

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._update_status(state, status="completed")
