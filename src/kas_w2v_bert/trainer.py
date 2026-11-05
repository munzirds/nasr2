"""
Wav2Vec2-BERT 2.0 CTC Model Builder & Fault-Tolerant Training Orchestrator.
===========================================================================
Handles:
1. `Wav2Vec2BertForCTC` initialization with convolutional adapter, SpecAugment, and `ctc_zero_infinity=True`.
2. Optional freezing of bottom Conformer blocks or full acoustic encoder.
3. Hardware-aware `TrainingArguments` (BF16/FP16/TF32, gradient checkpointing, length-grouped batching).
4. Automatic checkpoint validation & seamless resumption after Colab disconnects.
"""

import inspect
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
)

from kas_w2v_bert.checkpoint_manager import (
    ColabResilienceCallback,
    find_latest_valid_checkpoint,
)
from kas_w2v_bert.collator import DataCollatorWav2Vec2BertCTC
from kas_w2v_bert.config import TrainingConfig
from kas_w2v_bert.data import (
    attach_tokenized_labels_to_split,
    load_and_prepare_kas_news_splits,
)
from kas_w2v_bert.normalizer import compute_metrics
from kas_w2v_bert.vocab import build_or_load_w2v_bert_processor


def build_w2v_bert_ctc_model(
    cfg: TrainingConfig,
    processor: Any,
) -> Wav2Vec2BertForCTC:
    """
    Instantiates `Wav2Vec2BertForCTC` configured for Kashmiri character-level CTC fine-tuning.
    """
    print(f"--> Loading Wav2Vec2BertForCTC from '{cfg.model_name_or_path}'...")
    model = Wav2Vec2BertForCTC.from_pretrained(
        cfg.model_name_or_path,
        attention_dropout=cfg.attention_dropout,
        hidden_dropout=cfg.hidden_dropout,
        feat_proj_dropout=cfg.feat_proj_dropout,
        mask_time_prob=cfg.mask_time_prob,
        mask_time_length=cfg.mask_time_length,
        mask_feature_prob=cfg.mask_feature_prob,
        mask_feature_length=cfg.mask_feature_length,
        layerdrop=cfg.layerdrop,
        ctc_loss_reduction=cfg.ctc_loss_reduction,
        ctc_zero_infinity=cfg.ctc_zero_infinity,
        add_adapter=cfg.add_adapter,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    if cfg.freeze_base_model and hasattr(model, "freeze_base_model"):
        print("--> Freezing entire wav2vec2_bert base encoder (training adapter + CTC head only)...")
        model.freeze_base_model()
    elif cfg.freeze_conformer_layers > 0 and hasattr(model, "wav2vec2_bert"):
        encoder_layers = getattr(model.wav2vec2_bert.encoder, "layers", [])
        n_freeze = min(cfg.freeze_conformer_layers, len(encoder_layers))
        print(f"--> Freezing feature projection and bottom {n_freeze}/{len(encoder_layers)} Conformer layers...")
        if hasattr(model.wav2vec2_bert, "feature_projection"):
            for param in model.wav2vec2_bert.feature_projection.parameters():
                param.requires_grad = False
        for layer in encoder_layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable_params / max(1, total_params)) * 100.0
    print(
        f"[✓] Model initialized: {trainable_params:,} / {total_params:,} trainable parameters ({pct:.1f}%)"
    )
    return model


def make_compute_metrics_fn(processor: Any):
    """Creates the Hugging Face Trainer `compute_metrics` callback for Kashmiri CTC decoding."""

    def compute_eval_metrics(pred) -> Dict[str, float]:
        pred_logits = pred.predictions
        if isinstance(pred_logits, tuple):
            pred_logits = pred_logits[0]
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        m = compute_metrics(label_str, pred_str)
        return {
            "da_wer": m["da_wer"],
            "dg_wer": m["dg_wer"],
            "cer": m["cer"],
            "dg_cer": m["dg_cer"],
            "diacritic_gap": m["diacritic_degradation_gap"],
        }

    return compute_eval_metrics


def create_training_arguments(cfg: TrainingConfig) -> TrainingArguments:
    """Builds `TrainingArguments` compatible across `transformers` versions."""
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    kwargs: Dict[str, Any] = {
        "output_dir": cfg.checkpoints_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "max_grad_norm": cfg.max_grad_norm,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "bf16": bool(cfg.bf16),
        "fp16": bool(cfg.fp16),
        "tf32": bool(cfg.tf32) if torch.cuda.is_available() else False,
        eval_key: cfg.eval_strategy,
        "eval_steps": cfg.eval_steps if cfg.eval_strategy == "steps" else None,
        "save_strategy": cfg.save_strategy,
        "save_steps": cfg.save_steps if cfg.save_strategy == "steps" else None,
        "logging_dir": os.path.join(cfg.logs_dir, "tensorboard"),
        "logging_steps": cfg.logging_steps,
        "save_total_limit": cfg.save_total_limit,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": cfg.metric_for_best_model,
        "greater_is_better": False,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "group_by_length": cfg.group_by_length,
        "length_column_name": "input_length",
        "remove_unused_columns": False,  # Required so collator receives raw `audio` and `labels`
        "report_to": ["tensorboard"],
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "push_to_hub": cfg.push_to_hub,
        "hub_model_id": cfg.hub_model_id,
        "hub_private_repo": cfg.hub_private_repo,
        "hub_strategy": "checkpoint" if cfg.push_to_hub else "every_save",
    }

    if cfg.warmup_steps and cfg.warmup_steps > 0:
        kwargs["warmup_steps"] = cfg.warmup_steps
    else:
        kwargs["warmup_ratio"] = cfg.warmup_ratio

    if "gradient_checkpointing_kwargs" in ta_params and cfg.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def run_training_pipeline(cfg: TrainingConfig) -> Tuple[str, Dict[str, Any]]:
    """
    End-to-end fault-tolerant training pipeline:
    1. Resolves GPU capabilities & ensures persistent directory layout on Google Drive.
    2. Loads `munzirahangar/kas-news-dataset` & deterministic train/val/test split indices.
    3. Builds or loads persisted Kashmiri `Wav2Vec2BertProcessor`.
    4. Prepares tokenized splits & initializes `Wav2Vec2BertForCTC`.
    5. Detects any valid existing checkpoint in `<output_dir>/checkpoints` and resumes automatically.
    6. Exports the final best model + processor to `<output_dir>/best_model`.
    """
    hw_info = cfg.resolve_hardware_defaults()
    cfg.ensure_dirs()
    cfg.save(os.path.join(cfg.artifacts_dir, "effective_config.yaml"))

    print("=" * 72)
    print("  KASHMIRI WAV2VEC2-BERT 2.0 CTC TRAINING PIPELINE (COLAB PRO READY)")
    print("=" * 72)
    print(f"  GPU Device:           {hw_info['gpu_name']} ({hw_info['vram_gb']} GB VRAM)")
    print(f"  Precision:            BF16={cfg.bf16} | FP16={cfg.fp16} | TF32={cfg.tf32}")
    print(
        f"  Batch Size:           {cfg.per_device_train_batch_size} per GPU x "
        f"{cfg.gradient_accumulation_steps} grad_accum = "
        f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} effective"
    )
    print(f"  Persistent Dir:       {cfg.output_dir}")
    print("=" * 72)

    # 1. Load dataset & deterministic splits
    raw_ds, splits, train_texts = load_and_prepare_kas_news_splits(cfg)

    # 2. Build or load cached Wav2Vec2BertProcessor
    processor = build_or_load_w2v_bert_processor(
        model_name_or_path=cfg.model_name_or_path,
        artifacts_dir=cfg.artifacts_dir,
        training_texts=train_texts,
        remove_diacritics=cfg.remove_diacritics,
        strip_punctuation=cfg.strip_punctuation,
    )

    # 3. Prepare Train & Validation splits (fast metadata/label attachment; audio decoded on-the-fly)
    train_ds = attach_tokenized_labels_to_split(
        raw_ds=raw_ds,
        indices=splits["train"],
        processor=processor,
        cfg=cfg,
        max_samples=cfg.max_train_samples,
        split_name="train",
    )
    val_ds = attach_tokenized_labels_to_split(
        raw_ds=raw_ds,
        indices=splits["validation"],
        processor=processor,
        cfg=cfg,
        max_samples=cfg.max_eval_samples,
        split_name="validation",
    )

    # 4. Initialize Model & Collator
    model = build_w2v_bert_ctc_model(cfg, processor)
    collator = DataCollatorWav2Vec2BertCTC(
        processor=processor,
        padding=True,
        sampling_rate=cfg.sampling_rate,
        min_duration_sec=cfg.min_duration_sec,
        max_duration_sec=cfg.max_duration_sec,
        audio_column=cfg.audio_column,
    )

    # 5. Setup TrainingArguments & Callbacks
    training_args = create_training_arguments(cfg)
    resilience_cb = ColabResilienceCallback(
        output_dir=cfg.output_dir,
        logs_dir=cfg.logs_dir,
        best_model_dir=cfg.best_model_dir,
        processor=processor,
    )
    callbacks = [resilience_cb]
    if cfg.early_stopping_patience and cfg.early_stopping_patience > 0 and cfg.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    proc_kwarg = (
        {"processing_class": processor}
        if "processing_class" in trainer_init_params
        else {"tokenizer": processor.feature_extractor}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics_fn(processor),
        callbacks=callbacks,
        **proc_kwarg,
    )

    # 6. Check for existing valid checkpoint to auto-resume after Colab disconnect
    resume_ckpt = None
    if cfg.auto_resume:
        resume_ckpt = find_latest_valid_checkpoint(
            cfg.checkpoints_dir,
            require_optimizer=True,
            quarantine_corrupted=True,
        )
        if resume_ckpt:
            print(f"\n[↻] Found valid checkpoint! Auto-resuming training from: {resume_ckpt}")
        else:
            print("\n--> No existing checkpoint found. Starting fresh training run...")

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted (KeyboardInterrupt / Colab Stop).")
        emergency_dir = os.path.join(cfg.checkpoints_dir, f"checkpoint-{trainer.state.global_step}")
        print(f"--> Saving emergency checkpoint to: {emergency_dir}")
        trainer.save_model(emergency_dir)
        trainer.state.save_to_json(os.path.join(emergency_dir, "trainer_state.json"))
        processor.save_pretrained(emergency_dir)
        resilience_cb._update_status(trainer.state, status="interrupted")
        raise

    # 7. Save final best model + processor to <output_dir>/best_model
    print(f"\n--> Saving best fine-tuned model & processor to: {cfg.best_model_dir}")
    trainer.save_model(cfg.best_model_dir)
    processor.save_pretrained(cfg.best_model_dir)
    trainer.save_metrics("train", train_result.metrics)

    print("\n--> Running final validation evaluation on best model...")
    final_eval_metrics = trainer.evaluate()
    trainer.save_metrics("eval", final_eval_metrics)
    print(
        f"[✓] Final Validation Metrics: "
        f"DA-WER={final_eval_metrics.get('eval_da_wer')}% | "
        f"DG-WER={final_eval_metrics.get('eval_dg_wer')}% | "
        f"CER={final_eval_metrics.get('eval_cer')}%"
    )
    return cfg.best_model_dir, final_eval_metrics

