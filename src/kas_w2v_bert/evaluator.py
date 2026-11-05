"""
Resumable, Fault-Tolerant Evaluator & Error Analysis for Kashmiri Wav2Vec2-BERT 2.0.
====================================================================================
Designed to survive Colab disconnections during evaluation:
1. Automatically locates the best model (`<output_dir>/best_model` or latest valid checkpoint).
2. Streams batch predictions incrementally to `<output_dir>/evaluation/eval_predictions_<split>.jsonl`
   with `os.fsync` after every batch.
3. If interrupted and restarted, reads already-completed `sample_idx` records from the JSONL file
   and only runs inference on the remaining samples.
4. Generates both machine-readable `eval_summary_<split>.json` and a rich Markdown report
   `eval_report_<split>.md` (overall DA-WER/DG-WER/CER/DDG, duration-bucket breakdown, and sample analysis).
"""

from datetime import datetime, timezone
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

try:
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
except ImportError:
    np = None  # type: ignore
    torch = None  # type: ignore
    tqdm = None  # type: ignore
    Wav2Vec2BertForCTC = None  # type: ignore
    Wav2Vec2BertProcessor = None  # type: ignore

from kas_w2v_bert.checkpoint_manager import (
    atomic_write_json,
    resolve_best_model_or_checkpoint,
)
from kas_w2v_bert.config import TrainingConfig
from kas_w2v_bert.data import decode_audio_dict, load_and_prepare_kas_news_splits
from kas_w2v_bert.normalizer import compute_metrics, normalize_kashmiri


def load_completed_predictions(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Reads an existing `eval_predictions_<split>.jsonl` file and returns a mapping
    `{sample_idx: record_dict}` for all valid completed lines.
    Silently ignores any truncated trailing line caused by an abrupt disconnect.
    """
    completed: Dict[int, Dict[str, Any]] = {}
    if not os.path.isfile(jsonl_path):
        return completed

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "sample_idx" in rec and "prediction" in rec and "reference" in rec:
                    completed[int(rec["sample_idx"])] = rec
            except Exception:
                # Ignore partial trailing line from abrupt power-off
                continue
    return completed


def _compute_duration_buckets(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    buckets = {
        "short (<5s)": [r for r in records if r.get("duration_sec", 0.0) < 5.0],
        "medium (5-15s)": [r for r in records if 5.0 <= r.get("duration_sec", 0.0) <= 15.0],
        "long (>15s)": [r for r in records if r.get("duration_sec", 0.0) > 15.0],
    }
    summary: Dict[str, Dict[str, Any]] = {}
    for name, items in buckets.items():
        if not items:
            continue
        refs = [r["reference"] for r in items]
        hyps = [r["prediction"] for r in items]
        m = compute_metrics(refs, hyps)
        summary[name] = {
            "count": len(items),
            "da_wer": m["da_wer"],
            "dg_wer": m["dg_wer"],
            "cer": m["cer"],
            "dg_cer": m["dg_cer"],
        }
    return summary


def generate_markdown_report(
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    report_path: str,
) -> None:
    """Writes a structured Markdown evaluation report with qualitative sample analysis."""
    metrics = summary["metrics"]
    sorted_by_cer = sorted(records, key=lambda r: r.get("cer", 0.0))
    best_samples = sorted_by_cer[:5]
    worst_samples = list(reversed(sorted_by_cer[-10:]))

    lines = [
        f"# Kashmiri Wav2Vec2-BERT 2.0 Evaluation Report (`{summary['split']}` split)",
        "",
        f"- **Model Checkpoint**: `{summary['model_path']}`",
        f"- **Dataset**: `{summary['dataset_name']}`",
        f"- **Evaluated Utterances**: `{summary['num_samples']}`",
        f"- **Total Audio Duration**: `{summary['total_audio_hours']:.2f} hours`",
        f"- **Evaluated At (UTC)**: `{summary['evaluated_at_utc']}`",
        "",
        "## 1. Corpus-Level ASR Metrics",
        "",
        "| Metric | Value (%) | Description |",
        "| :--- | :---: | :--- |",
        f"| **DA-WER** | **{metrics['da_wer']:.2f}%** | Diacritic-Aware Word Error Rate |",
        f"| **DG-WER** | **{metrics['dg_wer']:.2f}%** | Diacritic-Agnostic Word Error Rate |",
        f"| **CER** | **{metrics['cer']:.2f}%** | Diacritic-Aware Character Error Rate |",
        f"| **DG-CER** | **{metrics['dg_cer']:.2f}%** | Diacritic-Agnostic Character Error Rate |",
        f"| **DDG** | **{metrics['diacritic_degradation_gap']:.2f}%** | Diacritic Degradation Gap (`DA-WER - DG-WER`) |",
        "",
        "## 2. Performance by Audio Duration Bucket",
        "",
        "| Duration Bucket | Utterances | DA-WER (%) | DG-WER (%) | CER (%) | DG-CER (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for b_name, b_stats in summary.get("duration_buckets", {}).items():
        lines.append(
            f"| {b_name} | {b_stats['count']} | {b_stats['da_wer']:.2f}% | "
            f"{b_stats['dg_wer']:.2f}% | {b_stats['cer']:.2f}% | {b_stats['dg_cer']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## 3. Top 5 Best Transcriptions",
            "",
        ]
    )
    for idx, r in enumerate(best_samples, 1):
        lines.extend(
            [
                f"### {idx}. Sample `#{r['sample_idx']}` (Duration: `{r.get('duration_sec', 0.0):.1f}s` | DA-WER: `{r['da_wer']:.2f}%` | CER: `{r['cer']:.2f}%`)",
                f"- **Reference**: `{r['reference']}`",
                f"- **Prediction**: `{r['prediction']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Top 10 Highest-Error Transcriptions (Error Analysis)",
            "",
        ]
    )
    for idx, r in enumerate(worst_samples, 1):
        lines.extend(
            [
                f"### {idx}. Sample `#{r['sample_idx']}` (Duration: `{r.get('duration_sec', 0.0):.1f}s` | DA-WER: `{r['da_wer']:.2f}%` | CER: `{r['cer']:.2f}%`)",
                f"- **Reference**: `{r['reference']}`",
                f"- **Prediction**: `{r['prediction']}`",
                "",
            ]
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_resumable_evaluation(
    cfg: TrainingConfig,
    split: str = "test",
    checkpoint_path: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Runs fault-tolerant, resumable evaluation on `split` ('test' or 'validation').
    """
    hw_info = cfg.resolve_hardware_defaults()
    cfg.ensure_dirs()

    model_dir = checkpoint_path or resolve_best_model_or_checkpoint(cfg.output_dir)
    processor_dir = (
        model_dir
        if os.path.isfile(os.path.join(model_dir, "vocab.json"))
        else os.path.join(cfg.artifacts_dir, "processor")
    )

    print("=" * 72)
    print(f"  KASHMIRI WAV2VEC2-BERT 2.0 RESUMABLE EVALUATION (SPLIT='{split}')")
    print("=" * 72)
    print(f"  Model Checkpoint:     {model_dir}")
    print(f"  Processor Path:       {processor_dir}")
    print(f"  GPU Device:           {hw_info['gpu_name']} ({hw_info['vram_gb']} GB VRAM)")
    print("=" * 72)

    processor = Wav2Vec2BertProcessor.from_pretrained(processor_dir)
    model = Wav2Vec2BertForCTC.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg.bf16 and device.type == "cuda")
    use_fp16 = bool(cfg.fp16 and device.type == "cuda" and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    model.to(device)
    model.eval()

    raw_ds, splits, _ = load_and_prepare_kas_news_splits(cfg)
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Available splits: {list(splits.keys())}")

    target_indices = splits[split]
    if cfg.max_eval_samples and cfg.max_eval_samples > 0:
        target_indices = target_indices[: cfg.max_eval_samples]

    predictions_jsonl = os.path.join(cfg.eval_dir, f"eval_predictions_{split}.jsonl")
    summary_json = os.path.join(cfg.eval_dir, f"eval_summary_{split}.json")
    report_md = os.path.join(cfg.eval_dir, f"eval_report_{split}.md")

    if force_recompute and os.path.exists(predictions_jsonl):
        os.remove(predictions_jsonl)

    completed_map = load_completed_predictions(predictions_jsonl)
    remaining_indices = [idx for idx in target_indices if idx not in completed_map]

    if completed_map and remaining_indices:
        print(
            f"[↻] Resuming interrupted evaluation: {len(completed_map)}/{len(target_indices)} "
            f"already completed, {len(remaining_indices)} remaining."
        )
    elif not remaining_indices:
        print(
            f"[✓] All {len(target_indices)} utterances in '{split}' already evaluated in {predictions_jsonl}."
        )

    batch_size = cfg.per_device_eval_batch_size or 8
    start_time = time.time()

    if remaining_indices:
        sub_ds = raw_ds.select(remaining_indices)
        with open(predictions_jsonl, "a", encoding="utf-8") as out_f:
            for batch_start in tqdm(
                range(0, len(remaining_indices), batch_size),
                desc=f"Evaluating ({split})",
            ):
                batch_end = min(batch_start + batch_size, len(remaining_indices))
                batch_indices = remaining_indices[batch_start:batch_end]
                batch_rows = [sub_ds[i] for i in range(batch_start, batch_end)]

                input_features_list = []
                durations_sec = []
                references = []

                for row in batch_rows:
                    waveform = decode_audio_dict(
                        row[cfg.audio_column],
                        target_sr=cfg.sampling_rate,
                        min_duration_sec=cfg.min_duration_sec,
                        max_duration_sec=cfg.max_duration_sec,
                    )
                    durations_sec.append(round(waveform.shape[0] / float(cfg.sampling_rate), 2))
                    extracted = processor.feature_extractor(
                        waveform,
                        sampling_rate=cfg.sampling_rate,
                        return_tensors="np",
                    )
                    input_features_list.append({"input_features": extracted.input_features[0]})

                    raw_ref = row.get(cfg.text_column) or ""
                    references.append(
                        normalize_kashmiri(
                            raw_ref,
                            remove_diacritics=cfg.remove_diacritics,
                            strip_punctuation=cfg.strip_punctuation,
                        )
                    )

                padded = processor.feature_extractor.pad(
                    input_features_list,
                    padding=True,
                    return_tensors="pt",
                )
                input_features = padded["input_features"].to(device)
                attention_mask = padded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    if device.type == "cuda" and amp_dtype != torch.float32:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            logits = model(
                                input_features=input_features,
                                attention_mask=attention_mask,
                            ).logits
                    else:
                        logits = model(
                            input_features=input_features,
                            attention_mask=attention_mask,
                        ).logits

                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                raw_preds = processor.batch_decode(pred_ids)

                for s_idx, ref_str, raw_pred_str, dur in zip(
                    batch_indices, references, raw_preds, durations_sec
                ):
                    norm_pred = normalize_kashmiri(
                        raw_pred_str,
                        remove_diacritics=cfg.remove_diacritics,
                        strip_punctuation=cfg.strip_punctuation,
                    )
                    item_m = compute_metrics([ref_str], [norm_pred])
                    rec = {
                        "sample_idx": int(s_idx),
                        "duration_sec": dur,
                        "reference": ref_str,
                        "prediction": norm_pred,
                        "da_wer": item_m["da_wer"],
                        "dg_wer": item_m["dg_wer"],
                        "cer": item_m["cer"],
                        "dg_cer": item_m["dg_cer"],
                    }
                    completed_map[int(s_idx)] = rec
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Flush to Google Drive after every single batch so progress is never lost
                out_f.flush()
                os.fsync(out_f.fileno())

    ordered_records = [completed_map[idx] for idx in target_indices if idx in completed_map]
    all_refs = [r["reference"] for r in ordered_records]
    all_hyps = [r["prediction"] for r in ordered_records]

    corpus_metrics = compute_metrics(all_refs, all_hyps)
    total_audio_sec = sum(r.get("duration_sec", 0.0) for r in ordered_records)
    elapsed_sec = max(0.01, time.time() - start_time)

    summary_payload = {
        "split": split,
        "model_path": model_dir,
        "dataset_name": cfg.dataset_name,
        "num_samples": len(ordered_records),
        "total_audio_seconds": round(total_audio_sec, 2),
        "total_audio_hours": round(total_audio_sec / 3600.0, 4),
        "wall_time_seconds": round(elapsed_sec, 2),
        "metrics": corpus_metrics,
        "duration_buckets": _compute_duration_buckets(ordered_records),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    atomic_write_json(summary_json, summary_payload)
    generate_markdown_report(summary_payload, ordered_records, report_md)

    print("\n" + "=" * 72)
    print(f"  EVALUATION SUMMARY ({split.upper()} SPLIT, N={len(ordered_records)})")
    print("=" * 72)
    print(f"  DA-WER (Diacritic-Aware WER):      {corpus_metrics['da_wer']:.2f}%")
    print(f"  DG-WER (Diacritic-Agnostic WER):   {corpus_metrics['dg_wer']:.2f}%")
    print(f"  CER    (Diacritic-Aware CER):      {corpus_metrics['cer']:.2f}%")
    print(f"  DG-CER (Diacritic-Agnostic CER):   {corpus_metrics['dg_cer']:.2f}%")
    print(f"  DDG    (Diacritic Degradation):    {corpus_metrics['diacritic_degradation_gap']:.2f}%")
    print("-" * 72)
    print(f"  Predictions JSONL: {predictions_jsonl}")
    print(f"  Summary JSON:      {summary_json}")
    print(f"  Markdown Report:   {report_md}")
    print("=" * 72)

    return summary_payload
