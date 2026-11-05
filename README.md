# Kashmiri Wav2Vec2-BERT 2.0 ASR (`kas-w2v-bert`)

Fault-tolerant fine-tuning and evaluation suite for **Meta Wav2Vec2-BERT 2.0 (`facebook/w2v-bert-2.0`)** on **`munzirahangar/kas-news-dataset`** (22,425 Kashmiri broadcast news utterances, 16 kHz), engineered specifically for **Google Colab Pro (A100 / L4 GPUs)** with seamless recovery from runtime disconnections, timeouts, and preemptions.

---

## Key Features & Colab Resilience

1. **Native `Wav2Vec2BertForCTC` Support**:
   - Constructs `Wav2Vec2BertProcessor` by pairing `SeamlessM4TFeatureExtractor` (`80`-bin log-mel filterbanks with `stride=2` -> `160`-dim acoustic frames at `50 Hz`) with a custom Kashmiri `Wav2Vec2CTCTokenizer` built from `munzirahangar/kas-news-dataset`.
   - Uses `DataCollatorWav2Vec2BertCTC` to dynamically pad 2D `input_features` and `attention_mask`, enables the convolutional adapter (`add_adapter=True`), and sets `ctc_zero_infinity=True` to prevent gradient explosions on outlier clips.
2. **Instant Startup on Colab Restarts (No 30-Minute Audio Map Penalty)**:
   - Scans only the Parquet `sentence` text column (`<3 seconds`) to normalize transcripts, build `vocab.json`, and generate deterministic `train` (90%), `validation` (5%), and `test` (5%) index splits (`<output_dir>/artifacts/splits.json`).
   - Decodes 16 kHz audio (`soundfile`) and computes `SeamlessM4TFeatureExtractor` features **on-the-fly** inside parallel DataLoader workers while grouping batches by transcript token length (`group_by_length=True`).
3. **Fault-Tolerant Checkpointing & Auto-Resume**:
   - Saves checkpoints every `200` steps (`~10–15 minutes`) directly to Google Drive (`<output_dir>/checkpoints/checkpoint-XXXX`).
   - **Corrupted Checkpoint Quarantine**: If Colab disconnects *while* writing a checkpoint to Google Drive, `find_latest_valid_checkpoint()` automatically detects and quarantines the half-written checkpoint folder and resumes from the previous intact checkpoint.
   - **Continuous Best-Model Export**: Whenever validation `DA-WER` reaches a new best, a standalone copy of the model and processor is saved immediately to `<output_dir>/best_model`.
4. **Resumable Chunked Evaluation**:
   - `scripts/run_eval.py` streams batch predictions to `<output_dir>/evaluation/eval_predictions_<split>.jsonl` with `os.fsync()` after every batch.
   - If Colab disconnects mid-evaluation, re-running `scripts/run_eval.py` skips all already-completed `sample_idx` entries, finishes the remaining utterances, and writes `eval_summary_<split>.json` and `eval_report_<split>.md`.
5. **Dual Kashmiri Orthographic Metrics**:
   - Reports **DA-WER** (Diacritic-Aware WER), **DG-WER** (Diacritic-Agnostic WER), **CER**, **DG-CER**, and **DDG** (Diacritic Degradation Gap) using full Perso-Arabic Kashmiri Unicode canonicalization.

---

## Repository Structure

```text
nasr2/
├── configs/
│   └── w2v_bert_kas_news.yaml           # Default hyperparameters & Colab Pro paths
├── notebooks/
│   └── w2v_bert_kashmiri_colab_pro.ipynb # Turnkey Colab Pro notebook
├── scripts/
│   ├── colab_setup.sh                   # One-command Colab environment setup
│   ├── run_train.py                     # CLI training launcher (auto-resumes)
│   └── run_eval.py                      # CLI evaluation launcher (resumable JSONL)
├── src/
│   └── kas_w2v_bert/
│       ├── config.py                    # Dataclass config, YAML/CLI parser & GPU auto-tuner
│       ├── normalizer.py                # Kashmiri Unicode canonicalizer & DA/DG WER/CER metrics
│       ├── vocab.py                     # Fast text-only CTC vocab & Wav2Vec2BertProcessor builder
│       ├── data.py                      # HF dataset loader & deterministic split manager
│       ├── collator.py                  # 2D input_features + attention_mask CTC data collator
│       ├── checkpoint_manager.py        # Checkpoint validator, quarantine & Drive callbacks
│       ├── trainer.py                   # Wav2Vec2BertForCTC builder & Trainer orchestrator
│       └── evaluator.py                 # Resumable batch evaluator & Markdown report generator
└── tests/
    └── test_pipeline.py                 # Unit tests for normalizer, splits, checkpoints & eval
```

---

## Quickstart in Google Colab Pro

Open [`notebooks/w2v_bert_kashmiri_colab_pro.ipynb`](notebooks/w2v_bert_kashmiri_colab_pro.ipynb) in Google Colab (**Runtime -> Change runtime type -> A100 GPU or L4 GPU**), or run these commands in a Colab cell:

### 1. Mount Google Drive, Clone Repo & Install Dependencies
```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/<your-username>/nasr2.git /content/nasr2
%cd /content/nasr2
!bash scripts/colab_setup.sh
```

### 2. Run or Resume Fine-Tuning
Re-run this exact command at any time after a Colab disconnect or restart — it automatically validates checkpoints in `/content/drive/MyDrive/nasr2_kashmiri_w2v_bert/checkpoints` and resumes from the latest intact step:
```bash
python scripts/run_train.py \
    --config configs/w2v_bert_kas_news.yaml \
    --output_dir /content/drive/MyDrive/nasr2_kashmiri_w2v_bert \
    --save_steps 200 \
    --eval_steps 200
```

### 3. Run or Resume Evaluation on the Held-Out Test Split
```bash
python scripts/run_eval.py \
    --config configs/w2v_bert_kas_news.yaml \
    --output_dir /content/drive/MyDrive/nasr2_kashmiri_w2v_bert \
    --eval_split test
```

---

## Persistent Google Drive Output Layout

All state is stored inside `--output_dir` (e.g., `/content/drive/MyDrive/nasr2_kashmiri_w2v_bert`):
- `artifacts/splits.json` — Deterministic `train` / `validation` / `test` utterance indices.
- `artifacts/processor/` — Saved `Wav2Vec2BertProcessor` (`vocab.json`, `preprocessor_config.json`, `tokenizer_config.json`).
- `artifacts/effective_config.yaml` — Exact resolved hyperparameters and GPU batch settings.
- `checkpoints/checkpoint-XXXX/` — Rolling Trainer checkpoints (model, optimizer, scheduler, RNG state).
- `best_model/` — Standalone best model + processor (exported immediately whenever validation `DA-WER` improves).
- `logs/training_metrics.jsonl` & `run_status.json` — Live `fsync`-flushed step/eval metrics and heartbeat status.
- `evaluation/eval_predictions_test.jsonl`, `eval_summary_test.json`, `eval_report_test.md` — Resumable per-sample predictions, corpus metrics, duration-bucket breakdown, and qualitative error analysis.

