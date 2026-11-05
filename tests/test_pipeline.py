"""
Unit Tests for Kashmiri Wav2Vec2-BERT 2.0 Pipeline (`kas_w2v_bert`).
====================================================================
Tests:
1. Kashmiri Perso-Arabic Unicode canonicalization, DA vs DG normalization, and ASR metrics.
2. Deterministic CTC vocabulary construction (`vocab.json`).
3. Deterministic train/val/test split persistence (`splits.json`).
4. Corrupted checkpoint detection & automatic quarantine (`find_latest_valid_checkpoint`).
5. Resumable evaluation JSONL loading & recovery from truncated lines (`load_completed_predictions`).
"""

import json
import os
import sys
import tempfile
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kas_w2v_bert.config import TrainingConfig, load_config_from_cli
from kas_w2v_bert.normalizer import (
    canonicalize_unicode,
    compute_metrics,
    normalize_kashmiri,
)
from kas_w2v_bert.vocab import (
    build_kashmiri_ctc_vocab,
    extract_characters_from_texts,
)


class TestKashmiriNormalizerAndMetrics(unittest.TestCase):
    def test_unicode_canonicalization(self):
        # Arabic Kaf (\u0643) -> Keheh (\u06a9), Arabic Yeh (\u064a) -> Farsi Yeh (\u06cc)
        raw = "\u0643\u0627\u0634\u0631 \u0632\u0628\u0627\u0646\u064a"
        norm = canonicalize_unicode(raw)
        self.assertNotIn("\u0643", norm)
        self.assertNotIn("\u064a", norm)
        self.assertIn("\u06a9", norm)
        self.assertIn("\u06cc", norm)

    def test_da_vs_dg_normalization(self):
        # Text with Kashmiri central vowel (\u0672 = ٲ) and combining mark (\u065f) + punctuation
        sample = "کٲشُر، زَبان!"
        da = normalize_kashmiri(sample, remove_diacritics=False, strip_punctuation=True)
        dg = normalize_kashmiri(sample, remove_diacritics=True, strip_punctuation=True)

        self.assertNotIn("،", da)
        self.assertNotIn("!", da)
        self.assertIn("\u0672", da)  # Preserves ٲ in DA mode
        self.assertIn("\u064f", da)  # Preserves damma in DA mode

        self.assertNotIn("\u0672", dg)  # Mapped to ا in DG mode
        self.assertNotIn("\u064f", dg)  # Stripped in DG mode
        self.assertEqual(dg, "کاشر زبان")

    def test_compute_metrics_da_and_dg(self):
        refs = ["کٲشُر زَبان"]
        hyps = ["کاشر زبان"]  # Base consonants match, diacritics missing
        m = compute_metrics(refs, hyps)
        self.assertEqual(m["dg_wer"], 0.0)
        self.assertEqual(m["dg_cer"], 0.0)
        self.assertGreater(m["da_wer"], 0.0)
        self.assertEqual(m["diacritic_degradation_gap"], m["da_wer"] - m["dg_wer"])


class TestVocabAndSplits(unittest.TestCase):
    def test_vocab_builder_deterministic(self):
        texts = ["کٲشُر زَبان", "خَبَر نامہٕ"]
        chars = extract_characters_from_texts(texts, remove_diacritics=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.json")
            vocab = build_kashmiri_ctc_vocab(chars, output_path=vocab_path)
            self.assertEqual(vocab["[PAD]"], 0)
            self.assertEqual(vocab["[UNK]"], 1)
            self.assertEqual(vocab["|"], 4)
            self.assertNotIn(" ", vocab)
            self.assertTrue(os.path.isfile(vocab_path))

    def test_deterministic_splits_persistence(self):
        from kas_w2v_bert.data import compute_or_load_splits, resolve_dataset_columns

        audio_c, text_c = resolve_dataset_columns(
            ["audio_filepath", "normalized_text", "speaker_id"],
            preferred_audio="audio",
            preferred_text="sentence",
        )
        self.assertEqual(audio_c, "audio_filepath")
        self.assertEqual(text_c, "normalized_text")

        indices = list(range(100))
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = compute_or_load_splits(
                indices, tmpdir, val_ratio=0.1, test_ratio=0.1, seed=42, dataset_name="user/ds1"
            )
            s2 = compute_or_load_splits(
                indices, tmpdir, val_ratio=0.1, test_ratio=0.1, seed=42, dataset_name="user/ds1"
            )
            self.assertEqual(s1["train"], s2["train"])
            self.assertEqual(s1["validation"], s2["validation"])
            self.assertEqual(s1["test"], s2["test"])
            self.assertEqual(len(s1["train"]) + len(s1["validation"]) + len(s1["test"]), 100)
            # Ensure zero overlap between train, val, and test
            self.assertEqual(len(set(s1["train"]) & set(s1["validation"])), 0)
            self.assertEqual(len(set(s1["train"]) & set(s1["test"])), 0)
            self.assertEqual(len(set(s1["validation"]) & set(s1["test"])), 0)

    def test_discover_drive_shard_dirs(self):
        from kas_w2v_bert.data import discover_drive_shard_dirs

        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = os.path.join(tmpdir, "voicehub_dataset", "shards")
            b0 = os.path.join(shards_dir, "batch_000")
            b1 = os.path.join(shards_dir, "batch_001")
            b_incomplete = os.path.join(shards_dir, "batch_002")
            os.makedirs(b0)
            os.makedirs(b1)
            os.makedirs(b_incomplete)  # No state.json inside batch_002!

            with open(os.path.join(b0, "state.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(b1, "state.json"), "w") as f:
                f.write("{}")

            found = discover_drive_shard_dirs(os.path.join(tmpdir, "voicehub_dataset"))
            self.assertEqual(found, [b0, b1])


class TestCheckpointQuarantineAndResumableEval(unittest.TestCase):
    def test_quarantine_corrupted_checkpoint(self):
        # Import only file-validation functions without requiring torch
        from kas_w2v_bert.checkpoint_manager import (
            find_latest_valid_checkpoint,
            is_valid_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create a valid checkpoint-200
            ckpt_200 = os.path.join(tmpdir, "checkpoint-200")
            os.makedirs(ckpt_200)
            with open(os.path.join(ckpt_200, "trainer_state.json"), "w") as f:
                json.dump({"global_step": 200}, f)
            with open(os.path.join(ckpt_200, "config.json"), "w") as f:
                json.dump({"model_type": "wav2vec2-bert"}, f)
            with open(os.path.join(ckpt_200, "model.safetensors"), "wb") as f:
                f.write(b"valid_weights_payload")
            with open(os.path.join(ckpt_200, "optimizer.pt"), "wb") as f:
                f.write(b"valid_opt")
            with open(os.path.join(ckpt_200, "scheduler.pt"), "wb") as f:
                f.write(b"valid_sched")

            # 2. Create an incomplete/corrupted checkpoint-400 (simulating mid-save Colab disconnect)
            ckpt_400 = os.path.join(tmpdir, "checkpoint-400")
            os.makedirs(ckpt_400)
            with open(os.path.join(ckpt_400, "config.json"), "w") as f:
                json.dump({"model_type": "wav2vec2-bert"}, f)
            # Missing trainer_state.json and optimizer.pt!

            self.assertTrue(is_valid_checkpoint(ckpt_200, require_optimizer=True))
            self.assertFalse(is_valid_checkpoint(ckpt_400, require_optimizer=True))

            resolved = find_latest_valid_checkpoint(
                tmpdir, require_optimizer=True, quarantine_corrupted=True
            )
            self.assertEqual(resolved, ckpt_200)
            self.assertFalse(os.path.exists(ckpt_400))  # Corrupted checkpoint was quarantined!

    def test_resumable_eval_jsonl_recovery(self):
        from kas_w2v_bert.evaluator import (
            generate_markdown_report,
            load_completed_predictions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "eval_predictions_test.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sample_idx": 10,
                            "duration_sec": 8.5,
                            "reference": "کٲشُر زَبان",
                            "prediction": "کٲشُر زَبان",
                            "da_wer": 0.0,
                            "dg_wer": 0.0,
                            "cer": 0.0,
                            "dg_cer": 0.0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # Simulate a truncated line at EOF caused by an abrupt Colab disconnect
                f.write('{"sample_idx": 11, "duration_sec": 12.1, "reference": "خَبَر')

            completed = load_completed_predictions(jsonl_path)
            self.assertEqual(list(completed.keys()), [10])
            self.assertEqual(completed[10]["da_wer"], 0.0)

            report_md = os.path.join(tmpdir, "eval_report_test.md")
            generate_markdown_report(
                summary={
                    "split": "test",
                    "model_path": "/tmp/best_model",
                    "dataset_name": "munzirahangar/kas-news-dataset",
                    "num_samples": 1,
                    "total_audio_hours": 0.0024,
                    "evaluated_at_utc": "2026-11-05T00:00:00Z",
                    "metrics": {
                        "da_wer": 0.0,
                        "dg_wer": 0.0,
                        "cer": 0.0,
                        "dg_cer": 0.0,
                        "diacritic_degradation_gap": 0.0,
                    },
                    "duration_buckets": {
                        "medium (5-15s)": {
                            "count": 1,
                            "da_wer": 0.0,
                            "dg_wer": 0.0,
                            "cer": 0.0,
                            "dg_cer": 0.0,
                        }
                    },
                },
                records=list(completed.values()),
                report_path=report_md,
            )
            self.assertTrue(os.path.isfile(report_md))

    def test_cli_config_override(self):
        cfg = load_config_from_cli(
            [
                "--output_dir",
                "/tmp/test_out",
                "--learning_rate",
                "1e-4",
                "--save_steps",
                "150",
                "--remove_diacritics",
                "true",
            ]
        )
        self.assertEqual(cfg.output_dir, "/tmp/test_out")
        self.assertAlmostEqual(cfg.learning_rate, 1e-4)
        self.assertEqual(cfg.save_steps, 150)
        self.assertTrue(cfg.remove_diacritics)


if __name__ == "__main__":
    unittest.main()
