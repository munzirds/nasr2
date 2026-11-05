"""
Dynamic Data Collator for Wav2Vec2-BERT 2.0 CTC (`Wav2Vec2BertForCTC`).
=======================================================================
Key differences from standard Wav2Vec2 (`Wav2Vec2ForCTC`):
1. `Wav2Vec2BertProcessor` / `SeamlessM4TFeatureExtractor` produces 2D log-mel filterbank
   features named `input_features` (shape `[seq_len, 160]`) and `attention_mask` (`[seq_len]`),
   NOT 1D raw waveforms named `input_values`.
2. Supports both:
   - On-the-fly audio decoding + feature extraction inside DataLoader workers (from `audio` column), and
   - Pre-extracted `input_features` (when features are pre-cached).
3. Masks padded label positions with `-100` so PyTorch `CTCLoss` ignores padding.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch

from kas_w2v_bert.data import decode_audio_dict


@dataclass
class DataCollatorWav2Vec2BertCTC:
    processor: Any
    padding: Union[bool, str] = True
    sampling_rate: int = 16000
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    audio_column: str = "audio"

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        input_features_list: List[Dict[str, Any]] = []
        label_features_list: List[Dict[str, Any]] = []

        for item in features:
            if "input_features" in item and item["input_features"] is not None:
                feat = item["input_features"]
            else:
                waveform = decode_audio_dict(
                    item[self.audio_column],
                    target_sr=self.sampling_rate,
                    min_duration_sec=self.min_duration_sec,
                    max_duration_sec=self.max_duration_sec,
                )
                extracted = self.processor.feature_extractor(
                    waveform,
                    sampling_rate=self.sampling_rate,
                    return_tensors="np",
                )
                feat = extracted.input_features[0]

            input_features_list.append({"input_features": feat})

            if "labels" in item and item["labels"] is not None:
                label_ids = item["labels"]
            else:
                transcript = item.get("transcript") or item.get("sentence") or ""
                label_ids = self.processor.tokenizer(transcript).input_ids

            # Ensure non-empty label sequence for CTC stability
            if not label_ids:
                label_ids = [self.processor.tokenizer.unk_token_id]

            label_features_list.append({"input_ids": label_ids})

        batch = self.processor.feature_extractor.pad(
            input_features_list,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features_list,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch

