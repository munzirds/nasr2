"""
CTC Vocabulary & Wav2Vec2BertProcessor Builder for Kashmiri ASR.
================================================================
Note: `facebook/w2v-bert-2.0` is a self-supervised Conformer model that ships with a
`SeamlessM4TFeatureExtractor` (`preprocessor_config.json`) but NO tokenizer.
Calling `AutoProcessor.from_pretrained("facebook/w2v-bert-2.0")` directly will fail unless a
`Wav2Vec2CTCTokenizer` is first constructed from `vocab.json` and wrapped in `Wav2Vec2BertProcessor`.

This module:
1. Extracts unique normalized Kashmiri characters from training transcripts (without decoding audio).
2. Saves a deterministic `vocab.json` and full `Wav2Vec2BertProcessor` to `<output_dir>/artifacts/processor`.
3. Automatically reuses the persisted processor on Colab restarts so token IDs remain 100% identical.
"""

import json
import os
from typing import Dict, Iterable, Set

from kas_w2v_bert.normalizer import normalize_kashmiri

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
WORD_DELIMITER_TOKEN = "|"


def extract_characters_from_texts(
    texts: Iterable[str],
    remove_diacritics: bool = False,
    strip_punctuation: bool = True,
) -> Set[str]:
    """
    Extracts all unique characters from an iterable of transcripts after Kashmiri normalization.
    """
    unique_chars: Set[str] = set()
    for text in texts:
        norm = normalize_kashmiri(
            text,
            remove_diacritics=remove_diacritics,
            strip_punctuation=strip_punctuation,
        )
        unique_chars.update(set(norm))
    return unique_chars


def build_kashmiri_ctc_vocab(
    characters: Iterable[str],
    output_path: str,
    pad_token: str = PAD_TOKEN,
    unk_token: str = UNK_TOKEN,
    bos_token: str = BOS_TOKEN,
    eos_token: str = EOS_TOKEN,
    word_delimiter_token: str = WORD_DELIMITER_TOKEN,
) -> Dict[str, int]:
    """
    Constructs a deterministic CTC character vocabulary mapping each character to an integer ID,
    persists it as UTF-8 JSON, and returns the dictionary.
    """
    sorted_chars = sorted(
        list(set(c for c in characters if c not in (" ", "\t", "\n", "\r")))
    )

    vocab_dict: Dict[str, int] = {}
    idx = 0

    for special in (pad_token, unk_token, bos_token, eos_token, word_delimiter_token):
        vocab_dict[special] = idx
        idx += 1

    for char in sorted_chars:
        if char not in vocab_dict:
            vocab_dict[char] = idx
            idx += 1

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    return vocab_dict


def build_or_load_w2v_bert_processor(
    model_name_or_path: str,
    artifacts_dir: str,
    training_texts: Iterable[str],
    remove_diacritics: bool = False,
    strip_punctuation: bool = True,
    force_rebuild: bool = False,
):
    """
    Builds or loads a persisted `Wav2Vec2BertProcessor` (pairing `SeamlessM4TFeatureExtractor`
    with `Wav2Vec2CTCTokenizer`).

    If `<artifacts_dir>/processor/vocab.json` already exists (e.g. after a Colab restart),
    loads the existing processor directly to guarantee token-ID consistency across checkpoints.
    """
    from transformers import (
        SeamlessM4TFeatureExtractor,
        Wav2Vec2BertProcessor,
        Wav2Vec2CTCTokenizer,
    )

    processor_dir = os.path.join(artifacts_dir, "processor")
    vocab_path = os.path.join(processor_dir, "vocab.json")

    if not force_rebuild and os.path.exists(vocab_path):
        print(f"[✓] Loading cached Wav2Vec2BertProcessor from: {processor_dir}")
        return Wav2Vec2BertProcessor.from_pretrained(processor_dir)

    print(f"--> Building new Kashmiri CTC vocabulary in: {processor_dir}")
    os.makedirs(processor_dir, exist_ok=True)

    chars = extract_characters_from_texts(
        training_texts,
        remove_diacritics=remove_diacritics,
        strip_punctuation=strip_punctuation,
    )
    vocab_dict = build_kashmiri_ctc_vocab(chars, output_path=vocab_path)
    print(f"[✓] Built CTC vocabulary with {len(vocab_dict)} tokens ({len(chars)} unique characters).")

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        processor_dir,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        word_delimiter_token=WORD_DELIMITER_TOKEN,
    )

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name_or_path)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(processor_dir)
    print(f"[✓] Saved Wav2Vec2BertProcessor to: {processor_dir}")
    return processor

