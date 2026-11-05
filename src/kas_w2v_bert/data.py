"""
Dataset Loading, Deterministic Splitting & Fast Preparation for Kashmiri ASR.
=============================================================================
Designed specifically for `munzirahangar/kas-news-dataset` (22,425 utterances, ~11 GB) on Google Colab:
1. Authenticates with Hugging Face using `HF_TOKEN` (from environment, `.env`, or Colab Secrets).
2. Reads and normalizes the `sentence` column in <3 seconds without decoding 87 hours of audio up-front.
3. Creates deterministic `train` / `validation` / `test` split indices and caches them in
   `<output_dir>/artifacts/splits.json` so restarts never reshuffle splits or leak evaluation data.
4. Pre-tokenizes target labels (`labels`) and computes `input_length` for `group_by_length=True`
   batching while keeping `audio` lazy-decoded (`Audio(sampling_rate=16000, decode=False)`),
   enabling instant startup on Colab restarts.
"""

import io
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from kas_w2v_bert.config import TrainingConfig
from kas_w2v_bert.normalizer import normalize_kashmiri


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Resolves Hugging Face token from:
    1. Explicit argument
    2. `HF_TOKEN` or `HUGGINGFACE_TOKEN` environment variables
    3. Local `.env` file
    4. Google Colab `userdata.get('HF_TOKEN')`
    """
    if explicit_token:
        return explicit_token.strip()

    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    # Check .env in current working directory or project root
    for candidate in (".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")):
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                                token = v.strip().strip('"').strip("'")
                                if token:
                                    os.environ["HF_TOKEN"] = token
                                    return token
            except Exception:
                pass

    # Check Google Colab Secrets if running in Colab
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("HF_TOKEN")
        if token and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            return token.strip()
    except Exception:
        pass

    return None


def decode_audio_dict(
    audio_obj: Dict[str, Any],
    target_sr: int = 16000,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 30.0,
) -> np.ndarray:
    """
    Decodes an audio dictionary (either pre-decoded `{'array': ...}` or lazy `{'bytes': ..., 'path': ...}`)
    into a 1D mono `np.float32` waveform at `target_sr`, clamped to `[min_duration_sec, max_duration_sec]`.
    """
    arr: Optional[np.ndarray] = None
    sr: int = target_sr

    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate") or target_sr)
        else:
            raw_bytes = audio_obj.get("bytes")
            raw_path = audio_obj.get("path")
            import soundfile as sf

            if raw_bytes:
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            elif raw_path and os.path.exists(raw_path):
                arr, sr = sf.read(raw_path, dtype="float32", always_2d=False)

    if arr is None or arr.size == 0:
        # Return short silence fallback so a corrupted clip never crashes a long Colab training run
        return np.zeros(int(min_duration_sec * target_sr), dtype=np.float32)

    # Convert stereo/multi-channel to mono
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1).astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        except ImportError:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(sr, target_sr)
            arr = resample_poly(arr, target_sr // g, sr // g).astype(np.float32)

    # Clamp duration to [min_duration_sec, max_duration_sec]
    min_samples = int(min_duration_sec * target_sr)
    max_samples = int(max_duration_sec * target_sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    elif arr.shape[0] < min_samples:
        pad_width = min_samples - arr.shape[0]
        arr = np.pad(arr, (0, pad_width), mode="constant")

    # Replace any NaN/Inf values
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return arr


def resolve_dataset_columns(
    column_names: List[str],
    preferred_audio: str = "audio",
    preferred_text: str = "sentence",
) -> Tuple[str, str]:
    """
    Resolves the audio and text/transcript column names from `column_names`.
    Uses `preferred_audio` / `preferred_text` if present, otherwise auto-detects common HF ASR names.
    """
    audio_col = preferred_audio if preferred_audio in column_names else None
    if audio_col is None:
        for candidate in ("audio", "speech", "wav", "audio_filepath", "file"):
            if candidate in column_names:
                audio_col = candidate
                break
    if audio_col is None:
        raise ValueError(
            f"Could not find audio column '{preferred_audio}' in dataset columns: {column_names}. "
            f"Pass --audio_column <name> explicitly."
        )

    text_col = preferred_text if preferred_text in column_names else None
    if text_col is None:
        for candidate in ("sentence", "transcript", "text", "normalized", "normalized_text", "transcription"):
            if candidate in column_names:
                text_col = candidate
                break
    if text_col is None:
        raise ValueError(
            f"Could not find text column '{preferred_text}' in dataset columns: {column_names}. "
            f"Pass --text_column <name> explicitly."
        )

    return audio_col, text_col


def compute_or_load_splits(
    valid_indices: List[int],
    artifacts_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    dataset_name: Optional[str] = None,
    dataset_subset: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Deterministically partitions `valid_indices` into `train`, `validation`, and `test` index lists.
    Persists the result to `<artifacts_dir>/splits.json` so restarts always use identical splits.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    splits_path = os.path.join(artifacts_dir, "splits.json")

    metadata_signature: Dict[str, Any] = {
        "num_valid_indices": len(valid_indices),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
    }
    if dataset_name:
        metadata_signature["dataset_name"] = dataset_name
    if dataset_subset:
        metadata_signature["dataset_subset"] = dataset_subset

    if os.path.exists(splits_path):
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("signature") == metadata_signature:
                print(
                    f"[✓] Loaded cached dataset splits from {splits_path} "
                    f"(train={len(saved['train'])}, val={len(saved['validation'])}, test={len(saved['test'])})"
                )
                return {
                    "train": saved["train"],
                    "validation": saved["validation"],
                    "test": saved["test"],
                }
        except Exception as e:
            print(f"[!] Could not read cached splits.json ({e}); regenerating...")

    rng = random.Random(seed)
    shuffled = list(valid_indices)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n_total * val_ratio))) if val_ratio > 0 else 0

    test_idx = sorted(shuffled[:n_test])
    val_idx = sorted(shuffled[n_test : n_test + n_val])
    train_idx = sorted(shuffled[n_test + n_val :])

    payload = {
        "signature": metadata_signature,
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[✓] Saved deterministic splits to {splits_path} "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )
    return {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }


def discover_drive_shard_dirs(
    drive_ds_dir: str,
    shard_glob: str = "batch_*",
) -> List[str]:
    """
    Discovers valid Hugging Face Arrow shard directories (`state.json` present) inside:
    1. `<drive_ds_dir>/shards/<shard_glob>` (standard VoiceHub layout)
    2. `<drive_ds_dir>/<shard_glob>` (when `drive_ds_dir` points directly to `shards/`)
    3. `<drive_ds_dir>` itself (if it is a single `save_to_disk` dataset directory)
    """
    base_path = Path(drive_ds_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Google Drive dataset path does not exist: {drive_ds_dir}")

    # 1. Check <drive_ds_dir>/shards/batch_*
    shards_subdir = base_path / "shards"
    if shards_subdir.is_dir():
        shard_dirs = sorted(
            str(p)
            for p in shards_subdir.glob(shard_glob)
            if p.is_dir() and (p / "state.json").exists()
        )
        if shard_dirs:
            return shard_dirs

    # 2. Check <drive_ds_dir>/batch_* directly
    shard_dirs = sorted(
        str(p)
        for p in base_path.glob(shard_glob)
        if p.is_dir() and (p / "state.json").exists()
    )
    if shard_dirs:
        return shard_dirs

    # 3. Check if <drive_ds_dir> itself is a saved Dataset directory
    if (base_path / "state.json").exists():
        return [str(base_path)]

    raise FileNotFoundError(
        f"No valid saved dataset shards (matching '{shard_glob}' with state.json) found in "
        f"'{shards_subdir}' or '{base_path}'."
    )


def load_gdrive_sharded_dataset(
    drive_ds_dir: str,
    shard_glob: str = "batch_*",
    sampling_rate: int = 16000,
    preferred_audio_col: str = "audio",
    preferred_text_col: str = "transcript",
) -> Tuple[Any, str, str]:
    """
    Loads a sharded (or single) dataset from Google Drive using `load_from_disk` and
    `concatenate_datasets`, validates required columns, and casts the audio column to 16 kHz.
    """
    from datasets import Audio, DatasetDict, concatenate_datasets, load_from_disk

    # Handle DatasetDict saved to disk if dataset_dict.json is present
    base_path = Path(drive_ds_dir)
    if (base_path / "dataset_dict.json").exists():
        print(f"--> Loading DatasetDict from Google Drive: {drive_ds_dir}")
        ds_obj = load_from_disk(str(base_path))
        if isinstance(ds_obj, DatasetDict):
            ds = concatenate_datasets(list(ds_obj.values()))
        else:
            ds = ds_obj
    else:
        shard_dirs = discover_drive_shard_dirs(drive_ds_dir, shard_glob=shard_glob)
        print(f"--> Loading {len(shard_dirs)} shards from Google Drive ({drive_ds_dir})...")
        loaded_shards = [load_from_disk(p) for p in shard_dirs]
        ds = concatenate_datasets(loaded_shards) if len(loaded_shards) > 1 else loaded_shards[0]

    audio_col, text_col = resolve_dataset_columns(
        list(ds.column_names),
        preferred_audio=preferred_audio_col,
        preferred_text=preferred_text_col,
    )
    required = {audio_col, text_col}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. Found: {ds.column_names}"
        )

    # Explicitly decode/resample audio to the model's required 16 kHz rate
    ds = ds.cast_column(audio_col, Audio(sampling_rate=sampling_rate))
    print(f"Audio feature after {sampling_rate // 1000} kHz cast:", ds.features[audio_col])
    print("Rows:", len(ds))
    print("Columns:", ds.column_names)
    print("Features:", ds.features)
    return ds, audio_col, text_col


def load_and_prepare_kas_news_splits(
    cfg: TrainingConfig,
    token: Optional[str] = None,
) -> Tuple[Any, Dict[str, List[int]], List[str]]:
    """
    Loads an ASR dataset from either:
      - Google Drive (`cfg.dataset_source == 'gdrive'` or `cfg.drive_dataset_dir` / local path), or
      - Hugging Face Hub (`cfg.dataset_name`, e.g., `munzirahangar/kas-news-dataset`),
    extracts and normalizes transcripts without decoding all audio up-front, and returns:
      `(raw_dataset, split_indices_dict, train_texts_for_vocab)`
    """
    from datasets import Audio, load_dataset

    source_mode = (cfg.dataset_source or "auto").lower().strip()
    drive_dir = cfg.drive_dataset_dir
    if not drive_dir and (
        source_mode == "gdrive"
        or os.path.exists(cfg.dataset_name)
        or cfg.dataset_name.startswith("/content/drive")
    ):
        drive_dir = cfg.dataset_name

    use_gdrive = source_mode == "gdrive" or (source_mode == "auto" and bool(drive_dir))

    if use_gdrive:
        if not drive_dir:
            raise ValueError(
                "dataset_source='gdrive' requires --drive_dataset_dir (or passing the Drive path to --dataset_name)."
            )
        ds, resolved_audio_col, resolved_text_col = load_gdrive_sharded_dataset(
            drive_ds_dir=drive_dir,
            shard_glob=cfg.shard_glob,
            sampling_rate=cfg.sampling_rate,
            preferred_audio_col=cfg.audio_column,
            preferred_text_col=cfg.text_column,
        )
        cfg.audio_column = resolved_audio_col
        cfg.text_column = resolved_text_col
        dataset_identifier = f"gdrive:{os.path.abspath(drive_dir)}"
    else:
        hf_token = resolve_hf_token(token)
        print(
            f"--> Loading Hugging Face dataset: '{cfg.dataset_name}' (split='{cfg.dataset_split}')..."
        )
        ds = load_dataset(
            cfg.dataset_name,
            name=cfg.dataset_subset,
            split=cfg.dataset_split,
            token=hf_token,
            cache_dir=cfg.hf_cache_dir,
        )
        resolved_audio_col, resolved_text_col = resolve_dataset_columns(
            list(ds.column_names),
            preferred_audio=cfg.audio_column,
            preferred_text=cfg.text_column,
        )
        if resolved_audio_col != cfg.audio_column or resolved_text_col != cfg.text_column:
            print(
                f"--> Auto-detected dataset columns: audio_column='{resolved_audio_col}', "
                f"text_column='{resolved_text_col}'"
            )
            cfg.audio_column = resolved_audio_col
            cfg.text_column = resolved_text_col

        if cfg.audio_column in ds.column_names:
            ds = ds.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sampling_rate))
            print(
                f"Audio feature after {cfg.sampling_rate // 1000} kHz cast:",
                ds.features[cfg.audio_column],
            )
            print("Rows:", len(ds))
            print("Columns:", ds.column_names)
        dataset_identifier = cfg.dataset_name

    # Fast text-only pass (reads only the Arrow/Parquet string column, 0 audio bytes decoded)
    raw_texts: List[str] = ds.select_columns([cfg.text_column])[cfg.text_column]
    valid_indices: List[int] = []
    normalized_texts_map: Dict[int, str] = {}

    for idx, raw_t in enumerate(raw_texts):
        norm_t = normalize_kashmiri(
            raw_t or "",
            remove_diacritics=cfg.remove_diacritics,
            strip_punctuation=cfg.strip_punctuation,
        )
        if len(norm_t) > 0:
            valid_indices.append(idx)
            normalized_texts_map[idx] = norm_t

    print(
        f"[✓] Verified {len(valid_indices)}/{len(raw_texts)} utterances with non-empty normalized Kashmiri transcripts."
    )

    splits = compute_or_load_splits(
        valid_indices=valid_indices,
        artifacts_dir=cfg.artifacts_dir,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        dataset_name=dataset_identifier,
        dataset_subset=cfg.dataset_subset,
    )

    train_texts = [normalized_texts_map[i] for i in splits["train"]]
    return ds, splits, train_texts


def attach_tokenized_labels_to_split(
    raw_ds: Any,
    indices: List[int],
    processor: Any,
    cfg: TrainingConfig,
    max_samples: Optional[int] = None,
    split_name: str = "train",
) -> Any:
    """
    Selects `indices` from `raw_ds` and attaches tokenized CTC `labels`, normalized `transcript`,
    `sample_idx`, and `input_length` (for `group_by_length=True`) without decoding audio.
    """
    if max_samples is not None and max_samples > 0 and len(indices) > max_samples:
        indices = indices[:max_samples]

    sub_ds = raw_ds.select(indices)
    raw_texts: List[str] = sub_ds.select_columns([cfg.text_column])[cfg.text_column]

    norm_texts: List[str] = []
    labels_list: List[List[int]] = []
    input_lengths: List[int] = []

    for t in raw_texts:
        norm_t = normalize_kashmiri(
            t or "",
            remove_diacritics=cfg.remove_diacritics,
            strip_punctuation=cfg.strip_punctuation,
        )
        token_ids = processor.tokenizer(norm_t).input_ids
        norm_texts.append(norm_t)
        labels_list.append(token_ids)
        input_lengths.append(len(token_ids))

    sub_ds = sub_ds.add_column("sample_idx", indices)
    sub_ds = sub_ds.add_column("transcript", norm_texts)
    sub_ds = sub_ds.add_column("labels", labels_list)
    sub_ds = sub_ds.add_column("input_length", input_lengths)

    print(f"[✓] Prepared '{split_name}' split: {len(sub_ds)} utterances ready.")
    return sub_ds
