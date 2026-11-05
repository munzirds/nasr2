"""
Kashmiri (Koshur) Perso-Arabic Text Normalization & ASR Evaluation Metrics.
===========================================================================
Provides:
1. Unicode canonicalization (NFKC, standardizing Perso-Arabic variants: Yeh, Kaf, Heh,
   deprecated precomposed wavy-hamza forms, and zero-width formatting marks).
2. Punctuation & symbol filtering that strictly preserves vocalic combining marks (Unicode category 'M').
3. Diacritic-Aware (DA) and Diacritic-Agnostic (DG) normalization modes:
   - DA: Retains all vocalic combining diacritics and specialized Kashmiri vowel letters.
   - DG: Strips combining marks and maps specialized Kashmiri vowel letters (ٲ, ٳ, ۄ, ۆ, ۇ, ێ, ؠ, ۓ)
         to base Perso-Arabic letters (ا, و, ی, ے).
4. Comprehensive ASR evaluation metrics: DA-WER, DG-WER, CER, DG-CER, and Diacritic Degradation Gap (DDG).
"""

import unicodedata
from typing import Dict, List

try:
    import jiwer
except ImportError:
    jiwer = None


# Deprecated precomposed characters and cross-keyboard Perso-Arabic variants
UNICODE_CANONICAL_MAPPINGS: Dict[str, str] = {
    "\u0673": "\u0627\u065F",  # Deprecated Alef with Wavy Hamza Below -> Alef + U+065F
    "\u0649": "\u06CC",        # Arabic Alef Maksura -> Farsi/Urdu Yeh
    "\u064A": "\u06CC",        # Arabic Yeh (with two dots) -> Farsi/Urdu Yeh
    "\u0643": "\u06A9",        # Arabic Kaf -> Keheh
    "\u0647": "\u06C1",        # Arabic Heh -> Heh Goal
}

# Specialized Kashmiri vowel letters mapped to base Perso-Arabic letters in DG mode
KASHMIRI_VOWEL_LETTER_TO_BASE: Dict[str, str] = {
    "\u0672": "\u0627",  # ٲ (Alef with wavy hamza above) -> ا (Alef)
    "\u0673": "\u0627",  # ٳ (Alef with wavy hamza below) -> ا (Alef)
    "\u06C4": "\u0648",  # ۄ (Waw with ring) -> و (Waw)
    "\u06C6": "\u0648",  # ۆ (Waw with small V) -> و (Waw)
    "\u06C7": "\u0648",  # ۇ (Waw with damma) -> و (Waw)
    "\u06CE": "\u06CC",  # ێ (Yeh with small V) -> ی (Yeh)
    "\u0620": "\u06CC",  # ؠ (Kashmiri Yeh / palatalization marker) -> ی (Yeh)
    "\u06D3": "\u06D2",  # ۓ (Yeh barree with hamza) -> ے (Yeh barree)
}


def canonicalize_unicode(text: str) -> str:
    """
    Applies Unicode normalization (NFKC) and standardizes Perso-Arabic code points
    frequently interchanged across Kashmiri, Urdu, Persian, and Arabic keyboards.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    # Replace ZWNJ with space; strip ZWJ, Tatweel (Kashida), and BOM
    text = (
        text.replace("\u200C", " ")
        .replace("\u200D", "")
        .replace("\u0640", "")
        .replace("\uFEFF", "")
    )
    for src, dst in UNICODE_CANONICAL_MAPPINGS.items():
        text = text.replace(src, dst)
    return text


def strip_combining_marks(text: str) -> str:
    """Strips all combining diacritical marks (Unicode category 'M')."""
    if not text:
        return ""
    return "".join(c for c in text if not unicodedata.category(c).startswith("M"))


def strip_punctuation_and_symbols(text: str) -> str:
    """
    Removes punctuation and symbols (Unicode categories starting with 'P' or 'S')
    while strictly preserving letters, numbers, whitespace, and combining diacritical marks.
    """
    if not text:
        return ""
    cleaned = []
    for c in text:
        cat = unicodedata.category(c)
        if not (cat.startswith("P") or cat.startswith("S")):
            cleaned.append(c)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split()).strip()


def normalize_kashmiri(
    text: str,
    remove_diacritics: bool = False,
    strip_punctuation: bool = True,
) -> str:
    """
    Full normalization pipeline for Kashmiri Perso-Arabic text.

    Args:
        text: Input raw text string.
        remove_diacritics: If True, operates in Diacritic-Agnostic (DG) mode:
                           strips all combining marks and normalizes Kashmiri vowel letters
                           (ٲ, ٳ, ۄ, ۆ, ێ, ؠ) to base Perso-Arabic letters (ا, و, ی).
                           If False, operates in Diacritic-Aware (DA) mode: retains all marks.
        strip_punctuation: If True, removes non-alphanumeric punctuation and collapses whitespace.
    """
    if not text:
        return ""

    text = canonicalize_unicode(text)

    if remove_diacritics:
        text = strip_combining_marks(text)
        for k, v in KASHMIRI_VOWEL_LETTER_TO_BASE.items():
            text = text.replace(k, v)

    if strip_punctuation:
        text = strip_punctuation_and_symbols(text)
    else:
        text = " ".join(text.split()).strip()

    return text


def _levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Pure-Python dynamic programming edit distance fallback when jiwer is unavailable."""
    if len(seq1) < len(seq2):
        return _levenshtein_distance(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)
    previous_row = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _compute_error_rate(references: List[str], hypotheses: List[str], char_level: bool = False) -> float:
    """Computes WER or CER in [0.0, 100.0+], safely handling empty strings."""
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0
    refs_filtered = [p[0] for p in pairs]
    hyps_filtered = [p[1] for p in pairs]

    if jiwer is not None:
        if char_level:
            return float(jiwer.cer(refs_filtered, hyps_filtered)) * 100.0
        return float(jiwer.wer(refs_filtered, hyps_filtered)) * 100.0

    total_edits = 0
    total_length = 0
    for r, h in zip(refs_filtered, hyps_filtered):
        r_tokens = list(r) if char_level else r.split()
        h_tokens = list(h) if char_level else h.split()
        total_edits += _levenshtein_distance(r_tokens, h_tokens)
        total_length += len(r_tokens)
    return (total_edits / max(1, total_length)) * 100.0


def compute_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """
    Computes Kashmiri ASR evaluation metrics:
    - da_wer: Diacritic-Aware Word Error Rate (%)
    - dg_wer: Diacritic-Agnostic Word Error Rate (%)
    - cer: Diacritic-Aware Character Error Rate (%)
    - dg_cer: Diacritic-Agnostic Character Error Rate (%)
    - diacritic_degradation_gap: da_wer - dg_wer (%)
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have identical length.")
    if not references:
        return {
            "da_wer": 0.0,
            "dg_wer": 0.0,
            "cer": 0.0,
            "dg_cer": 0.0,
            "diacritic_degradation_gap": 0.0,
        }

    refs_da = [normalize_kashmiri(r, remove_diacritics=False) for r in references]
    hyps_da = [normalize_kashmiri(h, remove_diacritics=False) for h in hypotheses]

    refs_dg = [normalize_kashmiri(r, remove_diacritics=True) for r in references]
    hyps_dg = [normalize_kashmiri(h, remove_diacritics=True) for h in hypotheses]

    da_wer = _compute_error_rate(refs_da, hyps_da, char_level=False)
    dg_wer = _compute_error_rate(refs_dg, hyps_dg, char_level=False)
    cer = _compute_error_rate(refs_da, hyps_da, char_level=True)
    dg_cer = _compute_error_rate(refs_dg, hyps_dg, char_level=True)
    gap = da_wer - dg_wer

    return {
        "da_wer": round(da_wer, 2),
        "dg_wer": round(dg_wer, 2),
        "cer": round(cer, 2),
        "dg_cer": round(dg_cer, 2),
        "diacritic_degradation_gap": round(gap, 2),
    }

