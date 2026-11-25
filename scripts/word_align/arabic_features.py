"""Arabic / Uthmani word feature extraction for PAW-based forced alignment."""
from typing import Dict, List, Tuple
import json

NON_CONNECTORS = set('اأإآٱدذرزوؤةىء')
TALL_LETTERS = set('اأإآٱلكطظ')
ALIF_LETTERS = set('اأإآٱ')
BASE_LETTERS = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآٱؤئىةء')
WIDE_LETTERS = set('سشصضطظك')
NARROW_LETTERS = set('اأإآٱلرذدزوء')


def load_quran_text(path: str) -> Dict[str, str]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def clean_uthmani_base(word: str) -> str:
    """Keep only base Arabic letters (ignoring tashkeel, dagger alif, waqf marks)."""
    return ''.join(ch for ch in word if ch in BASE_LETTERS)


def get_uthmani_words_for_ayah(quran_text: Dict[str, str], aya_index: int) -> List[str]:
    txt = quran_text.get(str(aya_index), '')
    if not txt:
        return []
    return [t for t in txt.split() if clean_uthmani_base(t)]


def get_word_features(uthmani_word: str) -> Dict[str, object]:
    """
    Extract calligraphic layout features invariant to kashida elongation:
      - exp_paws: expected number of Pieces of Arabic Words (connected strokes)
      - weight: approximate relative horizontal width weight
      - exp_tall: expected number of tall vertical strokes (ا ل ك ط ظ)
      - starts_alif: True if word starts with standalone Alif (ا أ إ آ ٱ)
      - starts_tall: True if word starts with any tall letter (ا ل ك ط ظ)
    """
    base = clean_uthmani_base(uthmani_word)
    if not base:
        return {
            'base': '',
            'exp_paws': 1,
            'weight': 1.0,
            'exp_tall': 0,
            'starts_alif': False,
            'starts_tall': False,
        }

    # Count PAWs: each non-connector ends a PAW
    paws = 0
    i = 0
    n = len(base)
    while i < n:
        paws += 1
        while i < n and base[i] not in NON_CONNECTORS:
            i += 1
        i += 1

    weight = 0.0
    for ch in base:
        if ch in WIDE_LETTERS:
            weight += 1.5
        elif ch in NARROW_LETTERS:
            weight += 0.65
        else:
            weight += 1.0

    exp_tall = sum(1 for ch in base if ch in TALL_LETTERS)
    return {
        'base': base,
        'exp_paws': max(1, paws),
        'weight': max(0.8, weight),
        'exp_tall': exp_tall,
        'starts_alif': base[0] in ALIF_LETTERS,
        'starts_tall': base[0] in TALL_LETTERS,
    }
