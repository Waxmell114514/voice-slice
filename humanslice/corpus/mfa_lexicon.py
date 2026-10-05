"""Syllable-level pronunciation lexicon for MFA's Mandarin model.

MFA's ``mandarin_china_mfa`` dictionary is word-based (only ~90 single-character
entries), but every syllable in it carries exactly one tone-marked phone. We split each
multi-character entry into syllables (using pypinyin readings to know where codas -n /
-ng / -r belong) and count which phone sequence each toned syllable (``ni3``) uses.
Alignment then runs on toned-pinyin tokens, so polyphones are resolved by pypinyin in
context and every word interval is exactly one syllable.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from humanslice.text.zh import is_hanzi, split_pinyin

TONE_MARKS = "˥˦˧˨˩"
TONE_CONTOURS = {1: "˥", 2: "˧˥", 3: "˨˩˦", 4: "˥˩", 5: "˧"}
CONSONANT_BASES = {
    "p", "t", "k", "m", "n", "f", "x", "ɕ", "ʂ", "ʐ", "s", "l", "ʎ", "ɲ", "ts", "tɕ", "ʈʂ", "ʔ",
}


def base_phone(phone: str) -> str:
    """Strip tone letters and secondary articulations (ʰ ʷ ʲ)."""
    return re.sub(rf"[{TONE_MARKS}ʰʷʲ]", "", phone)


def is_toned(phone: str) -> bool:
    return any(mark in phone for mark in TONE_MARKS)


def is_consonant(phone: str) -> bool:
    return not is_toned(phone) and base_phone(phone) in CONSONANT_BASES


def _expected_coda(pinyin: str) -> str | None:
    if pinyin.endswith("ng"):
        return "ŋ"
    if pinyin.endswith("n"):
        return "n"
    if pinyin.endswith("r") and len(pinyin) > 1:
        return "ɻ"
    return None


def split_word(phones: list[str], syllables: list[str]) -> list[list[str]] | None:
    """Split a word's phones into per-syllable groups; ``syllables`` are toneless pinyin."""
    groups: list[list[str]] = []
    pos = 0
    for syllable in syllables:
        toned = next((i for i in range(pos, len(phones)) if is_toned(phones[i])), None)
        if toned is None:
            return None
        end = toned + 1
        coda = _expected_coda(syllable)
        if coda and end < len(phones) and base_phone(phones[end]) == coda:
            end += 1
        groups.append(phones[pos:end])
        pos = end
    return groups if pos == len(phones) else None


def build_lexicon(dictionary_path: Path, cache_path: Path | None = None, top: int = 2) -> dict[str, list[str]]:
    """Map toned pinyin (``"ni3"``) -> up to ``top`` most frequent phone strings."""
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    from pypinyin import Style, lazy_pinyin

    counts: dict[str, Counter] = defaultdict(Counter)
    for line in dictionary_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word, phones = parts[0], parts[-1].split()
        if not word or not all(is_hanzi(ch) for ch in word):
            continue
        readings = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
        if len(readings) != len(word) or not all(re.fullmatch(r"[a-zv]+[1-5]", r) for r in readings):
            continue
        groups = split_word(phones, [r[:-1] for r in readings])
        if groups is None:
            continue
        for reading, group in zip(readings, groups):
            counts[reading][" ".join(group)] += 1

    lexicon: dict[str, list[str]] = {}
    for reading, counter in counts.items():
        rule = rule_pronunciation(reading)
        # Drop variants that come from polyphone mismatches (e.g. 率 lv4 vs shuai4).
        variants = [p for p, _ in counter.most_common() if rule is None or _consistent(p, rule)]
        if variants:
            lexicon[reading] = variants[:top]
    if cache_path is not None:
        cache_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=0), encoding="utf-8")
    return lexicon


# ---------------------------------------------------------------- rule-based fallback

_INITIAL_IPA = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x", "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ʐ", "z": "ts", "c": "tsʰ", "s": "s",
}
# phonetic final -> (pre-nucleus glide phones, toned nucleus, coda phones)
_FINAL_IPA = {
    "a": ([], "a", []), "ai": ([], "aj", []), "ao": ([], "aw", []), "an": ([], "a", ["n"]),
    "ang": ([], "a", ["ŋ"]), "o": ([], "o", []), "ou": ([], "ow", []), "ong": ([], "u", ["ŋ"]),
    "e": ([], "o", []), "ei": ([], "ej", []), "en": ([], "ə", ["n"]), "eng": ([], "ə", ["ŋ"]),
    "er": ([], "o", ["ɻ"]), "i": ([], "i", []), "ir": ([], "ʐ̩", []), "i0": ([], "z̩", []),
    "ia": (["j"], "a", []), "ie": (["j"], "e", []), "iao": (["j"], "aw", []), "iou": (["j"], "ow", []),
    "ian": (["j"], "e", ["n"]), "in": ([], "i", ["n"]), "iang": (["j"], "a", ["ŋ"]), "ing": ([], "i", ["ŋ"]),
    "iong": ([], "u", ["ŋ"]), "io": (["j"], "o", []),
    "u": ([], "u", []), "ua": (["w"], "a", []), "uo": (["w"], "o", []), "uai": (["w"], "aj", []),
    "uei": (["w"], "ej", []), "uan": (["w"], "a", ["n"]), "uen": (["w"], "ə", ["n"]),
    "uang": (["w"], "a", ["ŋ"]), "ueng": (["w"], "ə", ["ŋ"]),
    "v": ([], "y", []), "ve": (["ɥ"], "e", []), "van": (["ɥ"], "e", ["n"]), "vn": ([], "y", ["n"]),
}


def rule_pronunciation(toned: str) -> str | None:
    match = re.fullmatch(r"([a-zv]+)([1-5])", toned)
    if not match:
        return None
    initial, final = split_pinyin(match.group(1))
    if final not in _FINAL_IPA:
        return None
    glides, nucleus, coda = _FINAL_IPA[final]
    if initial in ("j", "q", "x"):
        glides = []  # medial absorbed by the alveolo-palatal initial
    if initial == "r" and final == "ir":
        initial = ""  # MFA writes 日 as a single syllabic ʐ̩
    phones = ([_INITIAL_IPA[initial]] if initial else []) + glides + [nucleus + TONE_CONTOURS[int(match.group(2))]] + coda
    return " ".join(phones)


def _signature(pronunciation: str) -> tuple[str, str]:
    """(initial class, nucleus vowel class) used to compare pronunciations loosely."""
    phones = pronunciation.split()
    first = base_phone(phones[0]) if phones and is_consonant(phones[0]) else ""
    first = {"ɲ": "n", "ʎ": "l", "ʔ": ""}.get(first, first)
    nucleus = next((base_phone(p) for p in phones if is_toned(p)), "")
    vowel = nucleus[:1]
    vowel = "e" if vowel in ("o", "ə", "e") else vowel
    return first, vowel


def _consistent(variant: str, rule: str) -> bool:
    return _signature(variant) == _signature(rule)


def pronunciations(lexicon: dict[str, list[str]], toned: str) -> list[str]:
    found = lexicon.get(toned)
    if found:
        return found
    rule = rule_pronunciation(toned)
    return [rule] if rule else []
