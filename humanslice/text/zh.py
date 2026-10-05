"""Mandarin text handling: Hanzi -> toneless pinyin syllables -> initial / final.

Spelling conventions (shared by the unit database, UTAU aliases and lyrics):
- syllables are toneless pinyin as written, with ``v`` for ``ü`` (``lv``, ``nve``, ``ju``, ``yuan``);
- the *phonetic final* undoes pinyin spelling rules so that units can be compared by sound:
  ``yuan -> van``, ``ju -> v``, ``wei -> uei``, ``liu -> iou``, ``zhi -> ir``, ``zi -> i0``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
)

# Zero-initial spellings (y / w forms) -> phonetic final.
_ZERO_INITIAL_FINALS = {
    "yi": "i", "ya": "ia", "yo": "io", "ye": "ie", "yao": "iao", "you": "iou", "yan": "ian",
    "yin": "in", "yang": "iang", "ying": "ing", "yong": "iong",
    "yu": "v", "yue": "ve", "yuan": "van", "yun": "vn",
    "wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "uei", "wan": "uan", "wen": "uen",
    "wang": "uang", "weng": "ueng",
}
# Abbreviated finals after a consonant initial.
_FINAL_EXPANSIONS = {"iu": "iou", "ui": "uei", "un": "uen"}

_HANZI = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@dataclass(frozen=True, slots=True)
class Syllable:
    char: str
    pinyin: str  # toneless, e.g. "yuan", "lv", "zhi"
    tone: int  # 1-4, 5 = neutral
    initial: str  # "" for zero initial
    final: str  # phonetic final, e.g. "van", "ir", "uei"


def split_pinyin(pinyin: str) -> tuple[str, str]:
    """Split a toneless pinyin syllable into (initial, phonetic final)."""
    syllable = pinyin.lower().replace("ü", "v").replace("u:", "v")
    if syllable in _ZERO_INITIAL_FINALS:
        return "", _ZERO_INITIAL_FINALS[syllable]
    if syllable in ("er", "r"):
        return "", "er"
    for initial in INITIALS:
        if syllable.startswith(initial) and len(syllable) > len(initial):
            final = syllable[len(initial):]
            if initial in ("j", "q", "x") and final.startswith("u"):
                final = "v" + final[1:]
            elif initial in ("zh", "ch", "sh", "r") and final == "i":
                final = "ir"
            elif initial in ("z", "c", "s") and final == "i":
                final = "i0"
            final = _FINAL_EXPANSIONS.get(final, final)
            return initial, final
    return "", syllable


_FALLING = {"ai", "ei", "ao", "ou"}
_TRIPHTHONG = {"iao", "iou", "uai", "uei"}
_RISING = {"ia", "ie", "ua", "uo", "ve", "io"}


def main_vowel_position(final: str) -> float:
    """Where the sustainable main vowel sits inside a final, as a fraction of its length.

    Singers hold the main vowel and move to the off-glide / nasal coda only at the end:
    ``ao`` holds a (early), ``ia`` holds a (late), ``an`` holds a (before the n).
    """
    if final in _FALLING:
        return 0.3
    if final in _TRIPHTHONG:
        return 0.5
    if final in _RISING:
        return 0.7
    if final.endswith(("n", "ng")):
        return 0.35
    return 0.5


def is_hanzi(char: str) -> bool:
    return bool(_HANZI.fullmatch(char))


def normalize_text(text: str) -> str:
    """NFKC-normalize, spell out Arabic numbers in Chinese, and drop spaces."""
    text = unicodedata.normalize("NFKC", text)
    if re.search(r"\d", text):
        try:
            import cn2an

            text = cn2an.transform(text, "an2cn")
        except Exception:
            text = re.sub(r"\d", lambda m: "零一二三四五六七八九"[int(m.group())], text)
    return re.sub(r"\s+", "", text)


def text_to_syllables(text: str) -> list[Syllable | None]:
    """Convert text to one entry per character; non-Hanzi characters map to ``None``.

    Uses pypinyin's phrase dictionary so polyphones are resolved in context.
    Callers treat ``None`` as a break: units around it are not contiguous.
    """
    from pypinyin import Style, lazy_pinyin

    text = normalize_text(text)
    readings = lazy_pinyin(
        text,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        v_to_u=False,
        errors=lambda chars: [None] * len(chars),
    )
    result: list[Syllable | None] = []
    for char, reading in zip(text, readings):
        if reading is None or not is_hanzi(char):
            result.append(None)
            continue
        match = re.fullmatch(r"([a-zv]+)([1-5])", reading)
        if match is None:
            result.append(None)
            continue
        pinyin, tone = match.group(1), int(match.group(2))
        initial, final = split_pinyin(pinyin)
        result.append(Syllable(char=char, pinyin=pinyin, tone=tone, initial=initial, final=final))
    return result
