from __future__ import annotations

import pytest

from humanslice.corpus.mfa_lexicon import rule_pronunciation, split_word
from humanslice.export.presamp_zh import CONSONANT_GROUP, VOWEL_GROUP, utau_spelling
from humanslice.text.zh import split_pinyin, text_to_syllables


@pytest.mark.parametrize(
    ("pinyin", "expected"),
    [
        ("ba", ("b", "a")),
        ("zhi", ("zh", "ir")),
        ("zi", ("z", "i0")),
        ("ju", ("j", "v")),
        ("quan", ("q", "van")),
        ("yuan", ("", "van")),
        ("wei", ("", "uei")),
        ("liu", ("l", "iou")),
        ("dui", ("d", "uei")),
        ("lun", ("l", "uen")),
        ("lv", ("l", "v")),
        ("er", ("", "er")),
        ("a", ("", "a")),
    ],
)
def test_split_pinyin(pinyin: str, expected: tuple[str, str]) -> None:
    assert split_pinyin(pinyin) == expected


def test_text_to_syllables_resolves_polyphones_and_marks_non_hanzi() -> None:
    syllables = text_to_syllables("银行行走，A绿")
    readings = [None if s is None else f"{s.pinyin}{s.tone}" for s in syllables]
    assert readings == ["yin2", "hang2", "xing2", "zou3", None, None, "lv4"]


def test_text_to_syllables_spells_out_numbers() -> None:
    chars = "".join(s.char for s in text_to_syllables("2个") if s is not None)
    assert chars == "两个" or chars == "二个"


def test_split_word_assigns_codas_by_pinyin() -> None:
    # 牛奶: "n j ow˧˥ n aj˨˩˦" -- the second n is the onset of nai, not a coda of niu.
    assert split_word("n j ow˧˥ n aj˨˩˦".split(), ["niu", "nai"]) == [["n", "j", "ow˧˥"], ["n", "aj˨˩˦"]]
    # 因为: "i˥ n w ej˥˩" -- n is the coda of yin.
    assert split_word("i˥ n w ej˥˩".split(), ["yin", "wei"]) == [["i˥", "n"], ["w", "ej˥˩"]]
    assert split_word("a˥".split(), ["a", "a"]) is None


def test_rule_pronunciation_shapes() -> None:
    assert rule_pronunciation("ni3") == "n i˨˩˦"
    assert rule_pronunciation("xue2") == "ɕ e˧˥"
    assert rule_pronunciation("ri4") == "ʐ̩˥˩"
    assert rule_pronunciation("guang1") == "k w a˥ ŋ"


def test_presamp_groups_cover_common_syllables() -> None:
    assert VOWEL_GROUP["ni"] == "i"
    assert VOWEL_GROUP["yuan"] == "en0"
    assert CONSONANT_GROUP["jia"] == "jy"
    assert CONSONANT_GROUP["shuo"] == "shw"
    assert utau_spelling("lve") == "lue"
