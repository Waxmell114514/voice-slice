"""Minimal Praat TextGrid (long text format) reader / writer for interval tiers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Interval:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TextGrid:
    xmin: float = 0.0
    xmax: float = 0.0
    tiers: dict[str, list[Interval]] = field(default_factory=dict)

    def tier(self, name: str) -> list[Interval]:
        """Return a tier by exact name, falling back to a case-insensitive prefix match."""
        if name in self.tiers:
            return self.tiers[name]
        for tier_name, intervals in self.tiers.items():
            if tier_name.lower().startswith(name.lower()):
                return intervals
        raise KeyError(f"TextGrid has no tier {name!r} (tiers: {list(self.tiers)})")


_NUMBER = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"


def read_textgrid(path: str | Path) -> TextGrid:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gbk"):
        try:
            content = raw.decode(encoding)
            if "TextGrid" in content:
                break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode TextGrid: {path}")

    grid = TextGrid()
    header = re.search(rf"xmin\s*=\s*{_NUMBER}\s*xmax\s*=\s*{_NUMBER}", content)
    if header:
        grid.xmin, grid.xmax = float(header.group(1)), float(header.group(2))

    for block in re.split(r"item\s*\[\d+\]\s*:", content)[1:]:
        if '"IntervalTier"' not in block:
            continue
        name_match = re.search(r'name\s*=\s*"((?:[^"]|"")*)"', block)
        name = name_match.group(1).replace('""', '"') if name_match else f"tier{len(grid.tiers) + 1}"
        intervals = [
            Interval(float(m.group(1)), float(m.group(2)), m.group(3).replace('""', '"'))
            for m in re.finditer(
                rf'xmin\s*=\s*{_NUMBER}\s*xmax\s*=\s*{_NUMBER}\s*text\s*=\s*"((?:[^"]|"")*)"',
                block,
            )
        ]
        grid.tiers[name] = intervals
    return grid


def write_textgrid(grid: TextGrid, path: str | Path) -> None:
    def esc(text: str) -> str:
        return text.replace('"', '""')

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {grid.xmin}",
        f"xmax = {grid.xmax}",
        "tiers? <exists>",
        f"size = {len(grid.tiers)}",
        "item []:",
    ]
    for index, (name, intervals) in enumerate(grid.tiers.items(), start=1):
        lines += [
            f"    item [{index}]:",
            '        class = "IntervalTier"',
            f'        name = "{esc(name)}"',
            f"        xmin = {grid.xmin}",
            f"        xmax = {grid.xmax}",
            f"        intervals: size = {len(intervals)}",
        ]
        for number, interval in enumerate(intervals, start=1):
            lines += [
                f"        intervals [{number}]:",
                f"            xmin = {interval.start}",
                f"            xmax = {interval.end}",
                f'            text = "{esc(interval.text)}"',
            ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
