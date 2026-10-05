"""Download a small development subset of GTSinger (CC BY-NC-SA 4.0) into devdata/.

- corpus/: one singer's Paired_Speech_Group segments (speech reading lyrics) with
  ground-truth word/phone TextGrids, used as the "speech material library".
- songs/<name>/: Control_Group segments of target songs (singing + notes + phones).

Usage: python scripts/fetch_devdata.py [--singer ZH-Tenor-1] [--song-singer ZH-Alto-1 --song 传奇]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "GTSinger/GTSinger"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = Path(__file__).resolve().parents[1] / "devdata"


def list_tree(path: str, attempts: int = 6) -> list[dict]:
    """List a dataset folder recursively, following the API's pagination cursor."""
    url = API + urllib.parse.quote(path) + "?recursive=true"
    items: list[dict] = []
    while url:
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    page = json.load(response)
                    link = response.headers.get("Link", "")
                break
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        items.extend(page)
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else ""
    return items


def download(remote: str, local: Path, attempts: int = 6) -> None:
    if local.exists():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            data = urllib.request.urlopen(RESOLVE + urllib.parse.quote(remote), timeout=300).read()
            break
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    tmp = local.with_suffix(local.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(local)


def fetch(jobs: list[tuple[str, Path]]) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(lambda job: download(*job), jobs), start=1):
            if index % 50 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--singer", default="ZH-Tenor-1", help="singer whose paired speech forms the corpus")
    parser.add_argument("--song-singer", default="ZH-Alto-1")
    parser.add_argument("--song", action="append", default=None, help="song title(s) to fetch as targets")
    parser.add_argument("--holdout", action="append", default=[], help="songs excluded from the speech corpus")
    args = parser.parse_args()
    songs = args.song or ["传奇"]

    print(f"Listing Chinese/{args.singer} ...", flush=True)
    corpus_jobs = []
    for item in list_tree(f"Chinese/{args.singer}"):
        parts = item["path"].split("/")
        # Chinese/<singer>/<technique>/<song>/<group>/<file>
        if item["type"] != "file" or len(parts) != 6 or parts[4] != "Paired_Speech_Group":
            continue
        technique, song, name = parts[2], parts[3], parts[5]
        if song in args.holdout or not name.endswith((".wav", ".TextGrid", ".json")):
            continue
        corpus_jobs.append((item["path"], ROOT / "corpus" / f"{technique}_{song}_{name}"))
    print(f"Corpus: {len(corpus_jobs)} files", flush=True)
    fetch(corpus_jobs)

    print(f"Listing Chinese/{args.song_singer} ...", flush=True)
    song_jobs = []
    for item in list_tree(f"Chinese/{args.song_singer}"):
        parts = item["path"].split("/")
        if item["type"] != "file" or len(parts) != 6 or parts[4] != "Control_Group" or parts[3] not in songs:
            continue
        song_jobs.append((item["path"], ROOT / "songs" / f"{args.song_singer}_{parts[3]}_{parts[2]}" / parts[5]))
    print(f"Songs: {len(song_jobs)} files", flush=True)
    fetch(song_jobs)
    write_labels(ROOT / "corpus")
    return 0


def write_labels(folder: Path) -> None:
    """Sidecar .lab transcripts (Hanzi only) from the GTSinger word annotations."""
    for json_path in folder.glob("*.json"):
        words = json.loads(json_path.read_text(encoding="utf-8"))
        text = "".join(w["word"] for w in words if not w["word"].startswith("<"))
        json_path.with_suffix(".lab").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
