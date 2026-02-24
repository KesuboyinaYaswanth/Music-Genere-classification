from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".au", ".ogg", ".flac"}


@dataclass(frozen=True)
class TrackRecord:
    genre: str
    filepath: Path


def discover_tracks(dataset_root: Path) -> list[TrackRecord]:
    """Find audio files in a folder structure like: dataset/genre_name/*.wav"""
    records: list[TrackRecord] = []
    for genre_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for file_path in genre_dir.iterdir():
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                records.append(TrackRecord(genre=genre_dir.name, filepath=file_path))
    return records


def save_manifest(records: list[TrackRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["genre", "filepath"])
        writer.writeheader()
        for record in records:
            writer.writerow({"genre": record.genre, "filepath": str(record.filepath)})


def load_manifest(manifest_csv: Path) -> list[TrackRecord]:
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [TrackRecord(genre=row["genre"], filepath=Path(row["filepath"])) for row in reader]


def build_manifest(dataset_root: Path, output_csv: Path | None = None) -> list[TrackRecord]:
    records = discover_tracks(dataset_root)
    if output_csv:
        save_manifest(records, output_csv)
    return records
