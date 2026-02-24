from pathlib import Path

from music_genre_classification.dataset import build_manifest


def test_build_manifest_discovers_audio_files(tmp_path: Path) -> None:
    rock = tmp_path / "rock"
    jazz = tmp_path / "jazz"
    rock.mkdir()
    jazz.mkdir()
    (rock / "song1.wav").write_bytes(b"dummy")
    (jazz / "song2.mp3").write_bytes(b"dummy")
    (jazz / "notes.txt").write_text("ignore")

    manifest = build_manifest(tmp_path)

    assert len(manifest) == 2
    assert {record.genre for record in manifest} == {"rock", "jazz"}
