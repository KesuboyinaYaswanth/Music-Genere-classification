from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


FEATURE_NAMES = [
    "tempo",
    *[f"mfcc_{i}_mean" for i in range(1, 21)],
    *[f"mfcc_{i}_std" for i in range(1, 21)],
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "chroma_stft_mean",
    "chroma_stft_std",
]


def extract_feature_vector(audio_path: Path, sr: int = 22050, duration: int = 30) -> np.ndarray:
    """Extract a compact feature vector for classical ML models."""
    y, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    if len(y) == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    vector = np.concatenate(
        [
            np.array([tempo], dtype=np.float32),
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.array([np.mean(spectral_centroid), np.std(spectral_centroid)]),
            np.array([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]),
            np.array([np.mean(chroma_stft), np.std(chroma_stft)]),
        ]
    )

    return vector.astype(np.float32)
