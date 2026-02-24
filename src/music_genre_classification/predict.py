from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from music_genre_classification.features import extract_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict music genre from an audio file.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/genre_model.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    features = extract_feature_vector(args.audio_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print(f"Predicted genre: {prediction}")
    print("Top classes:")
    for cls, prob in sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
