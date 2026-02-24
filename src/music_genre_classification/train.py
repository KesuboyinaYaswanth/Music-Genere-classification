from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from music_genre_classification.dataset import TrackRecord, build_manifest, load_manifest
from music_genre_classification.features import FEATURE_NAMES, extract_feature_vector


def load_or_build_manifest(dataset_root: Path, manifest_csv: Path | None) -> list[TrackRecord]:
    if manifest_csv and manifest_csv.exists():
        return load_manifest(manifest_csv)
    manifest = build_manifest(dataset_root, output_csv=manifest_csv)
    if not manifest:
        raise ValueError(f"No audio files found under {dataset_root}")
    return manifest


def build_feature_matrix(manifest: list[TrackRecord]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    for row in manifest:
        path = Path(row.filepath)
        try:
            features.append(extract_feature_vector(path))
            labels.append(row.genre)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not features:
        raise RuntimeError("Feature extraction failed for every file.")

    return np.vstack(features), np.array(labels)


def train_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Pipeline:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a music genre classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--output-model", type=Path, default=Path("models/genre_model.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_or_build_manifest(args.dataset_root, args.manifest)
    X, y = build_feature_matrix(manifest)
    print(f"Training on {X.shape[0]} tracks with {X.shape[1]} features")
    print(f"Feature count definition: {len(FEATURE_NAMES)}")

    model = train_model(X, y)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)
    print(f"Saved model to {args.output_model}")


if __name__ == "__main__":
    main()
