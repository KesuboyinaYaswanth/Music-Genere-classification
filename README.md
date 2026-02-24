# Music Genre Classification (Machine Learning Project)

This repository now contains a complete starter project for **music genre classification** using Python and classical machine learning.

## Project Goal
Given an audio track, predict its genre (for example: rock, jazz, classical, hiphop, etc.).

## Approach
The baseline pipeline uses:
1. **Feature extraction** from raw audio with `librosa`:
   - tempo
   - MFCC statistics
   - spectral centroid statistics
   - zero-crossing rate statistics
   - chroma statistics
2. **Supervised learning** with a `RandomForestClassifier` in a scikit-learn pipeline.
3. **Inference CLI** to predict genre on a new track.

## Recommended Dataset Structure
Put your dataset under `data/raw` in this format:

```text
data/raw/
  rock/
    track001.wav
    track002.wav
  jazz/
    track001.wav
  classical/
    track001.wav
```

> This project will scan subfolders as genre labels.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
PYTHONPATH=src python -m music_genre_classification.train \
  --dataset-root data/raw \
  --manifest data/manifest.csv \
  --output-model models/genre_model.joblib
```

## Predict
```bash
PYTHONPATH=src python -m music_genre_classification.predict path/to/new_song.wav \
  --model-path models/genre_model.joblib
```

## Next Improvements
- Add data augmentation (time-stretch, pitch-shift, noise).
- Try deep learning with spectrogram CNNs.
- Add cross-validation and hyperparameter tuning.
- Track experiments with MLflow or Weights & Biases.
