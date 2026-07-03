# Multimodal Sentiment Analysis Using Text, Audio, and Video

This repository contains a beginner-friendly Jupyter Notebook for a B.Tech minor project on multimodal sentiment analysis.

The notebook trains a sentiment classifier using the included `LabeledText.xlsx` dataset. The dataset contains text captions and labels:

- `negative`
- `neutral`
- `positive`

## Project Pipeline

1. Load the Excel dataset.
2. Clean and preprocess text captions.
3. Extract text features using TF-IDF.
4. Include audio MFCC and video frame-feature extraction functions.
5. Use zero-vector fallback for missing audio/video files.
6. Fuse text, audio, and video feature vectors.
7. Train a dense neural network classifier.
8. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
9. Predict sentiment for new samples.

## Files

- `Multimodal_Sentiment_Analysis_Using_Text_Audio_Video.ipynb`: Complete project notebook.
- `LabeledText.xlsx`: Dataset used by the notebook.

## Note

The included dataset is text-only, so the current model mainly learns from caption text. The notebook still includes audio and video feature extraction code so the project can be extended with real media files later.
