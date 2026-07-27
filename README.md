# Multimodal Sentiment Analysis Using Text, Audio, and Video

This repository contains a beginner-friendly Jupyter Notebook for a B.Tech minor project on multimodal sentiment analysis.

The notebook supports the MELD raw dataset for text, audio, and video analysis. If MELD is not available locally, it falls back to the included `LabeledText.xlsx` text dataset.

For MELD, the target sentiment labels are:

- `negative`
- `neutral`
- `positive`

## Project Pipeline

1. Load MELD train/dev/test CSV files.
2. Map each utterance to its `.mp4` clip.
3. Clean and preprocess text utterances.
4. Extract text features using TF-IDF.
5. Extract audio features using MFCC with `librosa`.
6. Extract video frame features using OpenCV.
7. Fuse text, audio, and video feature vectors.
8. Train a dense neural network classifier.
9. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
10. Predict sentiment for new samples.

## Files

- `Multimodal_Sentiment_Analysis_Using_Text_Audio_Video.ipynb`: Complete project notebook.
- `LabeledText.xlsx`: Dataset used by the notebook.
- `requirements.txt`: Python dependencies for full audio/video feature extraction.

## MELD Folder Setup

Keep the raw MELD folder locally like this:

```text
MELD-RAW/
  MELD.Raw/
    train/train_sent_emo.csv
    dev_sent_emo.csv
    test_sent_emo.csv
    train/train_splits/
    dev/dev_splits_complete/
    test/output_repeated_splits_test/
```

The `MELD-RAW/` folder is ignored by Git because it is too large to upload to GitHub.
