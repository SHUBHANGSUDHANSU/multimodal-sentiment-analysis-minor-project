# Multimodal Sentiment Analysis Using Text, Audio, and Video

I have built a **Multimodal Sentiment Analysis** project that predicts a person's sentiment by using text, audio, and video information together.

The system is designed to classify sentiment by combining three modalities:

- **Text**: what the person says.
- **Audio**: how the person says it, such as tone and speech pattern.
- **Video**: visual cues such as face, movement, and expressions.

The main idea behind this project is that human sentiment is not always clear from text alone. The same sentence can express different emotions depending on the speaker's tone and facial expression. By using text, audio, and video together, the model can understand sentiment in a more complete way.

## Project Objective

The objective of this project is to build a complete machine learning/deep learning pipeline that can classify sentiment using multimodal input.

The notebook covers:

1. Dataset loading.
2. Text preprocessing.
3. Text feature extraction.
4. Audio feature extraction.
5. Video feature extraction.
6. Feature fusion.
7. Model training.
8. Model evaluation.
9. Final prediction.

## Current Status

The notebook supports two dataset modes:

| Mode | Dataset | Status |
|---|---|---|
| Primary mode | MELD raw dataset | Supports text, audio, and video |
| Fallback mode | `LabeledText.xlsx` | Text-only sentiment classification |

If the folder `MELD-RAW/MELD.Raw` is available locally, the notebook automatically uses MELD.

If MELD is not available, the notebook uses `LabeledText.xlsx` so the project still runs end-to-end.

## Dataset Used

### 1. MELD Raw Dataset

The main dataset supported by this project is **MELD: Multimodal EmotionLines Dataset**.

MELD contains:

- Text utterances.
- Video clips in `.mp4` format.
- Audio inside the `.mp4` clips.
- Sentiment labels.
- Emotion labels.
- Train, development, and test splits.

For sentiment classification, the labels are:

- `negative`
- `neutral`
- `positive`

For emotion classification, MELD also supports labels such as:

- `anger`
- `disgust`
- `fear`
- `joy`
- `neutral`
- `sadness`
- `surprise`

By default, this notebook performs **sentiment classification**.

### 2. LabeledText Fallback Dataset

The repository also contains:

```text
LabeledText.xlsx
```

This dataset contains:

- `File Name`
- `Caption`
- `LABEL`

This fallback dataset is text-only, so it is useful when MELD is not downloaded.

## MELD Folder Setup

The MELD raw dataset is large, so it is not uploaded to GitHub.

Keep the MELD folder locally in this structure:

```text
MELD-RAW/
  MELD.Raw/
    train/
      train_sent_emo.csv
      train_splits/
        dia0_utt0.mp4
        dia0_utt1.mp4
        ...
    dev_sent_emo.csv
    dev/
      dev_splits_complete/
        dia0_utt0.mp4
        dia0_utt1.mp4
        ...
    test_sent_emo.csv
    test/
      output_repeated_splits_test/
        dia0_utt0.mp4
        dia0_utt1.mp4
        ...
```

The notebook maps each CSV row to its video file using this pattern:

```text
dia{Dialogue_ID}_utt{Utterance_ID}.mp4
```

Example:

```text
Dialogue_ID = 0
Utterance_ID = 1
Video file = dia0_utt1.mp4
```

## Why MELD Is Not Uploaded to GitHub

The `MELD-RAW/` folder is ignored in `.gitignore` because it is very large, around several GBs.

GitHub is meant for source code, notebooks, and documentation. Large datasets should be downloaded separately from Kaggle or the official dataset source.

This repository includes the code needed to use MELD, but the dataset should be kept locally.

## Project Architecture

The project follows a pipeline-based architecture:

```text
MELD Dataset
   ↓
Text Utterance + Audio/Video Clip + Label
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Text Features: TF-IDF
Audio Features: MFCC
Video Features: Frame-level OpenCV features
   ↓
Early Fusion
   ↓
Dense Neural Network Classifier
   ↓
Sentiment Prediction
   ↓
Negative / Neutral / Positive
```

## Step-by-Step Pipeline

### 1. Data Loading

The notebook first checks whether the MELD raw dataset exists locally.

If MELD exists, it loads:

```text
train_sent_emo.csv
dev_sent_emo.csv
test_sent_emo.csv
```

Each row contains an utterance, speaker, sentiment label, emotion label, dialogue ID, utterance ID, and timestamp information.

### 2. Text Preprocessing

The text is cleaned before feature extraction.

The preprocessing steps include:

- Convert text to lowercase.
- Remove URLs.
- Remove mentions.
- Remove special characters.
- Replace noisy characters.
- Remove extra spaces.

This makes the input cleaner and easier for the model to learn from.

### 3. Text Feature Extraction

Text is converted into numerical features using **TF-IDF**.

TF-IDF stands for **Term Frequency-Inverse Document Frequency**.

It gives importance to useful words and reduces the importance of very common words.

Example:

- Words like `happy`, `angry`, `sad`, and `scared` may get higher importance.
- Words like `the`, `is`, and `and` get lower importance.

### 4. Audio Feature Extraction

Audio features are extracted from the `.mp4` clips using `librosa`.

The notebook extracts **MFCC features**.

MFCC stands for **Mel-Frequency Cepstral Coefficients**.

MFCC is useful for speech emotion analysis because it captures voice-related patterns such as:

- Tone
- Pitch
- Frequency
- Speaking style
- Energy

The notebook extracts 40 MFCC coefficients and takes the mean across time frames to create one fixed-size audio vector.

### 5. Video Feature Extraction

Video features are extracted from `.mp4` clips using OpenCV.

The notebook:

- Reads video frames.
- Resizes frames.
- Converts frames to grayscale.
- Converts each frame into a compact numerical vector.
- Takes the average across frames.

This creates a simple frame-level visual feature representation.

In a more advanced version, this can be replaced with CNN models such as:

- VGG16
- ResNet
- MobileNet
- Custom CNN

### 6. Feature Fusion

The project uses **early fusion**.

Early fusion means combining all features before classification.

```text
Fused Feature = Text Feature + Audio Feature + Video Feature
```

The fused feature vector is then normalized using standard scaling.

### 7. Model Training

The classifier is a dense neural network.

The model architecture is:

```text
Input Layer
   ↓
Dense Layer with ReLU
   ↓
Dropout
   ↓
Dense Layer with ReLU
   ↓
Dropout
   ↓
Softmax Output Layer
```

The output layer predicts one sentiment class:

- `negative`
- `neutral`
- `positive`

### 8. Model Evaluation

The notebook evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

These metrics help understand how well the model performs on unseen test data.

### 9. Prediction Function

The notebook includes a prediction function that accepts:

- Text
- Audio path
- Video path

For MELD, the same `.mp4` file is used as both the audio and video input because the `.mp4` contains both audio and visual data.

The function returns:

- Predicted sentiment label.
- Confidence score.
- Probability for each class.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Project development and explanation |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, scaling, metrics, train-test utilities |
| TensorFlow/Keras | Dense neural network classifier |
| Librosa | Audio feature extraction using MFCC |
| OpenCV | Video frame processing |
| Matplotlib | Training curve visualization |
| Seaborn | Confusion matrix visualization |
| FFmpeg | Helps read audio from `.mp4` files |

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For macOS, install FFmpeg:

```bash
brew install ffmpeg
```

If you do not have Homebrew, install FFmpeg from:

```text
https://ffmpeg.org/download.html
```

## How to Run the Project

1. Clone or download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MELD raw dataset from Kaggle or the official source.
4. Place the dataset in this structure:

```text
MELD-RAW/MELD.Raw/
```

5. Open the notebook:

```text
Multimodal_Sentiment_Analysis_Using_Text_Audio_Video.ipynb
```

6. Run all cells from top to bottom.

7. Check these output lines:

```text
Real audio files used: ...
Real video files used: ...
```

If both values are greater than `0`, real audio and video features are being extracted.

## Quick-Run Mode

The notebook uses quick-run mode by default.

MELD contains many video clips, and extracting features from every clip can take time. Quick-run mode samples a smaller balanced subset so the notebook can run faster on a normal laptop.

To use more data, change this setting inside the notebook:

```python
USE_QUICK_RUN = False
```

Or increase:

```python
QUICK_SAMPLES_PER_CLASS
```

## Feature Caching

The notebook stores extracted audio/video features in:

```text
.feature_cache/
```

This avoids extracting the same features again and again.

If you install new audio/video libraries and want to regenerate features, delete:

```text
.feature_cache/
```

Then rerun the notebook.

## What Works Now

The notebook can:

- Load the MELD raw dataset.
- Locate `.mp4` files for each utterance.
- Extract text features.
- Extract audio features if `librosa` and FFmpeg are installed.
- Extract video features if OpenCV is installed.
- Fuse text, audio, and video features.
- Train a classifier.
- Evaluate the model.
- Predict sentiment for new samples.

## Limitations

This is a beginner-friendly academic project, so some parts are intentionally simple.

Current limitations:

- Text uses TF-IDF, not BERT by default.
- Video uses simple frame-level features, not a pretrained CNN.
- Audio uses MFCC, not advanced speech models.
- Quick-run mode uses a subset of the full dataset.
- The project is not deployed as a web app yet.

## Future Improvements

The project can be improved by:

- Using BERT for text embeddings.
- Using wav2vec2 or openSMILE for audio features.
- Using ResNet, VGG16, or MobileNet for video features.
- Applying attention-based fusion.
- Training on the full MELD dataset.
- Adding face detection before video feature extraction.
- Deploying the model using Streamlit or FastAPI.

## Honest Answer If Asked About Audio and Video

If asked whether the project really supports audio and video, answer:

> Yes, the notebook supports real audio and video when the MELD raw dataset and required libraries are installed. The `.mp4` files are used for both audio and video. Audio features are extracted using MFCC with Librosa, and video features are extracted using OpenCV frame processing.

If asked why the MELD dataset is not uploaded to GitHub, answer:

> MELD raw data is very large, so it is not uploaded to GitHub. The GitHub repository contains the complete code and setup instructions, while the dataset is downloaded separately from Kaggle or the official MELD source.

## Repository Files

```text
.
├── Multimodal_Sentiment_Analysis_Using_Text_Audio_Video.ipynb
├── LabeledText.xlsx
├── requirements.txt
├── README.md
└── .gitignore
```

Local-only dataset folder:

```text
MELD-RAW/
```

This folder is ignored by Git because it is too large for GitHub.
