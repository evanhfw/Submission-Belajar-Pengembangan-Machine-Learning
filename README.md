# Indonesian Sentiment Analysis

A machine learning project that analyzes sentiment in Indonesian text using XGBoost and transformer-based models. The project provides tools to classify text as positive, negative, or neutral.

## Overview

This project implements a sentiment analysis system that:

- Scrapes reviews from Google Play Store (specifically from the MyPertamina app)
- Processes and labels Indonesian text data
- Trains an XGBoost model using TF-IDF feature extraction
- Implements an easy-to-use inference interface

## Features

- Text sentiment classification (positive, negative, neutral)
- Pre-processed data using TF-IDF vectorization
- Optimized XGBoost classifier with Bayesian hyperparameter tuning
- Simple API for sentiment prediction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/indonesian-sentiment-analysis.git
cd indonesian-sentiment-analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure you have the model files in the `model/` directory:
   - `xgb_sentiment_analysis_pipeline.joblib`
   - `label_encoder.joblib`

## Usage

### Quick Start

```python
from sentiment_analyzer.XGBoostSentimentAnalyzer import XGBoostSentimentAnalyzer

# Initialize the sentiment analyzer
sentiment_analyzer = XGBoostSentimentAnalyzer()

# Predict sentiment for a text
text = "aplikasi ini sangat bagus"
prediction = sentiment_analyzer(text)
print(f"Sentiment: {prediction}")
```

### Using the Inference Notebook

The project includes a Jupyter notebook for easy inference:

1. Open `notebook_inference.ipynb`
2. Change the `TEXT` variable to your input text
3. Run the notebook cells to get a sentiment prediction

## Dataset

The model was trained on reviews scraped from the MyPertamina app on Google Play Store:

- 30,000 reviews in Indonesian language
- Distribution: ~74% negative, ~24% positive, ~2% neutral
- Initial labeling performed using a pre-trained multilingual sentiment model

## Model Details

### Feature Extraction

- TF-IDF Vectorizer
- n-gram range: (1, 2)
- No IDF weighting

### Classifier

- XGBoost with optimized hyperparameters:
  - Learning rate: 0.087
  - Max depth: 8
  - Number of estimators: 600
  - Subsample: 0.763
  - Colsample bytree: 0.594
  - Gamma: 2.46e-09

### Performance

The model was evaluated using an 80/20 train/test split with stratification.

## Project Structure

```
├── model/                   # Saved model files
├── notebook_inference.ipynb # Inference notebook
├── notebook_training_model.ipynb # Training notebook
├── sentiment_analyzer/      # Python package
│   └── XGBoostSentimentAnalyzer.py # Sentiment analyzer class
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## License

[Include your license information here]

## Contributing

[Include contribution guidelines here]

## Acknowledgments

- Initial sentiment labeling performed using `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- This project was built with scikit-learn, XGBoost, and other open-source libraries
