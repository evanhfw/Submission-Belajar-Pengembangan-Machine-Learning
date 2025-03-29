import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import InconsistentVersionWarning
from xgboost import XGBClassifier

import warnings

# Abaikan peringatan version mismatch dari scikit-learn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Abaikan peringatan dari XGBoost
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class XGBoostSentimentAnalyzer:
    def __init__(self):
        self.model = joblib.load('model/xgb_sentiment_analysis_pipeline.joblib')
        self.label_encoder = joblib.load('model/label_encoder.joblib')
        
    def __call__(self, sentiment: str):
        try:
            self.prediction = self.model.predict([sentiment])
            self.prediction = self.label_encoder.inverse_transform(self.prediction)
            return self.prediction[0]
        except Exception as E:
            print(E)