import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

class XGBoostSentimentAnalyzer:
    def __init__(self):
        self.model = joblib.load('model/xgb_sentiment_analysis_pipeline.joblib')
        self.label_encoder = joblib.load('model/label_encoder.joblib')
        print(type(self.model), type(self.label_encoder))
        
    # def __call__(self, sentiment: str):
    #     try:
    #         self.prediction = self.model.predict([sentiment])
    #         print(self.prediction)
    #         self.prediction = self.label_encoder.inverse_transform(self.prediction)
    #         return self.prediction
    #     except Exception as E:
    #         print(E)