"""
AutoMind ML Module
==================

Custom trained ML/NLP models for the car expert AI:

1. Intent Classifier - Classifies user query intents (TF-IDF + Logistic Regression)
2. Sentiment Analyzer - Analyzes sentiment in car reviews (Text Classification)
3. Car NER - Extracts car entities (spaCy Custom NER)
4. Price Predictor - Predicts car prices (Random Forest Regression)

All models are trained on car-domain specific data for optimal performance.
"""

from .intent_classifier import IntentClassifier, get_intent_classifier
from .sentiment_analyzer import SentimentAnalyzer, get_sentiment_analyzer
from .car_ner import CarEntityRecognizer, get_car_ner
from .price_predictor import CarPricePredictor, get_price_predictor

__all__ = [
    "IntentClassifier",
    "get_intent_classifier",
    "SentimentAnalyzer", 
    "get_sentiment_analyzer",
    "CarEntityRecognizer",
    "get_car_ner",
    "CarPricePredictor",
    "get_price_predictor",
]


def initialize_all_models():
    """
    Initialize all ML models.
    Call this at app startup to warm up the models.
    """
    print("🤖 Initializing AutoMind ML Models...")
    
    models = {
        "intent_classifier": get_intent_classifier(),
        "sentiment_analyzer": get_sentiment_analyzer(),
        "car_ner": get_car_ner(),
        "price_predictor": get_price_predictor(),
    }
    
    print("✅ All ML models initialized successfully!")
    return models
