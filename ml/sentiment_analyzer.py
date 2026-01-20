"""
Sentiment Analyzer for AutoMind
===============================

A machine learning model to analyze sentiment in car reviews and opinions.
Uses TF-IDF vectorization + Logistic Regression for sentiment classification.

NLP/ML Concepts:
- Sentiment Analysis
- Text Classification
- Feature Engineering with TF-IDF
- Model Persistence
"""

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Path for model persistence
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_analyzer.joblib")


# Training data - car review sentiments
TRAINING_DATA = [
    # POSITIVE sentiments
    ("This car is absolutely amazing, best purchase ever", "positive"),
    ("I love the performance, handles like a dream", "positive"),
    ("Incredible value for money, highly recommend", "positive"),
    ("The interior quality is outstanding", "positive"),
    ("Best car I've ever owned, no complaints", "positive"),
    ("Fuel economy is excellent, saves me money", "positive"),
    ("Smooth ride, comfortable on long trips", "positive"),
    ("The acceleration is insane, so much fun", "positive"),
    ("Great reliability, never had any issues", "positive"),
    ("Beautiful design, gets compliments everywhere", "positive"),
    ("Perfect family car, spacious and safe", "positive"),
    ("Love the technology features, very modern", "positive"),
    ("Excellent build quality, feels premium", "positive"),
    ("The sound system is fantastic", "positive"),
    ("Great handling, corners like it's on rails", "positive"),
    ("Very happy with this purchase", "positive"),
    ("Exceeded all my expectations", "positive"),
    ("The engine is powerful and responsive", "positive"),
    ("Fantastic car for the price", "positive"),
    ("Would definitely buy again", "positive"),
    ("Super reliable and dependable", "positive"),
    ("The safety features are top notch", "positive"),
    ("Impressed with the fuel efficiency", "positive"),
    ("Comfortable seats, perfect for commuting", "positive"),
    ("The trunk space is huge", "positive"),
    ("Love the sporty look", "positive"),
    ("Drives beautifully in all conditions", "positive"),
    ("Best in class performance", "positive"),
    ("The warranty coverage is excellent", "positive"),
    ("Really enjoying this car", "positive"),
    ("Outstanding value proposition", "positive"),
    ("The infotainment system is intuitive", "positive"),
    ("Great visibility and easy to park", "positive"),
    ("The brakes are responsive and strong", "positive"),
    ("Perfect daily driver", "positive"),
    ("Five stars, couldn't be happier", "positive"),
    ("The resale value is excellent", "positive"),
    ("Quiet cabin, very refined", "positive"),
    ("The leather seats are luxurious", "positive"),
    ("Impressive torque and power delivery", "positive"),
    
    # NEGATIVE sentiments
    ("Terrible car, worst purchase ever", "negative"),
    ("Constant problems, always in the shop", "negative"),
    ("Very disappointed with the quality", "negative"),
    ("The fuel economy is awful", "negative"),
    ("Too many mechanical issues", "negative"),
    ("Uncomfortable seats, back pain after driving", "negative"),
    ("The engine is noisy and rough", "negative"),
    ("Poor build quality, things keep breaking", "negative"),
    ("Not worth the money at all", "negative"),
    ("Regret buying this car", "negative"),
    ("The transmission is jerky and unreliable", "negative"),
    ("Cheap interior materials", "negative"),
    ("Terrible customer service from dealer", "negative"),
    ("The car has been nothing but trouble", "negative"),
    ("Brake problems since day one", "negative"),
    ("Way overpriced for what you get", "negative"),
    ("The electronics keep failing", "negative"),
    ("Very poor visibility", "negative"),
    ("Rattles and squeaks everywhere", "negative"),
    ("The AC stopped working already", "negative"),
    ("Disappointed with performance", "negative"),
    ("Paint quality is terrible", "negative"),
    ("Had to replace the engine at 50k miles", "negative"),
    ("Worst reliability I've experienced", "negative"),
    ("The dealer couldn't fix the issues", "negative"),
    ("Unsafe car, failed safety inspection", "negative"),
    ("The suspension is too harsh", "negative"),
    ("Electrical problems constantly", "negative"),
    ("Would never recommend this car", "negative"),
    ("Complete waste of money", "negative"),
    ("The infotainment system is buggy", "negative"),
    ("Poor handling, feels unstable", "negative"),
    ("The seats are falling apart", "negative"),
    ("Horrible experience overall", "negative"),
    ("Stay away from this model", "negative"),
    ("The trunk is way too small", "negative"),
    ("Engine overheating issues", "negative"),
    ("Awful gas mileage for its class", "negative"),
    ("The steering feels disconnected", "negative"),
    ("Multiple recalls already", "negative"),
    
    # NEUTRAL sentiments
    ("It's an okay car, nothing special", "neutral"),
    ("Average performance for the class", "neutral"),
    ("It gets the job done", "neutral"),
    ("Some pros and cons to consider", "neutral"),
    ("Decent car for the price point", "neutral"),
    ("Not bad but not great either", "neutral"),
    ("It meets my basic needs", "neutral"),
    ("The car is acceptable overall", "neutral"),
    ("Middle of the road experience", "neutral"),
    ("Neither impressed nor disappointed", "neutral"),
    ("Standard features for this segment", "neutral"),
    ("Adequate for daily commuting", "neutral"),
    ("Nothing to complain about really", "neutral"),
    ("It's a typical family sedan", "neutral"),
    ("Fair value for the money", "neutral"),
    ("Does what it's supposed to do", "neutral"),
    ("Average fuel economy", "neutral"),
    ("The ride quality is acceptable", "neutral"),
    ("Basic transportation, nothing more", "neutral"),
    ("It's a compromise in many ways", "neutral"),
    ("Some things I like, some I don't", "neutral"),
    ("Reasonably comfortable", "neutral"),
    ("Standard warranty coverage", "neutral"),
    ("Typical interior for the price", "neutral"),
    ("It works for my purposes", "neutral"),
    ("Not the best, not the worst", "neutral"),
    ("Moderately satisfied with purchase", "neutral"),
    ("Expected performance levels", "neutral"),
    ("Regular maintenance required", "neutral"),
    ("It's exactly what I expected", "neutral"),
]


class SentimentAnalyzer:
    """
    ML-based Sentiment Analyzer for car reviews.
    
    Uses TF-IDF + Logistic Regression pipeline.
    Classifies text as: positive, negative, or neutral
    """
    
    SENTIMENTS = ["positive", "negative", "neutral"]
    
    def __init__(self):
        """Initialize and load or train the model."""
        self.model = None
        self.load_or_train()
    
    def load_or_train(self):
        """Load existing model or train a new one."""
        if os.path.exists(SENTIMENT_MODEL_PATH):
            try:
                self.model = joblib.load(SENTIMENT_MODEL_PATH)
                print("[SentimentAnalyzer] Model loaded from disk")
                return
            except Exception as e:
                print(f"[SentimentAnalyzer] Error loading model: {e}")
        
        # Train new model
        self.train()
    
    def train(self):
        """Train the sentiment analysis model."""
        print("[SentimentAnalyzer] Training new model...")
        
        # Prepare data
        texts = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=3000
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ))
        ])
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[SentimentAnalyzer] Model accuracy: {accuracy:.2%}")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, SENTIMENT_MODEL_PATH)
        print(f"[SentimentAnalyzer] Model saved to {SENTIMENT_MODEL_PATH}")
    
    def analyze(self, text: str) -> dict:
        """
        Analyze the sentiment of a text.
        
        Returns:
            dict with 'sentiment', 'confidence', 'emoji', and all scores
        """
        if self.model is None:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Get prediction and probabilities
        sentiment = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        confidence = float(max(probs))
        
        # Sentiment to emoji mapping
        emoji_map = {
            "positive": "😊",
            "negative": "😞",
            "neutral": "😐"
        }
        
        # Get all probabilities
        all_sentiments = {
            label: float(prob) 
            for label, prob in zip(self.model.classes_, probs)
        }
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "emoji": emoji_map.get(sentiment, "🤔"),
            "scores": all_sentiments
        }
    
    def analyze_batch(self, texts: list) -> list:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_summary(self, reviews: list) -> dict:
        """
        Get an overall sentiment summary for multiple reviews.
        
        Returns:
            Summary statistics for the reviews
        """
        results = self.analyze_batch(reviews)
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            sentiment_counts[r["sentiment"]] += 1
        
        total = len(reviews)
        return {
            "total_reviews": total,
            "positive_count": sentiment_counts["positive"],
            "negative_count": sentiment_counts["negative"],
            "neutral_count": sentiment_counts["neutral"],
            "positive_percent": sentiment_counts["positive"] / total * 100 if total > 0 else 0,
            "negative_percent": sentiment_counts["negative"] / total * 100 if total > 0 else 0,
            "neutral_percent": sentiment_counts["neutral"] / total * 100 if total > 0 else 0,
            "overall": max(sentiment_counts, key=sentiment_counts.get)
        }


# Singleton instance
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the sentiment analyzer singleton."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


if __name__ == "__main__":
    # Test the analyzer
    analyzer = SentimentAnalyzer()
    
    test_reviews = [
        "Amazing car, absolutely love it! Best purchase ever.",
        "Terrible experience, constant problems and poor service.",
        "It's an okay car, does what I need it to do.",
        "The performance is incredible, handles like a dream!",
        "Worst reliability I've ever experienced, avoid this car.",
    ]
    
    print("\n" + "="*50)
    print("Testing Sentiment Analyzer")
    print("="*50)
    
    for review in test_reviews:
        result = analyzer.analyze(review)
        print(f"\n'{review[:50]}...'")
        print(f"  → {result['emoji']} {result['sentiment'].upper()} ({result['confidence']:.2%})")
