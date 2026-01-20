"""
Intent Classifier for AutoMind
==============================

A machine learning model trained to classify user intents for car-related queries.
Uses TF-IDF vectorization + Logistic Regression/SVM for classification.

NLP/ML Concepts:
- Text Vectorization (TF-IDF)
- Multi-class Classification
- Model Persistence (joblib)
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
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.joblib")


# Training data - labeled examples for each intent
TRAINING_DATA = [
    # SPECS intent - asking about specifications
    ("what are the specs of the toyota supra", "specs"),
    ("tell me the horsepower of bmw m3", "specs"),
    ("how much horsepower does the mustang have", "specs"),
    ("what's the 0-60 time for tesla model s", "specs"),
    ("what engine does the corvette have", "specs"),
    ("how fast is the porsche 911", "specs"),
    ("what's the top speed of ferrari 488", "specs"),
    ("tell me about the toyota camry specifications", "specs"),
    ("what's the fuel economy of honda accord", "specs"),
    ("how many miles per gallon does the prius get", "specs"),
    ("what transmission does the civic type r have", "specs"),
    ("torque of the dodge challenger", "specs"),
    ("weight of the mazda miata", "specs"),
    ("how big is the trunk in the model 3", "specs"),
    ("what's the range of the electric mustang", "specs"),
    ("tell me the specs", "specs"),
    ("specifications of audi r8", "specs"),
    ("what are the features of lexus lc", "specs"),
    ("performance specs of nissan gtr", "specs"),
    ("how powerful is the hellcat", "specs"),
    
    # COMPARISON intent - comparing cars
    ("compare bmw m3 vs mercedes c63", "comparison"),
    ("which is better mustang or camaro", "comparison"),
    ("civic vs corolla which should i buy", "comparison"),
    ("difference between audi a4 and bmw 3 series", "comparison"),
    ("tesla model 3 vs model y comparison", "comparison"),
    ("is the supra better than the z", "comparison"),
    ("compare sports cars under 50k", "comparison"),
    ("porsche vs ferrari which is faster", "comparison"),
    ("rav4 or crv which is more reliable", "comparison"),
    ("mustang gt vs camaro ss head to head", "comparison"),
    ("what's the difference between", "comparison"),
    ("compare the two cars", "comparison"),
    ("which one should i get", "comparison"),
    ("better value mustang or challenger", "comparison"),
    ("bmw vs audi vs mercedes", "comparison"),
    ("compare suv options", "comparison"),
    ("which truck is best f150 or silverado", "comparison"),
    ("tacoma vs ranger comparison", "comparison"),
    ("model s or taycan", "comparison"),
    ("should i buy the civic or the mazda 3", "comparison"),
    
    # PRICING intent - asking about prices
    ("how much does the toyota supra cost", "pricing"),
    ("what's the price of a new mustang", "pricing"),
    ("msrp of the bmw m4", "pricing"),
    ("how expensive is a ferrari", "pricing"),
    ("what's the starting price of tesla", "pricing"),
    ("can i afford a porsche", "pricing"),
    ("budget sports cars under 30k", "pricing"),
    ("how much is a used honda civic", "pricing"),
    ("price range for luxury sedans", "pricing"),
    ("what's a good deal on a camry", "pricing"),
    ("cost of ownership for bmw", "pricing"),
    ("insurance cost for mustang gt", "pricing"),
    ("depreciation of mercedes", "pricing"),
    ("is the miata worth the money", "pricing"),
    ("lease vs buy calculator", "pricing"),
    ("monthly payment for 40k car", "pricing"),
    ("total cost of buying a new car", "pricing"),
    ("what's the invoice price", "pricing"),
    ("dealer markup on bronco", "pricing"),
    ("best value cars 2024", "pricing"),
    
    # MAINTENANCE intent - reliability, maintenance, problems
    ("is the toyota camry reliable", "maintenance"),
    ("common problems with bmw", "maintenance"),
    ("how often do i change oil in honda", "maintenance"),
    ("maintenance schedule for ford f150", "maintenance"),
    ("reliability rating of lexus", "maintenance"),
    ("what are common issues with mustang", "maintenance"),
    ("how long do toyota engines last", "maintenance"),
    ("service interval for mercedes", "maintenance"),
    ("is bmw expensive to maintain", "maintenance"),
    ("repair costs for audi", "maintenance"),
    ("best reliable cars", "maintenance"),
    ("problems with hyundai", "maintenance"),
    ("recalls on ford explorer", "maintenance"),
    ("brake pad replacement cost", "maintenance"),
    ("how reliable is tesla", "maintenance"),
    ("transmission problems in cvt", "maintenance"),
    ("engine issues with ecoboost", "maintenance"),
    ("timing belt replacement", "maintenance"),
    ("maintenance tips for high mileage", "maintenance"),
    ("when to replace spark plugs", "maintenance"),
    
    # IMAGES intent - asking for pictures/images
    ("show me pictures of lamborghini", "images"),
    ("images of the new mustang", "images"),
    ("what does the supra look like", "images"),
    ("show me photos of ferrari", "images"),
    ("i want to see the bmw m3", "images"),
    ("pictures of corvette c8", "images"),
    ("show me the interior of model s", "images"),
    ("what does a gtr look like", "images"),
    ("images of classic mustang", "images"),
    ("show me red ferraris", "images"),
    ("photos of supercars", "images"),
    ("can i see pictures", "images"),
    ("show me what it looks like", "images"),
    ("display images of porsche 911", "images"),
    ("i want to see photos", "images"),
    ("visual of the new corvette", "images"),
    ("gallery of lamborghini aventador", "images"),
    ("show the exterior design", "images"),
    ("pics of the interior", "images"),
    ("what color options available", "images"),
    
    # VIDEOS intent - asking for video content
    ("show me videos of tesla plaid", "videos"),
    ("youtube review of mustang", "videos"),
    ("watch corvette c8 review", "videos"),
    ("video of lamborghini acceleration", "videos"),
    ("drag race videos", "videos"),
    ("show me test drive videos", "videos"),
    ("car review videos", "videos"),
    ("walkaround video of porsche", "videos"),
    ("i want to watch reviews", "videos"),
    ("youtube videos about bmw m3", "videos"),
    ("show me exhaust sound videos", "videos"),
    ("track test video", "videos"),
    ("top gear review video", "videos"),
    ("carwow drag race", "videos"),
    ("show me the launch video", "videos"),
    ("video comparison", "videos"),
    ("watch the review", "videos"),
    ("show me some videos", "videos"),
    ("youtube content about cars", "videos"),
    ("video tour of the car", "videos"),
    
    # HISTORY intent - car history, heritage
    ("history of ford mustang", "history"),
    ("when was the corvette first made", "history"),
    ("tell me about porsche heritage", "history"),
    ("evolution of the bmw m3", "history"),
    ("how old is the toyota supra", "history"),
    ("when did ferrari start", "history"),
    ("history of japanese sports cars", "history"),
    ("origins of lamborghini", "history"),
    ("classic car history", "history"),
    ("vintage mustang information", "history"),
    ("muscle car era", "history"),
    ("when was the first electric car", "history"),
    ("heritage of mercedes benz", "history"),
    ("story behind the gtr", "history"),
    ("who founded porsche", "history"),
    ("history of v8 engines", "history"),
    ("evolution of car design", "history"),
    ("when did turbo cars start", "history"),
    ("origin of the corvette name", "history"),
    ("classic ferrari models", "history"),
    
    # RECOMMENDATION intent - asking for suggestions
    ("what car should i buy", "recommendation"),
    ("recommend me a sports car", "recommendation"),
    ("best car for beginners", "recommendation"),
    ("suggest a reliable suv", "recommendation"),
    ("what's a good first car", "recommendation"),
    ("recommend something fun to drive", "recommendation"),
    ("best car for daily driving", "recommendation"),
    ("what should i get for family", "recommendation"),
    ("suggest cars under 30k", "recommendation"),
    ("best sports car for the money", "recommendation"),
    ("recommend a luxury sedan", "recommendation"),
    ("what's good for commuting", "recommendation"),
    ("suggest an electric car", "recommendation"),
    ("best truck for towing", "recommendation"),
    ("recommend a fast car", "recommendation"),
    ("what car suits my needs", "recommendation"),
    ("give me some options", "recommendation"),
    ("what would you recommend", "recommendation"),
    ("suggest something similar", "recommendation"),
    ("best picks for 2024", "recommendation"),
    
    # GENERAL intent - general questions, greetings
    ("hello", "general"),
    ("hi there", "general"),
    ("hey", "general"),
    ("what can you do", "general"),
    ("help me", "general"),
    ("thanks", "general"),
    ("thank you", "general"),
    ("goodbye", "general"),
    ("who are you", "general"),
    ("tell me about yourself", "general"),
    ("what is automind", "general"),
    ("how do you work", "general"),
    ("can you help me", "general"),
    ("i have a question", "general"),
    ("i need help", "general"),
    ("hello automind", "general"),
    ("good morning", "general"),
    ("what's up", "general"),
    ("how are you", "general"),
    ("nice to meet you", "general"),
]


class IntentClassifier:
    """
    ML-based Intent Classifier for car queries.
    
    Uses TF-IDF + Logistic Regression pipeline.
    """
    
    INTENTS = ["specs", "comparison", "pricing", "maintenance", 
               "images", "videos", "history", "recommendation", "general"]
    
    def __init__(self):
        """Initialize and load or train the model."""
        self.model = None
        self.load_or_train()
    
    def load_or_train(self):
        """Load existing model or train a new one."""
        if os.path.exists(INTENT_MODEL_PATH):
            try:
                self.model = joblib.load(INTENT_MODEL_PATH)
                print("[IntentClassifier] Model loaded from disk")
                return
            except Exception as e:
                print(f"[IntentClassifier] Error loading model: {e}")
        
        # Train new model
        self.train()
    
    def train(self):
        """Train the intent classification model."""
        print("[IntentClassifier] Training new model...")
        
        # Prepare data
        texts = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]
        
        # Create pipeline: TF-IDF + Logistic Regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                max_features=5000
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
        print(f"[IntentClassifier] Model accuracy: {accuracy:.2%}")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, INTENT_MODEL_PATH)
        print(f"[IntentClassifier] Model saved to {INTENT_MODEL_PATH}")
    
    def predict(self, text: str) -> dict:
        """
        Predict the intent of a text query.
        
        Returns:
            dict with 'intent' and 'confidence'
        """
        if self.model is None:
            return {"intent": "general", "confidence": 0.0}
        
        # Get prediction and probabilities
        intent = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        confidence = float(max(probs))
        
        # Get all probabilities
        all_intents = {
            label: float(prob) 
            for label, prob in zip(self.model.classes_, probs)
        }
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_intents": all_intents
        }
    
    def retrain(self, additional_data: list = None):
        """Retrain the model with optional additional data."""
        if additional_data:
            global TRAINING_DATA
            TRAINING_DATA.extend(additional_data)
        
        self.train()


# Singleton instance
_intent_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """Get or create the intent classifier singleton."""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()
    
    test_queries = [
        "what's the horsepower of the new mustang",
        "compare tesla model 3 vs model y",
        "how much does a bmw cost",
        "is toyota reliable",
        "show me pictures of ferrari",
        "youtube review of corvette",
        "history of porsche",
        "recommend me a sports car",
        "hello there",
    ]
    
    print("\n" + "="*50)
    print("Testing Intent Classifier")
    print("="*50)
    
    for query in test_queries:
        result = classifier.predict(query)
        print(f"\n'{query}'")
        print(f"  → Intent: {result['intent']} ({result['confidence']:.2%})")
