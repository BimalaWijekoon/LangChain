"""
Car Price Predictor for AutoMind
================================

A machine learning model to predict car prices based on:
- Make and Model
- Year
- Mileage
- Condition
- Engine specs

ML Concepts:
- Feature Engineering (categorical encoding, normalization)
- Ensemble Methods (Random Forest, XGBoost)
- Model Evaluation (RMSE, R², MAE)
- Feature Importance Analysis
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Path for model persistence
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "price_predictor.joblib")


# Training data - Realistic car listings with prices
TRAINING_DATA = [
    # Toyota
    {"make": "Toyota", "model": "Camry", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 203, "mpg": 32, "body_type": "sedan", "price": 28500},
    {"make": "Toyota", "model": "Camry", "year": 2023, "mileage": 15000, "condition": "excellent", "engine_hp": 203, "mpg": 32, "body_type": "sedan", "price": 24800},
    {"make": "Toyota", "model": "Camry", "year": 2022, "mileage": 28000, "condition": "good", "engine_hp": 203, "mpg": 32, "body_type": "sedan", "price": 22500},
    {"make": "Toyota", "model": "Camry", "year": 2020, "mileage": 45000, "condition": "good", "engine_hp": 203, "mpg": 32, "body_type": "sedan", "price": 19500},
    {"make": "Toyota", "model": "Camry", "year": 2018, "mileage": 65000, "condition": "fair", "engine_hp": 203, "mpg": 32, "body_type": "sedan", "price": 15800},
    
    {"make": "Toyota", "model": "Corolla", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 169, "mpg": 35, "body_type": "sedan", "price": 23500},
    {"make": "Toyota", "model": "Corolla", "year": 2022, "mileage": 22000, "condition": "excellent", "engine_hp": 169, "mpg": 35, "body_type": "sedan", "price": 19800},
    {"make": "Toyota", "model": "Corolla", "year": 2019, "mileage": 55000, "condition": "good", "engine_hp": 169, "mpg": 35, "body_type": "sedan", "price": 15500},
    
    {"make": "Toyota", "model": "Supra", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 382, "mpg": 25, "body_type": "coupe", "price": 56500},
    {"make": "Toyota", "model": "Supra", "year": 2023, "mileage": 5000, "condition": "excellent", "engine_hp": 382, "mpg": 25, "body_type": "coupe", "price": 51000},
    {"make": "Toyota", "model": "Supra", "year": 2021, "mileage": 20000, "condition": "good", "engine_hp": 382, "mpg": 25, "body_type": "coupe", "price": 45000},
    
    {"make": "Toyota", "model": "RAV4", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 203, "mpg": 30, "body_type": "suv", "price": 32500},
    {"make": "Toyota", "model": "RAV4", "year": 2022, "mileage": 25000, "condition": "excellent", "engine_hp": 203, "mpg": 30, "body_type": "suv", "price": 28000},
    
    # Honda
    {"make": "Honda", "model": "Civic", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 180, "mpg": 36, "body_type": "sedan", "price": 26500},
    {"make": "Honda", "model": "Civic", "year": 2023, "mileage": 12000, "condition": "excellent", "engine_hp": 180, "mpg": 36, "body_type": "sedan", "price": 23500},
    {"make": "Honda", "model": "Civic", "year": 2021, "mileage": 35000, "condition": "good", "engine_hp": 180, "mpg": 36, "body_type": "sedan", "price": 20500},
    {"make": "Honda", "model": "Civic", "year": 2018, "mileage": 70000, "condition": "fair", "engine_hp": 180, "mpg": 36, "body_type": "sedan", "price": 14500},
    
    {"make": "Honda", "model": "Civic Type R", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 315, "mpg": 28, "body_type": "hatchback", "price": 45000},
    {"make": "Honda", "model": "Civic Type R", "year": 2023, "mileage": 8000, "condition": "excellent", "engine_hp": 315, "mpg": 28, "body_type": "hatchback", "price": 42000},
    
    {"make": "Honda", "model": "Accord", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 192, "mpg": 33, "body_type": "sedan", "price": 30500},
    {"make": "Honda", "model": "Accord", "year": 2022, "mileage": 20000, "condition": "excellent", "engine_hp": 192, "mpg": 33, "body_type": "sedan", "price": 26500},
    
    # Ford
    {"make": "Ford", "model": "Mustang GT", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 480, "mpg": 21, "body_type": "coupe", "price": 46000},
    {"make": "Ford", "model": "Mustang GT", "year": 2023, "mileage": 8000, "condition": "excellent", "engine_hp": 480, "mpg": 21, "body_type": "coupe", "price": 42500},
    {"make": "Ford", "model": "Mustang GT", "year": 2021, "mileage": 25000, "condition": "good", "engine_hp": 460, "mpg": 21, "body_type": "coupe", "price": 36000},
    {"make": "Ford", "model": "Mustang GT", "year": 2019, "mileage": 45000, "condition": "good", "engine_hp": 460, "mpg": 21, "body_type": "coupe", "price": 31000},
    
    {"make": "Ford", "model": "F-150", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 400, "mpg": 22, "body_type": "truck", "price": 55000},
    {"make": "Ford", "model": "F-150", "year": 2022, "mileage": 30000, "condition": "good", "engine_hp": 400, "mpg": 22, "body_type": "truck", "price": 45000},
    {"make": "Ford", "model": "F-150", "year": 2020, "mileage": 55000, "condition": "fair", "engine_hp": 395, "mpg": 22, "body_type": "truck", "price": 35000},
    
    # Chevrolet
    {"make": "Chevrolet", "model": "Corvette", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 490, "mpg": 19, "body_type": "coupe", "price": 68000},
    {"make": "Chevrolet", "model": "Corvette", "year": 2023, "mileage": 5000, "condition": "excellent", "engine_hp": 490, "mpg": 19, "body_type": "coupe", "price": 64000},
    {"make": "Chevrolet", "model": "Corvette", "year": 2021, "mileage": 18000, "condition": "good", "engine_hp": 490, "mpg": 19, "body_type": "coupe", "price": 55000},
    
    {"make": "Chevrolet", "model": "Camaro SS", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 455, "mpg": 20, "body_type": "coupe", "price": 48000},
    {"make": "Chevrolet", "model": "Camaro SS", "year": 2022, "mileage": 15000, "condition": "excellent", "engine_hp": 455, "mpg": 20, "body_type": "coupe", "price": 42000},
    
    # BMW
    {"make": "BMW", "model": "M3", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 473, "mpg": 23, "body_type": "sedan", "price": 78000},
    {"make": "BMW", "model": "M3", "year": 2023, "mileage": 8000, "condition": "excellent", "engine_hp": 473, "mpg": 23, "body_type": "sedan", "price": 72000},
    {"make": "BMW", "model": "M3", "year": 2021, "mileage": 22000, "condition": "good", "engine_hp": 473, "mpg": 23, "body_type": "sedan", "price": 62000},
    
    {"make": "BMW", "model": "M4", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 503, "mpg": 22, "body_type": "coupe", "price": 82000},
    {"make": "BMW", "model": "M4", "year": 2022, "mileage": 12000, "condition": "excellent", "engine_hp": 503, "mpg": 22, "body_type": "coupe", "price": 74000},
    
    {"make": "BMW", "model": "3 Series", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 255, "mpg": 30, "body_type": "sedan", "price": 48000},
    {"make": "BMW", "model": "3 Series", "year": 2022, "mileage": 20000, "condition": "excellent", "engine_hp": 255, "mpg": 30, "body_type": "sedan", "price": 42000},
    
    # Mercedes
    {"make": "Mercedes", "model": "C63 AMG", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 469, "mpg": 21, "body_type": "sedan", "price": 85000},
    {"make": "Mercedes", "model": "C63 AMG", "year": 2022, "mileage": 15000, "condition": "excellent", "engine_hp": 469, "mpg": 21, "body_type": "sedan", "price": 75000},
    
    {"make": "Mercedes", "model": "E-Class", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 255, "mpg": 28, "body_type": "sedan", "price": 62000},
    {"make": "Mercedes", "model": "E-Class", "year": 2021, "mileage": 30000, "condition": "good", "engine_hp": 255, "mpg": 28, "body_type": "sedan", "price": 48000},
    
    # Porsche
    {"make": "Porsche", "model": "911", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 379, "mpg": 24, "body_type": "coupe", "price": 115000},
    {"make": "Porsche", "model": "911", "year": 2023, "mileage": 5000, "condition": "excellent", "engine_hp": 379, "mpg": 24, "body_type": "coupe", "price": 108000},
    {"make": "Porsche", "model": "911", "year": 2021, "mileage": 15000, "condition": "good", "engine_hp": 379, "mpg": 24, "body_type": "coupe", "price": 95000},
    
    {"make": "Porsche", "model": "Cayenne", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 348, "mpg": 21, "body_type": "suv", "price": 82000},
    {"make": "Porsche", "model": "Cayenne", "year": 2022, "mileage": 20000, "condition": "excellent", "engine_hp": 348, "mpg": 21, "body_type": "suv", "price": 72000},
    
    # Tesla
    {"make": "Tesla", "model": "Model S", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 670, "mpg": 120, "body_type": "sedan", "price": 85000},
    {"make": "Tesla", "model": "Model S", "year": 2023, "mileage": 10000, "condition": "excellent", "engine_hp": 670, "mpg": 120, "body_type": "sedan", "price": 78000},
    {"make": "Tesla", "model": "Model S", "year": 2021, "mileage": 30000, "condition": "good", "engine_hp": 670, "mpg": 120, "body_type": "sedan", "price": 62000},
    
    {"make": "Tesla", "model": "Model 3", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 283, "mpg": 132, "body_type": "sedan", "price": 42000},
    {"make": "Tesla", "model": "Model 3", "year": 2022, "mileage": 25000, "condition": "excellent", "engine_hp": 283, "mpg": 132, "body_type": "sedan", "price": 35000},
    
    {"make": "Tesla", "model": "Model Y", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 384, "mpg": 120, "body_type": "suv", "price": 48000},
    {"make": "Tesla", "model": "Model Y", "year": 2023, "mileage": 15000, "condition": "excellent", "engine_hp": 384, "mpg": 120, "body_type": "suv", "price": 43000},
    
    # Dodge
    {"make": "Dodge", "model": "Challenger Hellcat", "year": 2023, "mileage": 0, "condition": "new", "engine_hp": 717, "mpg": 18, "body_type": "coupe", "price": 72000},
    {"make": "Dodge", "model": "Challenger Hellcat", "year": 2021, "mileage": 20000, "condition": "excellent", "engine_hp": 717, "mpg": 18, "body_type": "coupe", "price": 62000},
    
    {"make": "Dodge", "model": "Charger", "year": 2023, "mileage": 0, "condition": "new", "engine_hp": 370, "mpg": 22, "body_type": "sedan", "price": 38000},
    {"make": "Dodge", "model": "Charger", "year": 2021, "mileage": 30000, "condition": "good", "engine_hp": 370, "mpg": 22, "body_type": "sedan", "price": 30000},
    
    # Nissan
    {"make": "Nissan", "model": "GT-R", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 565, "mpg": 19, "body_type": "coupe", "price": 120000},
    {"make": "Nissan", "model": "GT-R", "year": 2022, "mileage": 10000, "condition": "excellent", "engine_hp": 565, "mpg": 19, "body_type": "coupe", "price": 105000},
    
    {"make": "Nissan", "model": "370Z", "year": 2020, "mileage": 25000, "condition": "good", "engine_hp": 332, "mpg": 24, "body_type": "coupe", "price": 32000},
    {"make": "Nissan", "model": "Z", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 400, "mpg": 24, "body_type": "coupe", "price": 52000},
    
    # Subaru
    {"make": "Subaru", "model": "WRX", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 271, "mpg": 28, "body_type": "sedan", "price": 32000},
    {"make": "Subaru", "model": "WRX", "year": 2022, "mileage": 20000, "condition": "excellent", "engine_hp": 271, "mpg": 28, "body_type": "sedan", "price": 28000},
    {"make": "Subaru", "model": "WRX", "year": 2020, "mileage": 40000, "condition": "good", "engine_hp": 268, "mpg": 28, "body_type": "sedan", "price": 24000},
    
    # Mazda
    {"make": "Mazda", "model": "MX-5 Miata", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 181, "mpg": 30, "body_type": "convertible", "price": 30000},
    {"make": "Mazda", "model": "MX-5 Miata", "year": 2022, "mileage": 15000, "condition": "excellent", "engine_hp": 181, "mpg": 30, "body_type": "convertible", "price": 26500},
    
    {"make": "Mazda", "model": "Mazda3", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 191, "mpg": 32, "body_type": "hatchback", "price": 26000},
    
    # Audi
    {"make": "Audi", "model": "RS6", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 621, "mpg": 20, "body_type": "wagon", "price": 125000},
    {"make": "Audi", "model": "RS6", "year": 2022, "mileage": 15000, "condition": "excellent", "engine_hp": 591, "mpg": 20, "body_type": "wagon", "price": 105000},
    
    {"make": "Audi", "model": "R8", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 602, "mpg": 17, "body_type": "coupe", "price": 180000},
    {"make": "Audi", "model": "R8", "year": 2021, "mileage": 12000, "condition": "excellent", "engine_hp": 602, "mpg": 17, "body_type": "coupe", "price": 155000},
    
    # Economy cars
    {"make": "Hyundai", "model": "Elantra", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 147, "mpg": 37, "body_type": "sedan", "price": 22000},
    {"make": "Hyundai", "model": "Elantra", "year": 2022, "mileage": 25000, "condition": "excellent", "engine_hp": 147, "mpg": 37, "body_type": "sedan", "price": 18500},
    
    {"make": "Kia", "model": "K5", "year": 2024, "mileage": 0, "condition": "new", "engine_hp": 180, "mpg": 32, "body_type": "sedan", "price": 27000},
    {"make": "Kia", "model": "Stinger GT", "year": 2023, "mileage": 10000, "condition": "excellent", "engine_hp": 368, "mpg": 25, "body_type": "sedan", "price": 48000},
]


class CarPricePredictor:
    """
    ML model to predict car prices based on various features.
    
    Uses Random Forest with feature engineering for:
    - Categorical encoding (make, model, condition)
    - Numerical scaling (year, mileage, engine_hp)
    - Derived features (age, price per hp)
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.make_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        self.body_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        self.load_or_train()
    
    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Prepare features for model training/prediction.
        
        Applies:
        - Label encoding for categorical features
        - Standard scaling for numerical features
        - Derived feature creation
        """
        df = df.copy()
        current_year = 2024
        
        # Derived features
        df['age'] = current_year - df['year']
        df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        
        # Encode categorical features
        if fit:
            df['make_encoded'] = self.make_encoder.fit_transform(df['make'])
            df['model_encoded'] = self.model_encoder.fit_transform(df['model'])
            df['condition_encoded'] = self.condition_encoder.fit_transform(df['condition'])
            df['body_encoded'] = self.body_encoder.fit_transform(df['body_type'])
        else:
            # Handle unseen categories
            df['make_encoded'] = df['make'].apply(
                lambda x: self.make_encoder.transform([x])[0] if x in self.make_encoder.classes_ else -1
            )
            df['model_encoded'] = df['model'].apply(
                lambda x: self.model_encoder.transform([x])[0] if x in self.model_encoder.classes_ else -1
            )
            df['condition_encoded'] = df['condition'].apply(
                lambda x: self.condition_encoder.transform([x])[0] if x in self.condition_encoder.classes_ else -1
            )
            df['body_encoded'] = df['body_type'].apply(
                lambda x: self.body_encoder.transform([x])[0] if x in self.body_encoder.classes_ else -1
            )
        
        # Feature matrix
        feature_cols = [
            'make_encoded', 'model_encoded', 'age', 'mileage', 
            'condition_encoded', 'engine_hp', 'mpg', 'body_encoded',
            'mileage_per_year'
        ]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        
        # Scale features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    def train(self, data: list = None):
        """
        Train the price prediction model.
        """
        if data is None:
            data = TRAINING_DATA
        
        df = pd.DataFrame(data)
        
        print(f"[PricePredictor] Training on {len(df)} samples...")
        
        # Prepare features
        X = self._prepare_features(df, fit=True)
        y = df['price'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model (Random Forest works great for tabular data)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"[PricePredictor] Model trained successfully!")
        print(f"   RMSE: ${rmse:,.0f}")
        print(f"   MAE:  ${mae:,.0f}")
        print(f"   R²:   {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"   CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.is_trained = True
        self._save()
        
        return {"rmse": rmse, "mae": mae, "r2": r2}
    
    def predict(self, make: str, model: str, year: int, mileage: int,
                condition: str = "good", engine_hp: int = 200,
                mpg: int = 28, body_type: str = "sedan") -> dict:
        """
        Predict the price of a car.
        
        Returns:
            dict with predicted price and confidence info
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            "make": make,
            "model": model,
            "year": year,
            "mileage": mileage,
            "condition": condition,
            "engine_hp": engine_hp,
            "mpg": mpg,
            "body_type": body_type
        }])
        
        try:
            X = self._prepare_features(input_data, fit=False)
            
            # Get prediction and confidence interval
            predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            predicted_price = np.mean(predictions)
            confidence_low = np.percentile(predictions, 10)
            confidence_high = np.percentile(predictions, 90)
            
            return {
                "predicted_price": round(predicted_price, -2),  # Round to nearest 100
                "confidence_range": {
                    "low": round(confidence_low, -2),
                    "high": round(confidence_high, -2)
                },
                "input": {
                    "make": make,
                    "model": model,
                    "year": year,
                    "mileage": mileage,
                    "condition": condition
                }
            }
        except Exception as e:
            return {"error": str(e), "predicted_price": None}
    
    def get_feature_importance(self) -> dict:
        """Get feature importance rankings."""
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))
    
    def _save(self):
        """Save model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "make_encoder": self.make_encoder,
            "model_encoder": self.model_encoder,
            "condition_encoder": self.condition_encoder,
            "body_encoder": self.body_encoder,
            "scaler": self.scaler,
            "feature_names": self.feature_names
        }
        
        joblib.dump(model_data, MODEL_PATH)
        print(f"[PricePredictor] Model saved to {MODEL_PATH}")
    
    def load_or_train(self):
        """Load existing model or train a new one."""
        if os.path.exists(MODEL_PATH):
            try:
                model_data = joblib.load(MODEL_PATH)
                self.model = model_data["model"]
                self.make_encoder = model_data["make_encoder"]
                self.model_encoder = model_data["model_encoder"]
                self.condition_encoder = model_data["condition_encoder"]
                self.body_encoder = model_data["body_encoder"]
                self.scaler = model_data["scaler"]
                self.feature_names = model_data["feature_names"]
                self.is_trained = True
                print(f"[PricePredictor] Loaded model from {MODEL_PATH}")
            except Exception as e:
                print(f"[PricePredictor] Error loading model: {e}")
                self.train()
        else:
            self.train()


# Singleton instance
_price_predictor = None

def get_price_predictor() -> CarPricePredictor:
    """Get or create the price predictor singleton."""
    global _price_predictor
    if _price_predictor is None:
        _price_predictor = CarPricePredictor()
    return _price_predictor


if __name__ == "__main__":
    # Test the price predictor
    predictor = CarPricePredictor()
    
    print("\n" + "="*60)
    print("Testing Car Price Predictor")
    print("="*60)
    
    test_cases = [
        {"make": "Toyota", "model": "Camry", "year": 2022, "mileage": 30000, "condition": "good", "engine_hp": 203, "mpg": 32, "body_type": "sedan"},
        {"make": "Ford", "model": "Mustang GT", "year": 2023, "mileage": 10000, "condition": "excellent", "engine_hp": 480, "mpg": 21, "body_type": "coupe"},
        {"make": "Tesla", "model": "Model 3", "year": 2024, "mileage": 5000, "condition": "excellent", "engine_hp": 283, "mpg": 132, "body_type": "sedan"},
        {"make": "BMW", "model": "M3", "year": 2021, "mileage": 25000, "condition": "good", "engine_hp": 473, "mpg": 23, "body_type": "sedan"},
    ]
    
    for car in test_cases:
        result = predictor.predict(**car)
        print(f"\n🚗 {car['year']} {car['make']} {car['model']}")
        print(f"   Mileage: {car['mileage']:,} | Condition: {car['condition']}")
        if "predicted_price" in result and result["predicted_price"]:
            print(f"   💰 Predicted: ${result['predicted_price']:,.0f}")
            print(f"   📊 Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")
    
    print("\n" + "-"*40)
    print("Feature Importance:")
    importance = predictor.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feat}: {imp:.3f}")
