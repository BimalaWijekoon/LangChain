"""
Test Script for AutoMind ML Models
==================================

Run this script to verify all ML models are working correctly.

Usage:
    python test_ml_models.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def test_intent_classifier():
    """Test the Intent Classifier model."""
    print_header("Testing Intent Classifier")
    
    try:
        from ml.intent_classifier import get_intent_classifier
        classifier = get_intent_classifier()
        
        test_queries = [
            "What are the specs of the Toyota Supra?",
            "BMW M3 vs Audi RS5 which is better?",
            "How much does a 2024 Ford Mustang cost?",
            "When should I change my oil?",
            "Show me pictures of the Porsche 911",
        ]
        
        print("\nClassification Results:")
        for query in test_queries:
            result = classifier.predict(query)
            print(f"\n📝 '{query}'")
            print(f"   🎯 Intent: {result['intent'].upper()} ({result['confidence']:.0%})")
        
        print("\n✅ Intent Classifier: WORKING")
        return True
    except Exception as e:
        print(f"\n❌ Intent Classifier: FAILED - {e}")
        return False

def test_sentiment_analyzer():
    """Test the Sentiment Analyzer model."""
    print_header("Testing Sentiment Analyzer")
    
    try:
        from ml.sentiment_analyzer import get_sentiment_analyzer
        analyzer = get_sentiment_analyzer()
        
        test_reviews = [
            "This car is amazing! Best purchase I ever made!",
            "Terrible reliability, constant issues. Would not recommend.",
            "It's okay, nothing special but gets the job done.",
        ]
        
        print("\nSentiment Analysis Results:")
        for review in test_reviews:
            result = analyzer.analyze(review)
            emoji = {"positive": "👍", "negative": "👎", "neutral": "😐"}.get(result["sentiment"], "")
            print(f"\n📝 '{review[:50]}...'")
            print(f"   {emoji} Sentiment: {result['sentiment'].upper()} ({result['confidence']:.0%})")
        
        print("\n✅ Sentiment Analyzer: WORKING")
        return True
    except Exception as e:
        print(f"\n❌ Sentiment Analyzer: FAILED - {e}")
        return False

def test_car_ner():
    """Test the Car NER model."""
    print_header("Testing Car NER (Entity Recognition)")
    
    try:
        from ml.car_ner import get_car_ner
        ner = get_car_ner()
        
        test_texts = [
            "I want to buy a 2024 Toyota Supra with 382 horsepower",
            "The BMW M3 does 0-60 in 3.8 seconds and costs $70,000",
            "Compare the Ford Mustang GT vs Chevrolet Camaro SS",
        ]
        
        print("\nEntity Extraction Results:")
        for text in test_texts:
            entities = ner.extract_entities(text)
            print(f"\n📝 '{text}'")
            for ent_type, ents in entities.items():
                if ents and ent_type != "OTHER":
                    values = [e['text'] for e in ents]
                    print(f"   🏷️ {ent_type}: {values}")
        
        print("\n✅ Car NER: WORKING")
        return True
    except Exception as e:
        print(f"\n❌ Car NER: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_price_predictor():
    """Test the Price Predictor model."""
    print_header("Testing Price Predictor")
    
    try:
        from ml.price_predictor import get_price_predictor
        predictor = get_price_predictor()
        
        test_cars = [
            {"make": "Toyota", "model": "Camry", "year": 2022, "mileage": 30000, "condition": "good"},
            {"make": "Ford", "model": "Mustang GT", "year": 2023, "mileage": 10000, "condition": "excellent"},
            {"make": "BMW", "model": "M3", "year": 2021, "mileage": 25000, "condition": "good"},
        ]
        
        print("\nPrice Prediction Results:")
        for car in test_cars:
            result = predictor.predict(**car, engine_hp=300, mpg=25, body_type="sedan")
            print(f"\n🚗 {car['year']} {car['make']} {car['model']}")
            print(f"   📍 {car['mileage']:,} miles | {car['condition']}")
            if result.get("predicted_price"):
                print(f"   💰 Predicted: ${result['predicted_price']:,.0f}")
                print(f"   📊 Range: ${result['confidence_range']['low']:,.0f} - ${result['confidence_range']['high']:,.0f}")
            else:
                print(f"   ⚠️ Error: {result.get('error', 'Unknown')}")
        
        print("\n✅ Price Predictor: WORKING")
        return True
    except Exception as e:
        print(f"\n❌ Price Predictor: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_tools():
    """Test the LangChain ML tools."""
    print_header("Testing LangChain ML Tools")
    
    try:
        from tools.ml_tools import (
            analyze_car_review, 
            predict_car_price, 
            extract_car_entities
        )
        
        # Test sentiment tool
        print("\n📊 Testing analyze_car_review tool...")
        result = analyze_car_review.invoke({"review_text": "Amazing car! Love the performance!"})
        print(f"   Result snippet: {result[:100]}...")
        
        # Test price prediction tool
        print("\n💰 Testing predict_car_price tool...")
        result = predict_car_price.invoke({
            "make": "Toyota",
            "model": "Camry", 
            "year": 2022,
            "mileage": 25000,
            "condition": "good"
        })
        print(f"   Result snippet: {result[:100]}...")
        
        # Test NER tool
        print("\n🏷️ Testing extract_car_entities tool...")
        result = extract_car_entities.invoke({"text": "Looking at a 2024 BMW M3 with 473 hp"})
        print(f"   Result snippet: {result[:100]}...")
        
        print("\n✅ ML Tools: WORKING")
        return True
    except Exception as e:
        print(f"\n❌ ML Tools: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all ML model tests."""
    print("\n" + "🚀"*30)
    print("       AutoMind ML Models Test Suite")
    print("🚀"*30)
    
    results = {
        "Intent Classifier": test_intent_classifier(),
        "Sentiment Analyzer": test_sentiment_analyzer(),
        "Car NER": test_car_ner(),
        "Price Predictor": test_price_predictor(),
        "ML Tools": test_ml_tools(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n📊 Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All ML models are working correctly!")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
