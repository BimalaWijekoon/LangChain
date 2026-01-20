"""
ML-Powered Tools for AutoMind
=============================

LangChain tools that use our custom-trained ML models:
- Sentiment Analysis Tool
- Entity Extraction Tool  
- Price Prediction Tool
- Intent Analysis Tool
"""

from langchain_core.tools import tool
from typing import Optional


@tool
def analyze_car_review(review_text: str) -> str:
    """
    Analyze the sentiment of a car review or user opinion.
    
    Use this tool when users share their thoughts/experiences with a car,
    or when you need to analyze sentiment in car-related text.
    
    Args:
        review_text: The review or opinion text to analyze
        
    Returns:
        Sentiment analysis with score and explanation
    """
    from ml.sentiment_analyzer import get_sentiment_analyzer
    
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze(review_text)
    
    # Format response
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    emoji = {"positive": "👍", "negative": "👎", "neutral": "😐"}.get(sentiment, "")
    
    response = f"""
**Sentiment Analysis** {emoji}

📊 **Sentiment**: {sentiment.upper()}
📈 **Confidence**: {confidence:.0%}
💬 **Text**: "{review_text[:100]}{'...' if len(review_text) > 100 else ''}"

**Score Breakdown**:
- Positive: {result['scores']['positive']:.1%}
- Neutral: {result['scores']['neutral']:.1%}  
- Negative: {result['scores']['negative']:.1%}
"""
    return response


@tool
def extract_car_entities(text: str) -> str:
    """
    Extract car-related entities from text using NER (Named Entity Recognition).
    
    Extracts: Car makes, models, years, specs (horsepower, mpg), and prices.
    
    Use this when you need to understand what specific cars or specs 
    the user is talking about.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        Extracted car entities in structured format
    """
    from ml.car_ner import get_car_ner
    
    ner = get_car_ner()
    car_info = ner.extract_car_info(text)
    
    response = "**🔍 Extracted Car Entities**\n\n"
    
    if car_info["make"]:
        response += f"🏭 **Make**: {car_info['make'].title()}\n"
    if car_info["model"]:
        response += f"🚗 **Model**: {car_info['model'].title()}\n"
    if car_info["year"]:
        response += f"📅 **Year**: {car_info['year']}\n"
    if car_info["specs"]:
        response += f"⚡ **Specs**: {', '.join(car_info['specs'])}\n"
    if car_info["price"]:
        response += f"💰 **Price**: {car_info['price']}\n"
    
    # Show all entities
    all_ents = car_info["all_entities"]
    if all_ents.get("OTHER"):
        other_labels = set(e["label"] for e in all_ents["OTHER"])
        response += f"\n📌 **Other entities detected**: {', '.join(other_labels)}"
    
    if not any([car_info["make"], car_info["model"], car_info["year"]]):
        response = "No specific car entities found in the text."
    
    return response


@tool
def predict_car_price(make: str, model: str, year: int, mileage: int, 
                      condition: str = "good") -> str:
    """
    Predict the market price of a used car using ML.
    
    Use this tool when users ask about car values, pricing, or want to 
    know how much a specific car is worth.
    
    Args:
        make: Car manufacturer (Toyota, Honda, BMW, etc.)
        model: Car model name (Camry, Civic, M3, etc.)
        year: Model year (e.g., 2022)
        mileage: Current mileage in miles
        condition: Car condition - "new", "excellent", "good", "fair"
        
    Returns:
        Predicted price with confidence range
    """
    from ml.price_predictor import get_price_predictor
    
    predictor = get_price_predictor()
    
    # Estimate engine specs based on car type (simplified)
    engine_hp = 200  # Default
    mpg = 28  # Default
    body_type = "sedan"  # Default
    
    # Common high-performance models
    performance_models = ["mustang gt", "camaro ss", "corvette", "m3", "m4", 
                         "c63", "gt-r", "supra", "type r", "hellcat", "911"]
    if any(pm in model.lower() for pm in performance_models):
        engine_hp = 450
        mpg = 22
        body_type = "coupe"
    
    # SUVs
    suv_models = ["rav4", "cr-v", "explorer", "cayenne", "model x", "model y"]
    if any(s in model.lower() for s in suv_models):
        body_type = "suv"
        mpg = 26
    
    # Electric vehicles
    if make.lower() == "tesla":
        mpg = 120  # MPGe
    
    result = predictor.predict(
        make=make,
        model=model,
        year=year,
        mileage=mileage,
        condition=condition,
        engine_hp=engine_hp,
        mpg=mpg,
        body_type=body_type
    )
    
    if "error" in result and result.get("predicted_price") is None:
        return f"Unable to predict price: {result['error']}"
    
    price = result["predicted_price"]
    low = result["confidence_range"]["low"]
    high = result["confidence_range"]["high"]
    
    response = f"""
**💰 Price Prediction for {year} {make} {model}**

📊 **Estimated Value**: **${price:,.0f}**
📈 **Price Range**: ${low:,.0f} - ${high:,.0f}

**Vehicle Details**:
- 📅 Year: {year}
- 🛣️ Mileage: {mileage:,} miles
- ✨ Condition: {condition.title()}

*Note: This is an ML-based estimate. Actual prices may vary based on 
location, specific options, and market conditions.*
"""
    return response


@tool
def classify_user_intent(query: str) -> str:
    """
    Classify the intent of a user's car-related query.
    
    This is an internal tool to understand what the user is looking for.
    
    Categories: specs, comparison, pricing, maintenance, images, videos, 
    history, recommendation, general
    
    Args:
        query: The user's question or request
        
    Returns:
        Intent classification with confidence
    """
    from ml.intent_classifier import get_intent_classifier
    
    classifier = get_intent_classifier()
    result = classifier.predict(query)
    
    intent_emojis = {
        "specs": "📋",
        "comparison": "⚖️",
        "pricing": "💰",
        "maintenance": "🔧",
        "images": "🖼️",
        "videos": "🎬",
        "history": "📚",
        "recommendation": "💡",
        "general": "💬"
    }
    
    intent = result["intent"]
    emoji = intent_emojis.get(intent, "❓")
    
    response = f"""
**Intent Analysis** {emoji}

🎯 **Detected Intent**: {intent.upper()}
📊 **Confidence**: {result['confidence']:.0%}

**Top Predictions**:
"""
    for pred in result["top_3"]:
        response += f"- {pred['intent']}: {pred['confidence']:.0%}\n"
    
    return response


# Export all tools
ML_TOOLS = [
    analyze_car_review,
    extract_car_entities,
    predict_car_price,
    classify_user_intent,
]
