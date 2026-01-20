"""
Utility Helper Functions
Reusable functions for the AutoMind application.
"""

import re
from typing import Dict, Any, Optional


def format_car_specs(specs: Dict[str, Any]) -> str:
    """
    Format car specifications into a readable string.
    
    Args:
        specs: Dictionary containing car specifications.
        
    Returns:
        str: Formatted string with specifications.
    """
    formatted_lines = []
    
    spec_labels = {
        "horsepower": "🔥 Horsepower",
        "torque": "💪 Torque",
        "zero_to_sixty": "⚡ 0-60 mph",
        "top_speed": "🏎️ Top Speed",
        "price": "💰 Price",
        "mpg": "⛽ Fuel Economy",
        "engine": "🔧 Engine",
        "transmission": "⚙️ Transmission",
        "drivetrain": "🛞 Drivetrain",
        "weight": "⚖️ Weight",
    }
    
    for key, value in specs.items():
        label = spec_labels.get(key, key.replace("_", " ").title())
        formatted_lines.append(f"• {label}: {value}")
    
    return "\n".join(formatted_lines)


def validate_question(question: str) -> tuple[bool, Optional[str]]:
    """
    Validate a user's question.
    
    Args:
        question: The user's input question.
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not question:
        return False, "Please enter a question."
    
    if len(question) < 3:
        return False, "Question is too short. Please provide more details."
    
    if len(question) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."
    
    # Check for potential injection attempts (basic sanitization)
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False, "Invalid characters detected in question."
    
    return True, None


def extract_car_names(text: str) -> list[str]:
    """
    Extract potential car names from text.
    
    Args:
        text: Input text to search for car names.
        
    Returns:
        list: List of potential car names found.
    """
    # Common car brand patterns
    brands = [
        "BMW", "Mercedes", "Audi", "Porsche", "Ferrari", "Lamborghini",
        "Ford", "Chevrolet", "Dodge", "Tesla", "Toyota", "Honda",
        "Nissan", "Mazda", "Subaru", "Lexus", "Acura", "Infiniti",
        "Volkswagen", "VW", "Volvo", "Jaguar", "Land Rover", "Range Rover",
        "Bentley", "Rolls Royce", "Aston Martin", "McLaren", "Bugatti",
        "Maserati", "Alfa Romeo", "Fiat", "Jeep", "RAM", "GMC",
        "Cadillac", "Lincoln", "Buick", "Chrysler", "Hyundai", "Kia",
        "Genesis", "Rivian", "Lucid", "Polestar"
    ]
    
    found_cars = []
    text_upper = text.upper()
    
    for brand in brands:
        if brand.upper() in text_upper:
            # Find the full model name if possible
            pattern = rf'{brand}\s+[\w\d\-]+(?:\s+[\w\d\-]+)?'
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_cars.extend(matches if matches else [brand])
    
    return list(set(found_cars))


def sanitize_response(response: str) -> str:
    """
    Sanitize agent response for safe display.
    
    Args:
        response: Raw agent response.
        
    Returns:
        str: Sanitized response.
    """
    # Remove any potential HTML tags
    clean = re.sub(r'<[^>]+>', '', response)
    
    # Remove excessive whitespace
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    clean = re.sub(r' {2,}', ' ', clean)
    
    return clean.strip()
