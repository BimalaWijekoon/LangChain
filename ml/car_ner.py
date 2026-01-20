"""
Custom Named Entity Recognition (NER) for AutoMind
===================================================

A spaCy-based NER model trained to extract car-related entities:
- CAR_MAKE: Car manufacturer (Toyota, BMW, Ford, etc.)
- CAR_MODEL: Model name (Mustang, Camry, M3, etc.)
- YEAR: Model year (2024, 2023, etc.)
- SPEC_VALUE: Specification values (500hp, 3.5 seconds, 25 mpg, etc.)
- PRICE: Price values ($50,000, 30k, etc.)

NLP/ML Concepts:
- Named Entity Recognition
- Custom Entity Training
- spaCy NLP Pipeline
- Pattern Matching + ML Hybrid
"""

import os
import json
import spacy
from spacy.tokens import Span
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
import re

# Path for model persistence
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
NER_MODEL_PATH = os.path.join(MODEL_DIR, "car_ner")


# Car makes and models database
CAR_MAKES = [
    "toyota", "honda", "ford", "chevrolet", "chevy", "bmw", "mercedes", "mercedes-benz",
    "audi", "porsche", "ferrari", "lamborghini", "nissan", "mazda", "subaru",
    "volkswagen", "vw", "volvo", "jaguar", "land rover", "tesla", "lexus", "acura",
    "hyundai", "kia", "genesis", "dodge", "jeep", "ram", "chrysler", "gmc",
    "cadillac", "buick", "lincoln", "infiniti", "mitsubishi", "maserati",
    "alfa romeo", "fiat", "mini", "aston martin", "bentley", "rolls royce",
    "mclaren", "bugatti", "koenigsegg", "pagani", "lotus", "rivian", "lucid",
    "polestar", "rimac", "suzuki", "daihatsu", "isuzu", "hummer"
]

CAR_MODELS = [
    # Toyota
    "camry", "corolla", "rav4", "highlander", "tacoma", "tundra", "supra", "gr86", "4runner",
    "prius", "sienna", "avalon", "yaris", "c-hr", "venza", "sequoia", "land cruiser",
    # Honda
    "civic", "accord", "cr-v", "pilot", "odyssey", "hr-v", "passport", "ridgeline",
    "fit", "insight", "civic type r", "nsx",
    # Ford
    "mustang", "f-150", "f150", "f-250", "explorer", "escape", "bronco", "ranger",
    "expedition", "edge", "mach-e", "maverick", "gt", "focus", "fusion",
    # Chevrolet
    "corvette", "camaro", "silverado", "tahoe", "suburban", "equinox", "blazer",
    "traverse", "colorado", "malibu", "impala", "bolt",
    # BMW
    "m3", "m4", "m5", "m8", "x3", "x5", "x7", "3 series", "5 series", "7 series",
    "z4", "i4", "ix", "m2", "x1", "x6",
    # Mercedes
    "c-class", "e-class", "s-class", "a-class", "gle", "glc", "gls", "amg gt",
    "c63", "e63", "s63", "g-wagon", "g-class", "sl", "cls", "eqs", "eqe",
    # Porsche
    "911", "cayenne", "macan", "panamera", "taycan", "718", "boxster", "cayman",
    "carrera", "turbo s", "gt3", "gt4",
    # Tesla
    "model s", "model 3", "model x", "model y", "cybertruck", "roadster", "semi",
    "plaid",
    # Others
    "mustang gt", "camaro ss", "hellcat", "demon", "trackhawk", "gtr", "gt-r",
    "supra", "miata", "mx-5", "wrx", "sti", "370z", "400z", "z", "r8", "rs6",
    "rs7", "aventador", "huracan", "urus", "488", "f8", "sf90", "roma",
    "wrangler", "grand cherokee", "durango", "challenger", "charger", "giulia",
]


@Language.component("car_entity_recognizer")
def car_entity_recognizer(doc):
    """
    Custom spaCy component for recognizing car entities.
    Uses pattern matching + rules for entity extraction.
    """
    new_ents = []
    original_ents = list(doc.ents)
    
    # Process text for pattern matching
    text_lower = doc.text.lower()
    
    # --- CAR_MAKE Detection ---
    for make in CAR_MAKES:
        for match in re.finditer(r'\b' + re.escape(make) + r'\b', text_lower):
            start_char = match.start()
            end_char = match.end()
            
            # Convert char positions to token positions
            span = doc.char_span(start_char, end_char, label="CAR_MAKE")
            if span is not None:
                new_ents.append(span)
    
    # --- CAR_MODEL Detection ---
    for model in CAR_MODELS:
        for match in re.finditer(r'\b' + re.escape(model) + r'\b', text_lower):
            start_char = match.start()
            end_char = match.end()
            
            span = doc.char_span(start_char, end_char, label="CAR_MODEL")
            if span is not None:
                new_ents.append(span)
    
    # --- YEAR Detection (4-digit years 1950-2030) ---
    year_pattern = r'\b(19[5-9]\d|20[0-3]\d)\b'
    for match in re.finditer(year_pattern, doc.text):
        start_char = match.start()
        end_char = match.end()
        
        span = doc.char_span(start_char, end_char, label="YEAR")
        if span is not None:
            new_ents.append(span)
    
    # --- SPEC_VALUE Detection ---
    spec_patterns = [
        r'\b\d+\s*(hp|horsepower|bhp|whp)\b',  # Horsepower
        r'\b\d+\.?\d*\s*(lb-ft|lb ft|nm|n-m)\b',  # Torque
        r'\b\d+\.?\d*\s*(sec|seconds|s)\s*(0-60|0 to 60)?\b',  # Time
        r'\b(0-60|0 to 60|zero to sixty)\s*:?\s*\d+\.?\d*\s*(s|sec|seconds)?\b',
        r'\b\d+\.?\d*\s*(mpg|miles per gallon|l/100km)\b',  # Fuel economy
        r'\b\d+\.?\d*\s*(mph|km/h|kmh|kph)\b',  # Speed
        r'\b\d+\.?\d*\s*(liter|litre|l)\b',  # Engine size
        r'\b(v6|v8|v10|v12|i4|i6|flat-4|flat-6|inline-4|inline-6)\b',  # Engine type
        r'\b\d+\s*(mile|miles|km|kilometer)\s*range\b',  # Range
        r'\b\d+\s*(kw|kwh|kilowatt)\b',  # Electric specs
    ]
    
    for pattern in spec_patterns:
        for match in re.finditer(pattern, text_lower):
            start_char = match.start()
            end_char = match.end()
            
            span = doc.char_span(start_char, end_char, label="SPEC_VALUE")
            if span is not None:
                new_ents.append(span)
    
    # --- PRICE Detection ---
    price_patterns = [
        r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',  # $50,000.00
        r'\$\s*\d+k\b',  # $50k
        r'\b\d{1,3}(?:,\d{3})*\s*(?:dollars|usd)\b',  # 50,000 dollars
        r'\bmsrp\s*:?\s*\$?\s*\d+',  # MSRP: $50,000
    ]
    
    for pattern in price_patterns:
        for match in re.finditer(pattern, text_lower):
            start_char = match.start()
            end_char = match.end()
            
            span = doc.char_span(start_char, end_char, label="PRICE")
            if span is not None:
                new_ents.append(span)
    
    # Filter overlapping entities (keep longest)
    filtered_ents = filter_overlapping_entities(new_ents + original_ents)
    doc.ents = filtered_ents
    
    return doc


def filter_overlapping_entities(entities):
    """Remove overlapping entities, keeping the longest one."""
    if not entities:
        return []
    
    # Sort by start position, then by length (descending)
    sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    
    filtered = []
    last_end = -1
    
    for ent in sorted_ents:
        if ent.start >= last_end:
            filtered.append(ent)
            last_end = ent.end
    
    return filtered


class CarEntityRecognizer:
    """
    Custom NER for car-related entities.
    
    Extracts: CAR_MAKE, CAR_MODEL, YEAR, SPEC_VALUE, PRICE
    """
    
    ENTITY_TYPES = ["CAR_MAKE", "CAR_MODEL", "YEAR", "SPEC_VALUE", "PRICE"]
    
    def __init__(self):
        """Initialize the NER model."""
        self.nlp = None
        self.load_or_create()
    
    def load_or_create(self):
        """Load existing model or create a new one."""
        # Load base spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom component
        if "car_entity_recognizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("car_entity_recognizer", after="ner")
        
        print("[CarNER] Entity recognizer initialized")
        print(f"[CarNER] Pipeline: {self.nlp.pipe_names}")
    
    def extract_entities(self, text: str) -> dict:
        """
        Extract car-related entities from text.
        
        Returns:
            dict with entity types and their values
        """
        doc = self.nlp(text)
        
        entities = {
            "CAR_MAKE": [],
            "CAR_MODEL": [],
            "YEAR": [],
            "SPEC_VALUE": [],
            "PRICE": [],
            "OTHER": []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            else:
                entities["OTHER"].append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities
    
    def extract_car_info(self, text: str) -> dict:
        """
        Extract structured car information from text.
        
        Returns:
            Consolidated car info dict
        """
        entities = self.extract_entities(text)
        
        # Consolidate into structured format
        car_info = {
            "make": entities["CAR_MAKE"][0]["text"] if entities["CAR_MAKE"] else None,
            "model": entities["CAR_MODEL"][0]["text"] if entities["CAR_MODEL"] else None,
            "year": entities["YEAR"][0]["text"] if entities["YEAR"] else None,
            "specs": [e["text"] for e in entities["SPEC_VALUE"]],
            "price": entities["PRICE"][0]["text"] if entities["PRICE"] else None,
            "all_entities": entities
        }
        
        return car_info
    
    def get_entity_annotations(self, text: str) -> list:
        """
        Get entity annotations for visualization.
        
        Returns:
            List of (text, start, end, label) tuples
        """
        doc = self.nlp(text)
        return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


# Singleton instance
_car_ner = None

def get_car_ner() -> CarEntityRecognizer:
    """Get or create the car NER singleton."""
    global _car_ner
    if _car_ner is None:
        _car_ner = CarEntityRecognizer()
    return _car_ner


if __name__ == "__main__":
    # Test the NER
    ner = CarEntityRecognizer()
    
    test_texts = [
        "I want to buy a 2024 Toyota Supra with 382 horsepower",
        "The BMW M3 does 0-60 in 3.8 seconds and costs $70,000",
        "Compare the Ford Mustang GT vs Chevrolet Camaro SS",
        "The Tesla Model S Plaid has 1020 hp and 390 miles range",
        "Looking for a reliable Honda Civic from 2022 under $25k",
    ]
    
    print("\n" + "="*60)
    print("Testing Car Entity Recognizer")
    print("="*60)
    
    for text in test_texts:
        print(f"\n📝 '{text}'")
        print("-" * 40)
        
        entities = ner.extract_entities(text)
        for ent_type, ents in entities.items():
            if ents and ent_type != "OTHER":
                print(f"  {ent_type}: {[e['text'] for e in ents]}")
