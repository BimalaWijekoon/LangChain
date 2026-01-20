"""
RAG (Retrieval Augmented Generation) Tools for AutoMind
========================================================

This module implements a vector database with car specifications for fast,
offline lookup of common car data.

NLP/LangChain Concepts:
- Vector Embeddings: Converting text to numerical vectors
- Semantic Search: Finding similar content based on meaning
- FAISS: Facebook AI Similarity Search for fast vector lookup
- RAG Pattern: Retrieve relevant docs before generating response
"""

import os
import json
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path for vector store persistence
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "car_vectors")


# Sample car knowledge base - This would typically come from a larger database
CAR_KNOWLEDGE_BASE = [
    # Sports Cars
    {
        "category": "sports_car",
        "name": "Toyota GR Supra",
        "year": "2024",
        "specs": {
            "engine": "3.0L Twin-Scroll Turbo I6",
            "horsepower": 382,
            "torque": "368 lb-ft",
            "0_60": "3.9 seconds",
            "top_speed": "155 mph (limited)",
            "transmission": "8-speed automatic or 6-speed manual",
            "drivetrain": "RWD",
            "weight": "3,400 lbs",
            "mpg": "22 city / 30 highway"
        },
        "description": "The Toyota GR Supra is a legendary sports car reborn, co-developed with BMW. It features a BMW-sourced inline-6 engine with exceptional power delivery and handling."
    },
    {
        "category": "sports_car",
        "name": "Chevrolet Corvette C8",
        "year": "2024",
        "specs": {
            "engine": "6.2L V8 LT2",
            "horsepower": 495,
            "torque": "470 lb-ft",
            "0_60": "2.9 seconds",
            "top_speed": "194 mph",
            "transmission": "8-speed dual-clutch automatic",
            "drivetrain": "RWD (mid-engine)",
            "weight": "3,647 lbs",
            "mpg": "16 city / 24 highway"
        },
        "description": "The C8 Corvette represents a revolutionary mid-engine design for America's sports car. It offers supercar performance at a fraction of the price."
    },
    {
        "category": "sports_car",
        "name": "Ford Mustang GT",
        "year": "2024",
        "specs": {
            "engine": "5.0L Ti-VCT V8",
            "horsepower": 486,
            "torque": "418 lb-ft",
            "0_60": "4.1 seconds",
            "top_speed": "155 mph (limited)",
            "transmission": "6-speed manual or 10-speed automatic",
            "drivetrain": "RWD",
            "weight": "3,832 lbs",
            "mpg": "16 city / 24 highway"
        },
        "description": "The 7th generation Mustang continues the legacy with a powerful Coyote V8 engine and available manual transmission for purists."
    },
    {
        "category": "sports_car",
        "name": "Nissan Z",
        "year": "2024",
        "specs": {
            "engine": "3.0L Twin-Turbo V6",
            "horsepower": 400,
            "torque": "350 lb-ft",
            "0_60": "4.5 seconds",
            "top_speed": "155 mph (limited)",
            "transmission": "6-speed manual or 9-speed automatic",
            "drivetrain": "RWD",
            "weight": "3,540 lbs",
            "mpg": "18 city / 26 highway"
        },
        "description": "The Nissan Z pays homage to the legendary Z-car lineage with retro-inspired design and modern twin-turbo performance."
    },
    {
        "category": "sports_car", 
        "name": "Porsche 911 Carrera",
        "year": "2024",
        "specs": {
            "engine": "3.0L Twin-Turbo Flat-6",
            "horsepower": 394,
            "torque": "331 lb-ft",
            "0_60": "4.0 seconds",
            "top_speed": "182 mph",
            "transmission": "8-speed PDK or 7-speed manual",
            "drivetrain": "RWD",
            "weight": "3,354 lbs",
            "mpg": "18 city / 24 highway"
        },
        "description": "The iconic Porsche 911 continues its legacy as the benchmark sports car with its rear-engine layout and precision handling."
    },
    # Electric Vehicles
    {
        "category": "electric",
        "name": "Tesla Model S Plaid",
        "year": "2024",
        "specs": {
            "motor": "Tri-Motor AWD",
            "horsepower": 1020,
            "torque": "Instant electric torque",
            "0_60": "1.99 seconds",
            "top_speed": "200 mph",
            "range": "396 miles",
            "drivetrain": "AWD",
            "weight": "4,766 lbs",
            "charging": "250kW Supercharger"
        },
        "description": "The Tesla Model S Plaid is the fastest production sedan ever made, featuring tri-motor AWD and a sub-2 second 0-60 time."
    },
    {
        "category": "electric",
        "name": "Tesla Model 3",
        "year": "2024",
        "specs": {
            "motor": "Dual Motor AWD",
            "horsepower": 366,
            "torque": "Instant electric torque",
            "0_60": "4.2 seconds",
            "top_speed": "145 mph",
            "range": "333 miles",
            "drivetrain": "AWD",
            "weight": "4,034 lbs",
            "charging": "250kW Supercharger"
        },
        "description": "The Tesla Model 3 is the best-selling electric car globally, offering excellent range, technology, and performance at an accessible price."
    },
    {
        "category": "electric",
        "name": "BMW i4 M50",
        "year": "2024",
        "specs": {
            "motor": "Dual Motor AWD",
            "horsepower": 536,
            "torque": "586 lb-ft",
            "0_60": "3.7 seconds",
            "top_speed": "130 mph",
            "range": "271 miles",
            "drivetrain": "AWD",
            "weight": "5,018 lbs",
            "charging": "200kW DC fast charging"
        },
        "description": "The BMW i4 M50 brings M performance to the electric era with dual motors and BMW's signature driving dynamics."
    },
    {
        "category": "electric",
        "name": "Porsche Taycan Turbo S",
        "year": "2024",
        "specs": {
            "motor": "Dual Motor AWD",
            "horsepower": 750,
            "torque": "774 lb-ft (overboost)",
            "0_60": "2.6 seconds",
            "top_speed": "162 mph",
            "range": "280 miles",
            "drivetrain": "AWD",
            "weight": "5,132 lbs",
            "charging": "270kW DC fast charging"
        },
        "description": "The Porsche Taycan Turbo S combines Porsche's legendary handling with electric performance and 800V architecture."
    },
    # SUVs
    {
        "category": "suv",
        "name": "Toyota RAV4",
        "year": "2024",
        "specs": {
            "engine": "2.5L 4-Cylinder",
            "horsepower": 203,
            "torque": "184 lb-ft",
            "0_60": "8.4 seconds",
            "towing": "3,500 lbs",
            "transmission": "8-speed automatic",
            "drivetrain": "FWD or AWD",
            "weight": "3,615 lbs",
            "mpg": "27 city / 35 highway"
        },
        "description": "The Toyota RAV4 is the best-selling SUV in America, known for reliability, practicality, and excellent resale value."
    },
    {
        "category": "suv",
        "name": "Jeep Wrangler",
        "year": "2024",
        "specs": {
            "engine": "3.6L V6 or 2.0L Turbo",
            "horsepower": "285-470 hp",
            "torque": "260-470 lb-ft",
            "0_60": "6.5-7.5 seconds",
            "towing": "3,500 lbs",
            "transmission": "8-speed automatic or 6-speed manual",
            "drivetrain": "4WD",
            "weight": "4,439 lbs",
            "mpg": "17 city / 25 highway"
        },
        "description": "The Jeep Wrangler is the ultimate off-road vehicle with removable doors, roof, and legendary 4x4 capability."
    },
    {
        "category": "suv",
        "name": "BMW X5",
        "year": "2024",
        "specs": {
            "engine": "3.0L Turbo I6",
            "horsepower": 375,
            "torque": "398 lb-ft",
            "0_60": "5.3 seconds",
            "towing": "7,200 lbs",
            "transmission": "8-speed automatic",
            "drivetrain": "AWD",
            "weight": "5,060 lbs",
            "mpg": "21 city / 26 highway"
        },
        "description": "The BMW X5 is a luxury SUV that combines performance, technology, and versatility with BMW's signature driving dynamics."
    },
    # Trucks
    {
        "category": "truck",
        "name": "Ford F-150",
        "year": "2024",
        "specs": {
            "engine": "3.5L EcoBoost V6 or 5.0L V8",
            "horsepower": "290-450 hp",
            "torque": "400-510 lb-ft",
            "0_60": "5.0-7.0 seconds",
            "towing": "13,500 lbs max",
            "payload": "3,310 lbs max",
            "transmission": "10-speed automatic",
            "drivetrain": "RWD or 4WD",
            "mpg": "20 city / 24 highway"
        },
        "description": "The Ford F-150 is America's best-selling vehicle with aluminum body, powerful engine options, and class-leading capability."
    },
    {
        "category": "truck",
        "name": "Toyota Tacoma",
        "year": "2024",
        "specs": {
            "engine": "2.4L Turbo I4",
            "horsepower": 278,
            "torque": "317 lb-ft",
            "0_60": "7.0 seconds",
            "towing": "6,500 lbs",
            "payload": "1,685 lbs",
            "transmission": "8-speed automatic or 6-speed manual",
            "drivetrain": "RWD or 4WD",
            "mpg": "21 city / 26 highway"
        },
        "description": "The Toyota Tacoma is the best-selling midsize truck, known for off-road capability and legendary Toyota reliability."
    },
    {
        "category": "truck",
        "name": "Ram 1500",
        "year": "2024",
        "specs": {
            "engine": "5.7L HEMI V8",
            "horsepower": 395,
            "torque": "410 lb-ft",
            "0_60": "6.3 seconds",
            "towing": "12,750 lbs max",
            "payload": "2,300 lbs",
            "transmission": "8-speed automatic",
            "drivetrain": "RWD or 4WD",
            "mpg": "17 city / 23 highway"
        },
        "description": "The Ram 1500 features the most luxurious interior in the truck segment with available air suspension and powerful HEMI V8."
    },
    # Sedans
    {
        "category": "sedan",
        "name": "Honda Accord",
        "year": "2024",
        "specs": {
            "engine": "1.5L Turbo I4",
            "horsepower": 192,
            "torque": "192 lb-ft",
            "0_60": "7.2 seconds",
            "top_speed": "130 mph",
            "transmission": "CVT",
            "drivetrain": "FWD",
            "weight": "3,239 lbs",
            "mpg": "30 city / 38 highway"
        },
        "description": "The Honda Accord is a legendary midsize sedan offering reliability, comfort, and excellent fuel economy."
    },
    {
        "category": "sedan",
        "name": "BMW M3",
        "year": "2024",
        "specs": {
            "engine": "3.0L Twin-Turbo I6",
            "horsepower": 473,
            "torque": "406 lb-ft",
            "0_60": "4.1 seconds",
            "top_speed": "180 mph (Competition)",
            "transmission": "8-speed automatic or 6-speed manual",
            "drivetrain": "RWD or xDrive AWD",
            "weight": "3,830 lbs",
            "mpg": "16 city / 23 highway"
        },
        "description": "The BMW M3 is the benchmark sport sedan with race-bred performance, available manual transmission, and iconic styling."
    },
    # Luxury
    {
        "category": "luxury",
        "name": "Mercedes-Benz S-Class",
        "year": "2024",
        "specs": {
            "engine": "3.0L Turbo I6 + EQ Boost",
            "horsepower": 429,
            "torque": "384 lb-ft",
            "0_60": "4.9 seconds",
            "top_speed": "130 mph (limited)",
            "transmission": "9-speed automatic",
            "drivetrain": "AWD (4MATIC)",
            "weight": "4,740 lbs",
            "mpg": "21 city / 30 highway"
        },
        "description": "The Mercedes-Benz S-Class is the flagship luxury sedan featuring cutting-edge technology, supreme comfort, and executive presence."
    },
    # Hypercars
    {
        "category": "hypercar",
        "name": "Bugatti Chiron",
        "year": "2024",
        "specs": {
            "engine": "8.0L Quad-Turbo W16",
            "horsepower": 1500,
            "torque": "1180 lb-ft",
            "0_60": "2.4 seconds",
            "top_speed": "261 mph (Sport)",
            "transmission": "7-speed dual-clutch",
            "drivetrain": "AWD",
            "weight": "4,400 lbs",
            "mpg": "8 city / 13 highway"
        },
        "description": "The Bugatti Chiron is one of the fastest and most exclusive hypercars ever made, featuring an 8-liter W16 engine with four turbochargers."
    }
]


def create_documents() -> List[Document]:
    """
    Convert car knowledge base to LangChain Documents.
    
    Each car entry becomes a Document with:
    - page_content: Human-readable description with specs
    - metadata: Structured data for filtering
    """
    documents = []
    
    for car in CAR_KNOWLEDGE_BASE:
        # Create rich content for semantic search
        specs = car["specs"]
        content = f"""
{car['name']} ({car['year']}) - {car['category'].replace('_', ' ').title()}

{car['description']}

Specifications:
"""
        for key, value in specs.items():
            formatted_key = key.replace('_', ' ').title()
            content += f"- {formatted_key}: {value}\n"
        
        # Create document with metadata
        doc = Document(
            page_content=content.strip(),
            metadata={
                "name": car["name"],
                "year": car["year"],
                "category": car["category"],
                **{k: str(v) for k, v in specs.items()}
            }
        )
        documents.append(doc)
    
    return documents


def get_or_create_vector_store() -> FAISS:
    """
    Load existing vector store or create a new one.
    
    Uses HuggingFace embeddings (runs locally, no API needed).
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Check if vector store exists
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception:
            pass  # Recreate if loading fails
    
    # Create new vector store
    documents = create_documents()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save for future use
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store


# Global vector store instance
_vector_store: Optional[FAISS] = None


def get_vector_store() -> FAISS:
    """Get or initialize the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = get_or_create_vector_store()
    return _vector_store


@tool
def car_knowledge_search(query: str) -> str:
    """
    Search the local car knowledge base for specifications and information.
    
    This tool uses RAG (Retrieval Augmented Generation) with a vector database
    for fast, offline car data lookup. It's faster than web search for common cars.
    
    Use this tool for:
    - Quick lookup of popular car specifications
    - Finding cars in specific categories (sports car, SUV, electric)
    - Comparing specs from the knowledge base
    - Getting baseline info before web search
    
    Args:
        query: Search query like 'Corvette specs' or 'best electric SUV'
    
    Returns:
        Relevant car specifications from the knowledge base
    """
    try:
        vector_store = get_vector_store()
        
        # Perform similarity search
        results = vector_store.similarity_search_with_score(query, k=3)
        
        if not results:
            return "No matching cars found in the knowledge base. Try web search for more results."
        
        output = "**📚 From Car Knowledge Base (RAG):**\n\n"
        
        for doc, score in results:
            # Lower score = more similar in FAISS
            relevance = "🎯 High Match" if score < 0.5 else "📌 Related"
            output += f"{relevance}\n"
            output += f"{doc.page_content}\n"
            output += "-" * 40 + "\n\n"
        
        output += "_Data from local vector database - for latest info, use web search_"
        
        return output
        
    except Exception as e:
        return f"Knowledge base search error: {str(e)}"
