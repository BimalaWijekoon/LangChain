# 🏎️ AutoMind - AI Car Expert Chat Application

<div align="center">

![AutoMind Banner](https://img.shields.io/badge/AutoMind-AI%20Car%20Expert-ff4d4d?style=for-the-badge&logo=car&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**An intelligent AI-powered chat application for car enthusiasts, built with LangChain, LangGraph, and Google Gemini.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [ML Models](#-custom-ml-models)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Architecture](#-architecture)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Configuration](#-configuration)
8. [Usage](#-usage)
9. [API Endpoints](#-api-endpoints)
10. [LangChain Agent System](#-langchain-agent-system)
11. [Multi-Agent Architecture](#-multi-agent-architecture)
12. [Custom ML Models](#-custom-ml-models)
13. [RAG System](#-rag-retrieval-augmented-generation)
14. [Tools & Capabilities](#-tools--capabilities)
15. [Frontend Interface](#-frontend-interface)
16. [Development](#-development)
17. [NLP Concepts Used](#-nlp-concepts-used)
18. [Future Improvements](#-future-improvements)
19. [License](#-license)

---

## 🎯 Overview

**AutoMind** is a full-stack AI chat application designed for car enthusiasts. It leverages modern NLP technologies including LangChain, LangGraph, and Google's Gemini AI to provide intelligent, context-aware responses about anything car-related.

### What Makes AutoMind Special?

- 🧠 **ReAct Agent Architecture** - Uses reasoning + acting pattern for intelligent tool selection
- 🤖 **Multi-Agent System** - Specialized agents for comparisons, pricing, and maintenance
- 📚 **RAG-Powered** - FAISS vector database for fast car spec lookups
- 🔧 **Custom ML Models** - Trained intent classifier, sentiment analyzer, NER, and price predictor
- 📸 **Vision Capabilities** - Analyze car images using Gemini Vision API
- 💬 **Chat History** - Persistent chat sessions like ChatGPT/Gemini

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| 🔍 **Real-Time Search** | DuckDuckGo integration for current car specs, prices, and news |
| 📊 **Car Comparisons** | Head-to-head comparisons using specialized comparison agent |
| 💰 **Price Predictions** | ML-based used car price estimation |
| 📸 **Image Analysis** | Upload car photos for AI identification |
| 🎬 **Video Search** | Find YouTube reviews and demonstrations |
| 📚 **Knowledge Base** | RAG system with pre-loaded car specifications |
| 🗣️ **Sentiment Analysis** | Analyze car reviews and opinions |
| 🏷️ **Entity Extraction** | NER for car makes, models, years, specs |

### UI Features

| Feature | Description |
|---------|-------------|
| 📝 **Chat History** | Sidebar with saved conversations |
| ✏️ **Rename Chats** | Custom names for your conversations |
| 🌙 **Dark Theme** | Modern, car-themed dark interface |
| 📱 **Responsive** | Mobile-friendly design |
| ⚡ **Quick Actions** | Pre-built query buttons |
| 🖼️ **Image Preview** | Preview before uploading |

---

## 🛠️ Tech Stack

### Backend

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **Flask 3.0+** | Web framework |
| **LangChain 0.3+** | NLP framework for AI agents |
| **LangGraph 0.2+** | Agent workflow orchestration |
| **Google Gemini** | Large Language Model (gemini-2.5-flash) |
| **FAISS** | Vector database for RAG |
| **scikit-learn** | ML models (classification, regression) |
| **spaCy** | NER (Named Entity Recognition) |
| **Pydantic** | Data validation and serialization |

### Frontend

| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Structure and styling |
| **Vanilla JavaScript** | Interactive functionality |
| **Marked.js** | Markdown rendering |
| **Inter + Orbitron** | Typography (Google Fonts) |
| **LocalStorage** | Persistent chat history |

### APIs & Tools

| Service | Purpose |
|---------|---------|
| **DuckDuckGo Search** | Real-time web search |
| **DuckDuckGo Images** | Car image search |
| **Wikipedia API** | Car history and brand info |
| **YouTube Search** | Video content discovery |
| **Gemini Vision API** | Image analysis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Sidebar    │  │  Chat UI    │  │  Image Upload           │  │
│  │  - History  │  │  - Messages │  │  - Preview              │  │
│  │  - New Chat │  │  - Markdown │  │  - Base64 Encoding      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ POST /ask
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK API LAYER                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  routes/chat.py                                              ││
│  │  - Request validation                                        ││
│  │  - Agent invocation                                          ││
│  │  - Markdown rendering                                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CAR EXPERT AGENT                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  agents/car_expert.py (LangGraph ReAct Agent)               ││
│  │                                                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ││
│  │  │ Intent      │  │ Specialist  │  │ Vision              │  ││
│  │  │ Detection   │→ │ Routing     │  │ Analysis            │  ││
│  │  │ (ML-based)  │  │             │  │ (Gemini)            │  ││
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  ││
│  │         │                │                                   ││
│  │         ▼                ▼                                   ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              TOOL SELECTION (10 Tools)                  │││
│  │  │  Search | Wikipedia | YouTube | RAG | Calculator | ML   │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  MULTI-AGENT    │  │  RAG SYSTEM     │  │  ML MODELS      │
│  SYSTEM         │  │                 │  │                 │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │Comparison │  │  │  │  FAISS    │  │  │  │ Intent    │  │
│  │ Specialist│  │  │  │ VectorDB  │  │  │  │ Classifier│  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ Pricing   │  │  │  │HuggingFace│  │  │  │ Sentiment │  │
│  │ Expert    │  │  │  │Embeddings │  │  │  │ Analyzer  │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │Maintenance│  │  │  │   Car     │  │  │  │  Car NER  │  │
│  │ Advisor   │  │  │  │ Knowledge │  │  │  │           │  │
│  └───────────┘  │  │  │   Base    │  │  │  └───────────┘  │
│  ┌───────────┐  │  │  └───────────┘  │  │  ┌───────────┐  │
│  │   Specs   │  │  │                 │  │  │  Price    │  │
│  │  Expert   │  │  │                 │  │  │ Predictor │  │
│  └───────────┘  │  │                 │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Flow

```
User Query → Intent Classification → Agent Routing → Tool Selection → Response Generation
     ↓              ↓                     ↓               ↓              ↓
  "Compare     ML Classifier        Multi-Agent      Web Search     Markdown
   BMW vs       predicts            routes to       + RAG lookup   formatted
   Mercedes"   "comparison"        Comparison        + ML tools     response
                intent              Specialist
```

---

## 📁 Project Structure

```
AutoMind/
│
├── 📄 app.py                    # Flask application entry point
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env                      # Environment variables (API keys)
├── 📄 .env.example              # Environment template
├── 📄 .gitignore                # Git ignore rules
│
├── 📁 agents/                   # AI Agent implementations
│   ├── 📄 __init__.py
│   ├── 📄 car_expert.py         # Main LangGraph ReAct agent
│   └── 📄 multi_agent.py        # Specialist agents (comparison, pricing, etc.)
│
├── 📁 config/                   # Configuration management
│   ├── 📄 __init__.py
│   └── 📄 settings.py           # App settings, environment config
│
├── 📁 routes/                   # API routes
│   ├── 📄 __init__.py
│   └── 📄 chat.py               # /ask endpoint handler
│
├── 📁 tools/                    # LangChain tools
│   ├── 📄 __init__.py
│   ├── 📄 search_tools.py       # Web, image, Wikipedia, YouTube search
│   ├── 📄 utility_tools.py      # Unit converter, calculator
│   ├── 📄 rag_tools.py          # FAISS vector store, car knowledge
│   └── 📄 ml_tools.py           # ML-powered analysis tools
│
├── 📁 ml/                       # Custom ML models
│   ├── 📄 __init__.py
│   ├── 📄 intent_classifier.py  # TF-IDF + Logistic Regression
│   ├── 📄 sentiment_analyzer.py # Review sentiment analysis
│   ├── 📄 car_ner.py            # spaCy NER for car entities
│   └── 📄 price_predictor.py    # Random Forest price prediction
│
├── 📁 data/                     # Data storage
│   ├── 📁 car_vectors/          # FAISS vector store
│   └── 📁 models/               # Trained ML model files (.joblib)
│
├── 📁 templates/                # HTML templates
│   └── 📄 index.html            # Main chat interface
│
├── 📁 static/                   # Static assets
│   ├── 📁 css/
│   ├── 📁 js/
│   └── 📁 images/
│
├── 📁 utils/                    # Utility functions
│   └── 📄 __init__.py
│
└── 📁 tests/                    # Test files
    ├── 📄 test_features.py
    └── 📄 test_ml_models.py
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/BimalaWijekoon/LangChain.git
cd LangChain

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download spaCy model (for NER)
python -m spacy download en_core_web_sm

# 6. Create environment file
copy .env.example .env  # Windows
# OR
cp .env.example .env    # macOS/Linux

# 7. Add your Gemini API key to .env
# Edit .env and add: GEMINI_API_KEY=your_key_here

# 8. Run the application
python app.py
```

### Access the Application

Open your browser and navigate to: **http://localhost:5000**

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
GEMINI_MODEL=gemini-2.5-flash     # LLM model to use
FLASK_ENV=development              # development or production
DEBUG=True                         # Enable debug mode
AGENT_VERBOSE=True                 # Show agent reasoning logs
AGENT_MAX_ITERATIONS=10            # Max tool calls per query
SEARCH_MAX_RESULTS=5               # Results per search
```

### Configuration Classes (`config/settings.py`)

```python
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "automind-secret-key")
    DEBUG = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    AGENT_MAX_ITERATIONS = 10
    AGENT_VERBOSE = True
    SEARCH_MAX_RESULTS = 5
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

---

## 💻 Usage

### Basic Queries

| Query Type | Example |
|------------|---------|
| **Specifications** | "What are the specs of the 2024 Toyota Supra?" |
| **Comparisons** | "Compare BMW M3 vs Mercedes C63 AMG" |
| **Pricing** | "How much is a used 2022 Ford Mustang GT worth?" |
| **Maintenance** | "What's the maintenance schedule for Honda Civic?" |
| **Images** | "Show me pictures of Porsche 911" |
| **Videos** | "Find YouTube reviews of Tesla Model 3" |
| **Calculations** | "Calculate monthly payment for $50,000 car at 7% APR for 60 months" |

### Image Analysis

1. Click the 📷 image button in the input area
2. Select a car image from your device
3. Preview appears above the input
4. Optionally add a question like "What year is this car?"
5. Send to get AI analysis

### Chat Management

- **New Chat**: Click "New Chat" button in sidebar
- **Switch Chats**: Click any chat in the history list
- **Rename**: Hover over chat → click ✏️ icon
- **Delete**: Hover over chat → click 🗑️ icon
- **Clear Current**: Click trash icon in header

---

## 🔌 API Endpoints

### POST `/ask`

Main endpoint for all queries.

**Request:**
```json
{
    "question": "What are the specs of the 2024 Toyota Supra?",
    "image": "data:image/jpeg;base64,..." // Optional
}
```

**Response:**
```json
{
    "response": "<p>HTML formatted response</p>",
    "response_raw": "Markdown response",
    "image_links": ["https://..."],
    "search_used": true,
    "image_analyzed": false,
    "intent": "specs",
    "agent_used": "Specs Expert",
    "tools_used": ["car_knowledge_search", "car_web_search"],
    "status": "success"
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request (missing question/image) |
| 500 | Server error |

---

## 🤖 LangChain Agent System

### ReAct Agent Pattern

AutoMind uses the **ReAct (Reasoning + Acting)** pattern via LangGraph:

```python
from langgraph.prebuilt import create_react_agent

# Create agent with tools
agent = create_react_agent(llm, tools)

# Agent reasoning loop:
# 1. Reason about the query
# 2. Select appropriate tool(s)
# 3. Execute tool and observe results
# 4. Reason about results
# 5. Repeat or generate final answer
```

### System Prompt

```
You are AutoMind, an enthusiastic and knowledgeable Car Expert AI assistant! 🚗

YOUR CAPABILITIES:
- 🔍 Web Search: Real-time car specs, prices, reviews
- 🖼️ Image Search: Find car photos and images
- 📚 Wikipedia: Car history, brand info
- 🎬 YouTube: Video reviews and demonstrations
- 📊 Knowledge Base: Fast RAG database lookup
- 🔢 Calculator: Unit conversions, fuel costs, payments
- 💬 Sentiment Analysis: Analyze car reviews
- 💰 Price Prediction: ML-based used car pricing
- 🏷️ Entity Extraction: Identify cars, specs, prices

TOOL USAGE STRATEGY:
1. For common cars, try car_knowledge_search FIRST (fast, offline)
2. For latest info or uncommon cars, use car_web_search
3. ALWAYS use car_image_search when user asks for pictures
4. Use specialized ML tools for analysis tasks
```

### Conversation Memory

```python
# Manual chat history tracking
self.chat_history: List = []  # Stores HumanMessage, AIMessage
self.max_history = 20         # Last 10 exchanges

# Include history in agent invocation
messages = [SystemMessage(content=system_prompt)]
messages.extend(self.chat_history[-6:])  # Last 3 exchanges
messages.append(HumanMessage(content=query))
```

---

## 👥 Multi-Agent Architecture

### Specialist Agents

AutoMind implements 4 specialized agents coordinated by an orchestrator:

```
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-AGENT ORCHESTRATOR                     │
│                                                              │
│   Query: "Compare BMW M3 vs Mercedes C63"                   │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                            │
│              │ Intent Routing  │                            │
│              │ (Pattern Match) │                            │
│              └─────────────────┘                            │
│                       │                                      │
│       ┌───────────────┼───────────────┬────────────────┐    │
│       ▼               ▼               ▼                ▼    │
│ ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│ │Comparison │  │  Pricing  │  │Maintenance│  │   Specs   │ │
│ │ Specialist│  │  Expert   │  │  Advisor  │  │  Expert   │ │
│ └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
│      ✓                                                      │
└─────────────────────────────────────────────────────────────┘
```

### Agent Routing Logic

```python
def route_query(self, query: str) -> str:
    """Route query to the best specialist."""
    query_lower = query.lower()
    
    # Keyword-based scoring
    scores = {
        "comparison": sum(1 for word in ["compare", "vs", "versus", "better", "difference"] 
                        if word in query_lower),
        "pricing": sum(1 for word in ["price", "cost", "worth", "value", "msrp", "afford"] 
                      if word in query_lower),
        "maintenance": sum(1 for word in ["maintenance", "reliable", "problem", "repair", "service"] 
                          if word in query_lower),
        "specs": sum(1 for word in ["specs", "horsepower", "engine", "0-60", "top speed"] 
                    if word in query_lower),
    }
    
    best_agent = max(scores, key=scores.get)
    return best_agent
```

### Specialist Definitions

| Agent | Specialty | System Prompt Focus |
|-------|-----------|---------------------|
| **Comparison Specialist** | Head-to-head comparisons | Objective analysis, tables, pros/cons |
| **Pricing Expert** | Prices and value | MSRP, used values, depreciation |
| **Maintenance Advisor** | Reliability and service | Common problems, service schedules |
| **Specs Expert** | Technical specifications | Detailed spec sheets, performance data |

---

## 🧠 Custom ML Models

### 1. Intent Classifier

**Purpose:** Classify user queries into intent categories.

**Architecture:**
```
TF-IDF Vectorizer → Logistic Regression → Intent Label
```

**Intents:**
| Intent | Example Query |
|--------|---------------|
| `specs` | "What's the horsepower of BMW M3?" |
| `comparison` | "Compare Toyota Camry vs Honda Accord" |
| `pricing` | "How much does a Mustang GT cost?" |
| `maintenance` | "Is Toyota reliable?" |
| `recommendation` | "What's a good first car?" |
| `buying_advice` | "Should I buy used or new?" |
| `image_request` | "Show me pictures of Ferrari" |
| `general` | "Tell me about sports cars" |
| `greeting` | "Hello!" |

**Code:**
```python
# ml/intent_classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train on labeled examples
pipeline.fit(texts, labels)
```

### 2. Sentiment Analyzer

**Purpose:** Analyze sentiment in car reviews and opinions.

**Architecture:**
```
TF-IDF Vectorizer → Logistic Regression → Sentiment (positive/negative/neutral)
```

**Output:**
```json
{
    "sentiment": "positive",
    "confidence": 0.92,
    "scores": {
        "positive": 0.92,
        "neutral": 0.05,
        "negative": 0.03
    }
}
```

**Code:**
```python
# ml/sentiment_analyzer.py
def analyze(self, text: str) -> Dict:
    prediction = self.pipeline.predict([text])[0]
    probabilities = self.pipeline.predict_proba([text])[0]
    
    return {
        "sentiment": prediction,
        "confidence": float(max(probabilities)),
        "scores": {
            "positive": float(probabilities[2]),
            "neutral": float(probabilities[1]),
            "negative": float(probabilities[0])
        }
    }
```

### 3. Car NER (Named Entity Recognition)

**Purpose:** Extract car-related entities from text.

**Entity Types:**
| Entity | Example |
|--------|---------|
| `CAR_MAKE` | Toyota, BMW, Ford |
| `CAR_MODEL` | Mustang, Camry, M3 |
| `YEAR` | 2024, 2023 |
| `SPEC_VALUE` | 500hp, 3.5 seconds, 25 mpg |
| `PRICE` | $50,000, 30k |

**Architecture:**
```
spaCy Pipeline → Custom Entity Recognizer Component → Pattern Matching + Rules
```

**Code:**
```python
# ml/car_ner.py
@Language.component("car_entity_recognizer")
def car_entity_recognizer(doc):
    """Custom spaCy component for car entities."""
    
    # Pattern matching for makes
    for make in CAR_MAKES:
        for match in re.finditer(r'\b' + re.escape(make) + r'\b', text_lower):
            span = doc.char_span(match.start(), match.end(), label="CAR_MAKE")
            if span: new_ents.append(span)
    
    # Regex for specs
    spec_patterns = [
        r'\b\d+\s*(hp|horsepower|bhp)\b',  # Horsepower
        r'\b\d+\.?\d*\s*(mpg|miles per gallon)\b',  # Fuel economy
        # ... more patterns
    ]
```

### 4. Price Predictor

**Purpose:** Predict used car prices based on vehicle attributes.

**Features:**
| Feature | Type | Description |
|---------|------|-------------|
| `make` | Categorical | Manufacturer |
| `model` | Categorical | Model name |
| `year` | Numeric | Model year |
| `mileage` | Numeric | Odometer reading |
| `condition` | Categorical | new/excellent/good/fair/poor |
| `engine_hp` | Numeric | Horsepower |
| `mpg` | Numeric | Fuel economy |
| `body_type` | Categorical | sedan/coupe/suv/truck |

**Architecture:**
```
Feature Engineering → Label Encoding → Standard Scaling → Random Forest Regressor
```

**Output:**
```json
{
    "estimated_price": 35500,
    "price_range": {
        "low": 33725,
        "high": 37275
    },
    "confidence": 0.87,
    "factors": "Year, mileage, and condition significantly impact the price."
}
```

**Code:**
```python
# ml/price_predictor.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CarPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
```

---

## 📚 RAG (Retrieval Augmented Generation)

### Vector Database

AutoMind uses **FAISS** (Facebook AI Similarity Search) for fast similarity search:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vector_store = FAISS.from_documents(documents, embeddings)

# Similarity search
results = vector_store.similarity_search(query, k=3)
```

### Car Knowledge Base

Pre-loaded specifications for common cars:

```python
CAR_KNOWLEDGE_BASE = [
    {
        "category": "sports_car",
        "name": "Toyota GR Supra",
        "year": "2024",
        "specs": {
            "engine": "3.0L Twin-Scroll Turbo I6",
            "horsepower": 382,
            "torque": "368 lb-ft",
            "0_60": "3.9 seconds",
            "top_speed": "155 mph",
            "transmission": "8-speed automatic",
            "drivetrain": "RWD",
            "mpg": "22 city / 30 highway"
        },
        "description": "The Toyota GR Supra is a legendary sports car reborn..."
    },
    # ... more cars
]
```

### Categories Covered

- 🏎️ Sports Cars (Supra, Corvette, Mustang, 911, etc.)
- 🚗 Sedans (Camry, Accord, 3 Series, C-Class, etc.)
- 🚙 SUVs (RAV4, CR-V, X5, GLE, etc.)
- 🛻 Trucks (F-150, Silverado, Tacoma, etc.)
- ⚡ Electric (Model S, Model 3, Taycan, etc.)

---

## 🔧 Tools & Capabilities

### Search Tools (`tools/search_tools.py`)

| Tool | Function | Description |
|------|----------|-------------|
| `car_web_search` | DuckDuckGo | Real-time web search for car info |
| `car_image_search` | DuckDuckGo Images | Search and display car images |
| `car_wikipedia_search` | Wikipedia API | Historical and educational content |
| `youtube_car_videos` | YouTube Search | Find video reviews and content |

### Utility Tools (`tools/utility_tools.py`)

| Tool | Function | Description |
|------|----------|-------------|
| `unit_converter` | Conversion | mph↔km/h, mpg↔L/100km, etc. |
| `car_calculator` | Math | Loan payments, fuel costs, depreciation |

### RAG Tools (`tools/rag_tools.py`)

| Tool | Function | Description |
|------|----------|-------------|
| `car_knowledge_search` | FAISS | Fast offline spec lookups |

### ML Tools (`tools/ml_tools.py`)

| Tool | Function | Description |
|------|----------|-------------|
| `analyze_car_review` | Sentiment | Analyze review sentiment |
| `extract_car_entities` | NER | Extract makes, models, specs |
| `predict_car_price` | Regression | Estimate used car values |
| `classify_user_intent` | Classification | Determine query intent |

---

## 🎨 Frontend Interface

### Design System

| Element | Value |
|---------|-------|
| **Primary Background** | `#0a0a0b` |
| **Secondary Background** | `#111113` |
| **Accent Color** | `#ff4d4d` (red) |
| **Secondary Accent** | `#ff8c42` (orange) |
| **Text Primary** | `#ffffff` |
| **Text Secondary** | `#9a9a9f` |
| **Border** | `#2a2a2e` |

### Fonts

- **Primary:** Inter (sans-serif)
- **Accent/Branding:** Orbitron (display)

### Chat Interface

```
┌──────────────────────────────────────────────────────────────┐
│ ┌──────────────┐ ┌─────────────────────────────────────────┐ │
│ │              │ │                                         │ │
│ │  [New Chat]  │ │         Current Chat Title        [🗑️]  │ │
│ │              │ │                                         │ │
│ │ Recent Chats │ ├─────────────────────────────────────────┤ │
│ │              │ │                                         │ │
│ │ 💬 Chat 1    │ │         Welcome to AutoMind! 🚗         │ │
│ │ 💬 Chat 2    │ │                                         │ │
│ │ 💬 Chat 3    │ │    Ask me anything about cars...        │ │
│ │              │ │                                         │ │
│ │              │ │  [Quick Action] [Quick Action] [Quick]  │ │
│ │              │ │                                         │ │
│ │              │ │                                         │ │
│ │              │ │     👤 User message                     │ │
│ │              │ │                                         │ │
│ │              │ │     🏎️ AutoMind response                │ │
│ │              │ │                                         │ │
│ │──────────────│ ├─────────────────────────────────────────┤ │
│ │ 🏎️ AUTOMIND  │ │  [📷] [Ask about any car...    ] [➤]   │ │
│ └──────────────┘ └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Features

1. **Sidebar**
   - New Chat button
   - Chat history list
   - Rename/delete actions
   - Brand footer

2. **Chat Area**
   - Welcome screen with quick actions
   - Message bubbles (user/assistant)
   - Markdown rendering
   - Image display support

3. **Input Area**
   - Image upload button
   - Text input (auto-resize)
   - Send button
   - Image preview

4. **Responsive Design**
   - Collapsible sidebar on mobile
   - Overlay backdrop
   - Touch-friendly buttons

---

## 🔬 Development

### Running Tests

```bash
# Test ML models
python test_ml_models.py

# Test features
python test_features.py

# Run with verbose logging
AGENT_VERBOSE=True python app.py
```

### Adding New Tools

1. Create tool in `tools/` directory:

```python
from langchain_core.tools import tool

@tool
def my_new_tool(query: str) -> str:
    """
    Tool description for the LLM.
    
    Args:
        query: What to process
    
    Returns:
        Result string
    """
    # Implementation
    return result
```

2. Import in `agents/car_expert.py`:

```python
from tools.my_tools import my_new_tool
```

3. Add to tools list:

```python
self.tools = [
    # ... existing tools
    my_new_tool,
]
```

### Adding New ML Models

1. Create model in `ml/` directory
2. Include training data
3. Implement `train()`, `save()`, `load()`, and prediction methods
4. Create LangChain tool wrapper in `tools/ml_tools.py`
5. Add to agent's tool list

---

## 📖 NLP Concepts Used

| Concept | Implementation | Description |
|---------|----------------|-------------|
| **ReAct Agents** | LangGraph | Reasoning + Acting pattern |
| **Multi-Agent Systems** | Specialist routing | Divide-and-conquer approach |
| **RAG** | FAISS + HuggingFace | Retrieval Augmented Generation |
| **Tool Use** | LangChain Tools | Extending LLM capabilities |
| **Intent Classification** | TF-IDF + LogReg | Understanding user intent |
| **Sentiment Analysis** | ML Classification | Opinion mining |
| **NER** | spaCy + Patterns | Entity extraction |
| **Embeddings** | Sentence Transformers | Semantic similarity |
| **Prompt Engineering** | System prompts | Guiding LLM behavior |
| **Conversation Memory** | Message history | Context preservation |
| **Vision-Language** | Gemini Vision | Multimodal understanding |

---

## 🚧 Future Improvements

- [ ] **Database Integration** - PostgreSQL for persistent chat storage
- [ ] **User Authentication** - Login/signup functionality
- [ ] **Advanced RAG** - Hybrid search (dense + sparse)
- [ ] **Voice Input** - Speech-to-text support
- [ ] **Export Chats** - Download conversation history
- [ ] **API Rate Limiting** - Production-ready throttling
- [ ] **Model Fine-tuning** - Custom car-specific LLM
- [ ] **Real-time Prices** - API integration for live pricing
- [ ] **Car Image Recognition** - Identify car from photos
- [ ] **Multilingual Support** - Spanish, German, Japanese

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Bimala Wijekoon**

- GitHub: [@BimalaWijekoon](https://github.com/BimalaWijekoon)
- Project: [AutoMind Repository](https://github.com/BimalaWijekoon/LangChain)

---

<div align="center">

**Built with ❤️ and ☕ for Car Enthusiasts**

🏎️ AutoMind - Your AI Car Expert 🏎️

</div>
