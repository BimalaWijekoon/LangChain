"""
Car Expert Agent Module - LangChain + LangGraph Edition
========================================================

This module implements a proper LangChain-based AI agent for car-related queries.

NLP/LangChain Concepts Used:
1. LangGraph ReAct Agent - Modern agent framework for tool usage
2. Tool Integration - Multiple tools (DuckDuckGo, Wikipedia, YouTube, Calculator)
3. Conversation Memory - Manual chat history tracking
4. RAG (Retrieval Augmented Generation) - Vector database for car specs
5. Multi-Agent Architecture - Specialized agents for different queries
6. Structured Output - Pydantic models for data parsing
7. Intent Classification - Pattern matching for user intent
8. Named Entity Recognition (basic) - Extracting car names

Architecture:
- LangGraph Agent: Handles reasoning, tool selection, and responses
- Multi-Agent: Specialists for comparisons, pricing, maintenance
- RAG System: FAISS vector store with car knowledge base
- Gemini Vision: Direct API for image analysis only
"""

import re
import base64
from typing import Optional, List, Dict, Any

# LangChain Core Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# LangChain Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph - Modern Agent Framework
from langgraph.prebuilt import create_react_agent

# Gemini Vision (direct API for image analysis only)
from google import genai
from google.genai import types

# Local imports
from config.settings import get_config

# Import our new tools
from tools.search_tools import car_web_search, car_image_search, car_wikipedia_search, youtube_car_videos
from tools.utility_tools import unit_converter, car_calculator
from tools.rag_tools import car_knowledge_search

# Import ML-powered tools
from tools.ml_tools import analyze_car_review, extract_car_entities, predict_car_price, classify_user_intent

# Import multi-agent orchestrator
from agents.multi_agent import MultiAgentOrchestrator


class CarExpertAgent:
    """
    LangGraph-powered Car Expert Agent with Multi-Agent Support.
    
    This agent uses:
    - LangGraph ReAct Agent for reasoning and tool usage
    - Multiple Tools: Web search, Wikipedia, YouTube, Calculator, RAG
    - Multi-Agent Orchestrator for specialized queries
    - Manual conversation history for context awareness
    - Gemini Vision API for image analysis
    - RAG with FAISS for fast offline car data lookup
    """
    
    # Intent patterns for classifying user input
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|howdy|sup|yo)[\s!?.]*$',
        r'^how\s*(are|r)\s*(you|u|ya)[\s!?.]*$',
        r'^what\'?s\s*up[\s!?.]*$',
    ]
    
    THANKS_PATTERNS = [
        r'^(thanks|thank\s*you|thx|ty)[\s!?.]*$',
    ]
    
    FAREWELL_PATTERNS = [
        r'^(bye|goodbye|see\s*ya|later|cya)[\s!?.]*$',
    ]
    
    HELP_PATTERNS = [
        r'^(help|what\s*(can|do)\s*(you|u)\s*do|who\s*(are|r)\s*(you|u))[\s!?.]*$',
    ]
    
    def __init__(self, use_multi_agent: bool = True):
        """Initialize the LangGraph Car Expert Agent.
        
        Args:
            use_multi_agent: If True, use specialized agents for different query types
        """
        self.config = get_config()
        self.use_multi_agent = use_multi_agent
        
        # Chat history for context
        self.chat_history: List = []
        self.max_history = 20
        
        # Initialize components
        self._setup_llm()
        self._setup_agent()
        self._setup_vision()
        
        # Setup multi-agent system (lazy init to avoid slow startup)
        self._multi_agent = None
        
        print("[AutoMind] LangGraph Agent initialized successfully!")
        if use_multi_agent:
            print("[AutoMind] Multi-Agent mode enabled (specialists: comparison, pricing, maintenance)")
    
    def _setup_llm(self):
        """
        Set up the Language Model using LangChain's Google GenAI integration.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.GEMINI_MODEL,
            google_api_key=self.config.GEMINI_API_KEY,
            temperature=0.7,
        )
        
        if self.config.AGENT_VERBOSE:
            print(f"[AutoMind] LLM initialized: {self.config.GEMINI_MODEL}")
    
    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return """You are AutoMind, an enthusiastic and knowledgeable Car Expert AI assistant! 🚗

## YOUR PERSONALITY:
- You're passionate about cars and LOVE helping people
- Be friendly, engaging, and enthusiastic
- Use car emojis occasionally (🚗 🏎️ 💨 🔧 ⚡)

## YOUR CAPABILITIES:
- 🔍 **Web Search**: Real-time car specs, prices, reviews (car_web_search)
- �️ **Image Search**: Find car photos and images (car_image_search)
- 📚 **Wikipedia**: Car history, brand info, technical terms (car_wikipedia_search)
- 🎬 **YouTube**: Video reviews and demonstrations (youtube_car_videos)
- 📊 **Knowledge Base**: Fast lookup from RAG database (car_knowledge_search)
- 🔢 **Calculator**: Unit conversions, fuel costs, payments (unit_converter, car_calculator)
- 💬 **Sentiment Analysis**: Analyze car reviews (analyze_car_review)
- 💰 **Price Prediction**: ML-based used car pricing (predict_car_price)
- 🏷️ **Entity Extraction**: Identify cars, specs, prices in text (extract_car_entities)

## TOOL USAGE STRATEGY:
1. For common cars, try car_knowledge_search FIRST (fast, offline)
2. For latest info or uncommon cars, use car_web_search
3. For car history/technical terms, use car_wikipedia_search
4. For video reviews, use youtube_car_videos
5. For conversions/calculations, use unit_converter or car_calculator
6. For analyzing reviews or opinions, use analyze_car_review
7. For price estimates of used cars, use predict_car_price
8. To understand what cars/specs user mentions, use extract_car_entities

## CRITICAL RULES:
1. NEVER guess specific numbers (horsepower, price, 0-60 times)
2. ALWAYS use tools to find factual data
3. If search doesn't return relevant info, say so
4. Use bullet points for specifications
5. Create comparison tables when comparing cars
6. Be thorough but concise

## RESPONSE FORMAT:
- Use **bold** for important specs and car names
- Use bullet points for lists of specifications
- Include the model year when mentioning specs
- Cite your sources (web search, knowledge base, etc.)"""
    
    def _setup_agent(self):
        """
        Set up the LangGraph ReAct Agent with all tools.
        
        LangGraph provides a modern, graph-based approach to building agents.
        The ReAct agent:
        1. Receives a question
        2. Thinks about what to do  
        3. Decides if it needs to use a tool
        4. Uses the tool and observes the result
        5. Continues until it has a final answer
        """
        # Define all tools for the agent
        tools = [
            car_web_search,       # DuckDuckGo web search
            car_image_search,      # DuckDuckGo image search
            car_wikipedia_search,  # Wikipedia for history/info
            youtube_car_videos,    # YouTube video search
            car_knowledge_search,  # RAG vector database
            unit_converter,        # Unit conversions
            car_calculator,        # Car calculations
            # ML-powered tools
            analyze_car_review,    # Sentiment analysis
            predict_car_price,     # Price prediction
            extract_car_entities,  # NER entity extraction
        ]
        
        # Create the agent using LangGraph's prebuilt ReAct agent
        self.agent = create_react_agent(
            self.llm,
            tools=tools,
        )
        
        self.tools = tools  # Store for reference
        
        if self.config.AGENT_VERBOSE:
            print("[AutoMind] LangGraph ReAct Agent initialized")
            print(f"[AutoMind] Tools: {[t.name for t in tools]}")
    
    def _setup_vision(self):
        """
        Set up Gemini Vision for image analysis.
        
        This uses the new google.genai SDK for multimodal support.
        """
        # Initialize the new GenAI client
        self.genai_client = genai.Client(api_key=self.config.GEMINI_API_KEY)
        self.vision_model_name = self.config.GEMINI_MODEL
        
        self.vision_system_instruction = """You are AutoMind's vision system, specialized in identifying cars.

When analyzing a car image:
1. Identify the make, model, and approximate year
2. Describe notable features (body style, color, wheels, modifications)
3. Share interesting facts about this car model
4. If you see damage or issues, mention them
5. Be enthusiastic - you love cars! 🚗

If the image isn't a car, politely mention you specialize in cars."""
        
        if self.config.AGENT_VERBOSE:
            print("[AutoMind] Vision model initialized (google.genai SDK)")
    
    def _classify_intent(self, text: str) -> str:
        """
        Classify the intent of user input using pattern matching.
        
        This is a simple NLP technique - in production you might use
        a trained classifier or embeddings for better accuracy.
        
        Returns: 'greeting', 'thanks', 'farewell', 'help', or 'car_query'
        """
        text_lower = text.lower().strip()
        
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return 'greeting'
        
        for pattern in self.THANKS_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return 'thanks'
        
        for pattern in self.FAREWELL_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return 'farewell'
        
        for pattern in self.HELP_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return 'help'
        
        return 'car_query'
    
    def _handle_simple_intent(self, intent: str) -> str:
        """Handle simple intents without using the full agent."""
        responses = {
            'greeting': "Hey there, petrolhead! 🚗💨 Great to see you! What car can I help you with today?",
            'thanks': "You're welcome! 🏎️ Always happy to help a fellow car enthusiast!",
            'farewell': "See you later! 🚗💨 Keep those engines running!",
            'help': """Hey! I'm **AutoMind**, your AI car expert powered by LangChain + LangGraph! 🚗

**🔧 My Tools:**
• 🔍 **Web Search** - Real-time specs, prices, reviews
• 📚 **Wikipedia** - Car history and technical info
• 🎬 **YouTube** - Video reviews and walkarounds
• 📊 **Knowledge Base** - Fast RAG lookup for popular cars
• 🔢 **Calculator** - Unit conversions, loan payments, fuel costs

**🤖 Specialist Agents:**
• 🏁 **Comparison Expert** - "BMW M3 vs Mercedes C63"
• 💰 **Pricing Expert** - "How much is a Tesla Model 3?"
• 🔧 **Maintenance Advisor** - "Toyota Camry reliability"

**📸 Image Analysis:**
• Upload a car photo and I'll identify it!

**💬 Memory:**
• I remember our conversation context!

**Tech:** LangChain + LangGraph + FAISS RAG + Gemini Vision

Just ask me anything about cars! 🏎️"""
        }
        return responses.get(intent, "How can I help you with cars today?")
    
    @property
    def multi_agent(self) -> MultiAgentOrchestrator:
        """Lazy initialization of multi-agent orchestrator."""
        if self._multi_agent is None:
            self._multi_agent = MultiAgentOrchestrator()
        return self._multi_agent
    
    def _extract_car_name(self, text: str) -> str:
        """
        Extract car name from text using pattern matching (simple NER).
        """
        brands = [
            "toyota", "honda", "ford", "chevrolet", "chevy", "bmw", "mercedes",
            "audi", "porsche", "ferrari", "lamborghini", "nissan", "mazda",
            "subaru", "volkswagen", "vw", "volvo", "jaguar", "tesla", "lexus",
            "hyundai", "kia", "dodge", "jeep", "corvette", "mustang", "camaro",
            "supra", "civic", "accord", "corolla", "camry", "m3", "m4", "gtr"
        ]
        
        text_lower = text.lower()
        found = []
        
        for brand in brands:
            if brand in text_lower:
                pattern = rf'{brand}[\s\-]?[\w\d\-]*(?:\s+[\w\d\-]+)?'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                found.extend(matches)
        
        return max(found, key=len) if found else ""
    
    def _generate_image_links(self, car_name: str) -> List[Dict[str, str]]:
        """Generate helpful links for car images and info."""
        if not car_name:
            return []
        
        encoded = car_name.replace(" ", "+")
        return [
            {"title": "📸 Google Images", "url": f"https://www.google.com/search?tbm=isch&q={encoded}+car"},
            {"title": "🔍 DuckDuckGo", "url": f"https://duckduckgo.com/?q={encoded}&iax=images&ia=images"},
            {"title": "🎬 YouTube", "url": f"https://www.youtube.com/results?search_query={encoded}+review"}
        ]
    
    def _format_chat_history(self) -> List:
        """Format chat history for the agent."""
        messages = [SystemMessage(content=self._get_system_message())]
        
        # Add recent history
        for msg in self.chat_history[-self.max_history:]:
            messages.append(msg)
        
        return messages
    
    def _should_use_specialist(self, question: str) -> bool:
        """
        Determine if a specialist agent should handle this query.
        
        Specialists are better for:
        - Comparisons (vs, compare, better)
        - Pricing (cost, price, afford)
        - Maintenance (reliability, problems, service)
        """
        specialist_patterns = [
            r'\bvs\b', r'\bversus\b', r'\bcompare\b', r'\bbetter\b',
            r'\bprice\b', r'\bcost\b', r'\bafford\b', r'\bbudget\b',
            r'\bmaintenance\b', r'\breliab', r'\bproblem\b', r'\brepair\b'
        ]
        question_lower = question.lower()
        return any(re.search(p, question_lower) for p in specialist_patterns)
    
    def ask(self, question: str, image_data: str = None) -> Dict[str, Any]:
        """
        Process a user question using the LangGraph agent or Multi-Agent system.
        
        Args:
            question: User's question about cars
            image_data: Base64 encoded image (optional, for vision)
            
        Returns:
            Dictionary with response, image_links, and metadata
        """
        try:
            # If there's an image, use vision model
            if image_data:
                return self._analyze_image(question, image_data)
            
            # Classify intent
            intent = self._classify_intent(question)
            
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Intent: {intent}")
            
            # Handle simple intents directly (no agent needed)
            if intent != 'car_query':
                response = self._handle_simple_intent(intent)
                # Still save to history
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=response))
                return {
                    "response": response,
                    "image_links": [],
                    "search_used": False,
                    "intent": intent
                }
            
            # Check if we should use a specialist agent
            if self.use_multi_agent and self._should_use_specialist(question):
                return self._run_multi_agent(question)
            
            # Use main LangGraph agent for general car queries
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Running LangGraph agent for: {question}")
            
            # Build messages with history
            messages = self._format_chat_history()
            messages.append(HumanMessage(content=question))
            
            # Invoke the agent
            result = self.agent.invoke({"messages": messages})
            
            # Extract response from result
            response_messages = result.get("messages", [])
            response = ""
            for msg in reversed(response_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    response = msg.content
                    break
            
            if not response:
                response = "I couldn't generate a response. Please try again!"
            
            # Save to history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            
            # Extract car name for image links
            car_name = self._extract_car_name(question + " " + response)
            image_links = self._generate_image_links(car_name)
            
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Response generated, history size: {len(self.chat_history)}")
            
            return {
                "response": response,
                "image_links": image_links,
                "search_used": True,
                "intent": "car_query"
            }
            
        except Exception as e:
            error_msg = str(e)
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Error: {error_msg}")
            
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "response": "⏳ I'm being rate-limited. Please wait a moment and try again!",
                    "image_links": [],
                    "search_used": False
                }
            
            return {
                "response": f"😅 Oops! Something went wrong: {error_msg}",
                "image_links": [],
                "search_used": False
            }
    
    def _run_multi_agent(self, question: str) -> Dict[str, Any]:
        """
        Route query to the appropriate specialist agent.
        
        Multi-agent architecture enables specialized handling of:
        - Comparisons (Comparison Specialist)
        - Pricing (Pricing Expert)
        - Maintenance (Maintenance Advisor)
        """
        try:
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Using Multi-Agent system for: {question}")
            
            # Run the multi-agent orchestrator
            result = self.multi_agent.run(question, self.chat_history)
            
            response = result.get("response", "")
            agent_used = result.get("agent_used", "Unknown")
            tools_used = result.get("tools_used", [])
            
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Specialist used: {agent_used}")
                print(f"[AutoMind] Tools used: {tools_used}")
            
            # Add agent attribution to response
            response_with_attr = f"{response}\n\n---\n_Answered by: **{agent_used}**_"
            
            # Save to history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            
            # Extract car name for links
            car_name = self._extract_car_name(question + " " + response)
            image_links = self._generate_image_links(car_name)
            
            return {
                "response": response_with_attr,
                "image_links": image_links,
                "search_used": True,
                "intent": "car_query",
                "agent_used": agent_used,
                "tools_used": tools_used
            }
            
        except Exception as e:
            error_msg = str(e)
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Multi-agent error: {error_msg}")
            
            # Fallback to main agent
            return self.ask(question)
    
    def _analyze_image(self, question: str, image_data: str) -> Dict[str, Any]:
        """
        Analyze an uploaded car image using Gemini Vision.
        
        This uses the new google.genai SDK for multimodal (image + text) processing.
        Image analysis is kept separate from LangChain for better reliability.
        """
        try:
            if self.config.AGENT_VERBOSE:
                print("[AutoMind] Analyzing image with Gemini Vision (google.genai)...")
            
            # Decode base64 image
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Create image part using new google.genai types
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
            
            prompt = question if question else "What car is this? Tell me everything about it!"
            
            # Build the content with system instruction and user message
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt), image_part]
                )
            ]
            
            # Generate response using new client API
            response = self.genai_client.models.generate_content(
                model=self.vision_model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.vision_system_instruction,
                    temperature=0.7
                )
            )
            
            # Save to chat history
            self.chat_history.append(HumanMessage(content=f"[Uploaded car image] {prompt}"))
            self.chat_history.append(AIMessage(content=response.text))
            
            return {
                "response": response.text,
                "image_links": [],
                "search_used": False,
                "image_analyzed": True
            }
            
        except Exception as e:
            error_msg = str(e)
            if self.config.AGENT_VERBOSE:
                print(f"[AutoMind] Vision error: {error_msg}")
            
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "response": "⏳ Image analysis rate-limited. Please wait a moment!",
                    "image_links": [],
                    "search_used": False
                }
            
            return {
                "response": f"😅 I had trouble analyzing that image: {error_msg}",
                "image_links": [],
                "search_used": False
            }
    
    def clear_memory(self):
        """Clear conversation memory to start fresh."""
        self.chat_history = []
        if self.config.AGENT_VERBOSE:
            print("[AutoMind] Memory cleared")
    
    def health_check(self) -> bool:
        """Check if the agent is properly configured."""
        try:
            test = self.llm.invoke("Say OK")
            return True
        except Exception:
            return False
