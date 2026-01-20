"""
Multi-Agent System for AutoMind
===============================

This module implements specialized agents for different car-related tasks.
A supervisor agent routes queries to the appropriate specialist.

NLP/LangChain Concepts:
- Multi-Agent Architecture: Multiple specialized agents working together
- Agent Routing: Directing queries to the appropriate expert
- LangGraph Workflows: Building complex agent interactions
- Tool Specialization: Each agent has domain-specific tools
"""

from typing import List, Dict, Any, Literal, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config.settings import get_config
from tools.search_tools import car_web_search, car_wikipedia_search, youtube_car_videos
from tools.utility_tools import unit_converter, car_calculator
from tools.rag_tools import car_knowledge_search


class AgentResponse(BaseModel):
    """Structured response from any agent."""
    content: str
    agent_name: str
    tools_used: List[str] = []
    confidence: float = 0.9


# ============================================
# SPECIALIZED TOOLS FOR EACH AGENT
# ============================================

@tool
def search_car_prices(query: str) -> str:
    """
    Search for current car prices, deals, and market values.
    
    Use for: MSRP, invoice prices, used car values, deals, incentives
    
    Args:
        query: Price-related query like 'Toyota Camry 2024 price' or 'best car deals'
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(f"{query} price MSRP cost deals 2024 2025")
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        return f"Price search error: {str(e)}"


@tool
def search_car_reliability(query: str) -> str:
    """
    Search for car reliability ratings and common problems.
    
    Use for: Reliability ratings, common issues, recalls, long-term reviews
    
    Args:
        query: Reliability query like 'Toyota Camry reliability' or 'BMW X5 problems'
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(f"{query} reliability rating problems issues consumer reports JD Power")
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        return f"Reliability search error: {str(e)}"


@tool
def search_maintenance_info(query: str) -> str:
    """
    Search for car maintenance schedules and costs.
    
    Use for: Service intervals, maintenance costs, DIY guides, part prices
    
    Args:
        query: Maintenance query like 'Honda Accord oil change interval' or 'brake pad replacement cost'
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(f"{query} maintenance schedule service interval cost repair")
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        return f"Maintenance search error: {str(e)}"


@tool
def compare_cars_specs(query: str) -> str:
    """
    Search for car comparison data and head-to-head reviews.
    
    Use for: Comparing two or more cars, which is better, head-to-head tests
    
    Args:
        query: Comparison query like 'Mustang vs Camaro comparison' or 'best midsize SUV 2024'
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(f"{query} comparison vs versus review head to head 2024")
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        return f"Comparison search error: {str(e)}"


# ============================================
# SPECIALIZED AGENTS
# ============================================

class SpecialistAgent:
    """Base class for specialized agents."""
    
    def __init__(self, name: str, llm, tools: List, system_prompt: str):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent = create_react_agent(llm, tools)
    
    def run(self, query: str, chat_history: List = None) -> AgentResponse:
        """Execute the agent with the given query."""
        messages = [SystemMessage(content=self.system_prompt)]
        
        if chat_history:
            messages.extend(chat_history[-6:])  # Last 3 exchanges
        
        messages.append(HumanMessage(content=query))
        
        try:
            result = self.agent.invoke({"messages": messages})
            
            # Extract the final response
            final_message = result["messages"][-1]
            raw_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Handle content that may be a list (structured content from LangChain)
            if isinstance(raw_content, list):
                # Extract text from structured content
                text_parts = []
                for item in raw_content:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = '\n'.join(text_parts) if text_parts else str(raw_content)
            else:
                content = raw_content
            
            # Track tools used
            tools_used = []
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tools_used.extend([tc['name'] for tc in msg.tool_calls])
            
            return AgentResponse(
                content=content,
                agent_name=self.name,
                tools_used=list(set(tools_used)),
                confidence=0.9
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"I encountered an issue: {str(e)}",
                agent_name=self.name,
                tools_used=[],
                confidence=0.3
            )


def create_comparison_agent(llm) -> SpecialistAgent:
    """
    Create an agent specialized in comparing cars.
    
    This agent excels at:
    - Head-to-head comparisons
    - Ranking cars by specific criteria
    - Pros and cons analysis
    - Recommendation based on needs
    """
    tools = [car_web_search, car_knowledge_search, compare_cars_specs, unit_converter]
    
    system_prompt = """You are the Car Comparison Specialist for AutoMind! 🏁

Your expertise is comparing cars and helping users make decisions.

## YOUR APPROACH:
1. Always gather specs for ALL cars being compared
2. Create clear comparison tables
3. Highlight key differences
4. Give a balanced recommendation

## COMPARISON FORMAT:
| Spec | Car A | Car B | Winner |
|------|-------|-------|--------|

## ALWAYS CONSIDER:
- Performance (HP, 0-60, handling)
- Value (price, features per dollar)
- Practicality (space, fuel economy)
- Reliability (ratings, common issues)
- Fun factor (driving experience)

End with a clear recommendation based on user's priorities."""

    return SpecialistAgent("Comparison Specialist", llm, tools, system_prompt)


def create_pricing_agent(llm) -> SpecialistAgent:
    """
    Create an agent specialized in car pricing and deals.
    
    This agent excels at:
    - MSRP and invoice prices
    - Used car valuations
    - Total cost of ownership
    - Deals and incentives
    """
    tools = [search_car_prices, car_calculator, car_knowledge_search]
    
    system_prompt = """You are the Car Pricing Expert for AutoMind! 💰

Your expertise is helping users understand car costs and find the best deals.

## YOUR APPROACH:
1. Search for current market prices
2. Calculate total cost of ownership when relevant
3. Consider depreciation and resale value
4. Look for current deals and incentives

## PRICING BREAKDOWN FORMAT:
- **MSRP:** $XX,XXX
- **Invoice Price:** $XX,XXX (typically 5-8% below MSRP)
- **Average Transaction Price:** $XX,XXX
- **Estimated Monthly Payment:** $XXX (with assumptions)

## ALWAYS MENTION:
- Destination charges (usually $1,000-1,500)
- Common dealer add-ons to avoid
- Best time to buy (end of month/year)
- Negotiation tips

Be transparent about price ranges and market conditions."""

    return SpecialistAgent("Pricing Expert", llm, tools, system_prompt)


def create_maintenance_agent(llm) -> SpecialistAgent:
    """
    Create an agent specialized in car maintenance.
    
    This agent excels at:
    - Maintenance schedules
    - Reliability information
    - Common problems
    - DIY vs dealer service
    """
    tools = [search_maintenance_info, search_car_reliability, car_knowledge_search, car_wikipedia_search]
    
    system_prompt = """You are the Car Maintenance Advisor for AutoMind! 🔧

Your expertise is helping users maintain their cars and understand reliability.

## YOUR APPROACH:
1. Search for manufacturer maintenance schedules
2. Research common problems and recalls
3. Provide estimated maintenance costs
4. Suggest DIY vs professional service

## MAINTENANCE FORMAT:
**Routine Maintenance Schedule:**
- Oil Change: Every X miles
- Tire Rotation: Every X miles
- Brake Inspection: Every X miles
- Major Service: Every X miles

**Common Issues to Watch:**
- Issue 1: Description and typical cost
- Issue 2: Description and typical cost

**Reliability Rating:** X/10 (cite source)

## ALWAYS EMPHASIZE:
- Following manufacturer schedules
- Using correct fluids and parts
- Warning signs to watch for
- Estimated annual maintenance costs"""

    return SpecialistAgent("Maintenance Advisor", llm, tools, system_prompt)


def create_specs_agent(llm) -> SpecialistAgent:
    """
    Create an agent specialized in car specifications.
    
    This agent excels at:
    - Detailed specifications
    - Performance data
    - Technical explanations
    - Feature breakdowns
    """
    tools = [car_web_search, car_knowledge_search, car_wikipedia_search, youtube_car_videos, unit_converter]
    
    system_prompt = """You are the Car Specifications Expert for AutoMind! 📊

Your expertise is providing detailed, accurate car specifications.

## YOUR APPROACH:
1. First check the knowledge base for common cars
2. Search the web for the latest/specific data
3. Provide comprehensive spec sheets
4. Explain technical terms when helpful

## SPEC SHEET FORMAT:
**[Car Name] ([Year]) Specifications**

🏎️ **Performance:**
- Engine: [description]
- Horsepower: [X] hp @ [X] rpm
- Torque: [X] lb-ft @ [X] rpm
- 0-60 mph: [X] seconds
- Top Speed: [X] mph

⚙️ **Drivetrain:**
- Transmission: [type]
- Drivetrain: [FWD/RWD/AWD]

⛽ **Efficiency:**
- City: [X] MPG
- Highway: [X] MPG
- Range: [X] miles (if electric)

📏 **Dimensions:**
- Weight: [X] lbs
- Cargo: [X] cu ft

## ALWAYS:
- Cite the model year for specs
- Note if data is estimated or official
- Offer to provide video reviews if relevant"""

    return SpecialistAgent("Specs Expert", llm, tools, system_prompt)


# ============================================
# MULTI-AGENT ORCHESTRATOR
# ============================================

class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents.
    
    Routes queries to the appropriate specialist based on intent.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize shared LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.GEMINI_MODEL,
            google_api_key=self.config.GEMINI_API_KEY,
            temperature=0.7,
        )
        
        # Create specialized agents
        self.agents = {
            "comparison": create_comparison_agent(self.llm),
            "pricing": create_pricing_agent(self.llm),
            "maintenance": create_maintenance_agent(self.llm),
            "specs": create_specs_agent(self.llm),
        }
        
        # Query patterns for routing
        self.routing_patterns = {
            "comparison": [
                r'\bvs\b', r'\bversus\b', r'\bcompare\b', r'\bcomparison\b',
                r'\bbetter\b', r'\bworse\b', r'\bor\b.*\bcar\b',
                r'\bwhich\s+(one|car|should)\b', r'\bdifference\b'
            ],
            "pricing": [
                r'\bprice\b', r'\bcost\b', r'\bmsrp\b', r'\bafford\b',
                r'\bdeal\b', r'\bbuy\b', r'\blease\b', r'\bfinance\b',
                r'\bworth\b', r'\bvalue\b', r'\bdepreciation\b', r'\bbudget\b',
                r'\bhow\s+much\b', r'\bpayment\b'
            ],
            "maintenance": [
                r'\bmaintenance\b', r'\brepair\b', r'\bservice\b',
                r'\bproblem\b', r'\bissue\b', r'\breliab', r'\bbreak\b',
                r'\boil\b', r'\bbrake\b', r'\btire\b', r'\bengine\b.*\blight\b',
                r'\bwarranty\b', r'\brecall\b', r'\bfix\b'
            ],
            "specs": [
                r'\bspec\b', r'\bhorsepow', r'\btorque\b', r'\bmpg\b',
                r'\b0-60\b', r'\bzero.*(to|2).*sixty\b', r'\btop\s+speed\b',
                r'\bengine\b', r'\btransmission\b', r'\bfeature\b',
                r'\brange\b', r'\bhow\s+(fast|powerful|efficient)\b'
            ]
        }
        
        print("[AutoMind] Multi-Agent Orchestrator initialized")
        print(f"[AutoMind] Available specialists: {list(self.agents.keys())}")
    
    def route_query(self, query: str) -> str:
        """
        Determine which specialist should handle the query.
        
        Uses pattern matching with fallback to specs agent.
        """
        import re
        query_lower = query.lower()
        
        scores = {agent: 0 for agent in self.agents}
        
        for agent_name, patterns in self.routing_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[agent_name] += 1
        
        # Find agent with highest score
        best_agent = max(scores, key=scores.get)
        
        # If no strong match, default to specs
        if scores[best_agent] == 0:
            best_agent = "specs"
        
        return best_agent
    
    def run(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Route query to appropriate specialist and get response.
        """
        # Determine which agent should handle this
        agent_name = self.route_query(query)
        agent = self.agents[agent_name]
        
        if self.config.AGENT_VERBOSE:
            print(f"[AutoMind] Routing to: {agent_name.title()} Agent")
        
        # Run the specialist agent
        response = agent.run(query, chat_history)
        
        return {
            "response": response.content,
            "agent_used": response.agent_name,
            "tools_used": response.tools_used,
            "confidence": response.confidence
        }
    
    def run_all(self, query: str, chat_history: List = None) -> Dict[str, AgentResponse]:
        """
        Run query through all agents and return all responses.
        
        Useful for comprehensive analysis or when user wants multiple perspectives.
        """
        results = {}
        for name, agent in self.agents.items():
            results[name] = agent.run(query, chat_history)
        return results
