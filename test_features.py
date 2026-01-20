"""Test script for AutoMind features."""
import sys
sys.path.insert(0, 'G:\\NLP\\AutoMind')

print("=" * 60)
print("AutoMind Feature Tests")
print("=" * 60)

# Test 1: Multi-Agent Orchestrator
print("\n1. Testing Multi-Agent Orchestrator...")
from agents.multi_agent import MultiAgentOrchestrator
ma = MultiAgentOrchestrator()
print("   Multi-Agent initialized!")
print("   Routing tests:")
print(f"   - 'compare Mustang vs Camaro' -> {ma.route_query('compare Mustang vs Camaro')}")
print(f"   - 'how much is a Tesla' -> {ma.route_query('how much is a Tesla')}")
print(f"   - 'Toyota reliability' -> {ma.route_query('Toyota reliability')}")
print(f"   - 'Corvette horsepower' -> {ma.route_query('Corvette horsepower')}")

# Test 2: RAG Tool
print("\n2. Testing RAG Knowledge Base...")
from tools.rag_tools import car_knowledge_search
result = car_knowledge_search.invoke("Tesla Model S specs")
print(f"   RAG search result (first 300 chars):\n   {result[:300]}...")

# Test 3: Unit Converter
print("\n3. Testing Unit Converter...")
from tools.utility_tools import unit_converter
result = unit_converter.invoke({"value": 100, "conversion_type": "mph_to_kmh"})
print(f"   100 mph to km/h: {result}")

result = unit_converter.invoke({"value": 400, "conversion_type": "hp_to_kw"})
print(f"   400 hp to kW: {result}")

# Test 4: Full Agent
print("\n4. Testing Full Agent Initialization...")
from agents.car_expert import CarExpertAgent
agent = CarExpertAgent()
print(f"   Agent tools: {[t.name for t in agent.tools]}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
