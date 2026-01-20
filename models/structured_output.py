"""
Structured Output Models for AutoMind
=====================================

This module defines Pydantic models for parsing and structuring car data.

NLP/LangChain Concepts:
- Structured Output: Using LLMs to extract data into typed schemas
- Pydantic Models: Type-safe data validation
- Output Parsers: Converting LLM responses to structured data
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class CarSpecifications(BaseModel):
    """
    Structured model for car specifications.
    
    Used by the LLM to extract and return specs in a consistent format.
    """
    make: str = Field(description="Car manufacturer (e.g., Toyota, Ford, BMW)")
    model: str = Field(description="Car model name (e.g., Supra, Mustang, M3)")
    year: Optional[int] = Field(default=None, description="Model year (e.g., 2024)")
    
    # Performance
    horsepower: Optional[int] = Field(default=None, description="Engine horsepower")
    torque: Optional[str] = Field(default=None, description="Torque with units (e.g., '400 lb-ft')")
    zero_to_sixty: Optional[str] = Field(default=None, description="0-60 mph time (e.g., '3.9 seconds')")
    top_speed: Optional[str] = Field(default=None, description="Top speed with units")
    
    # Engine/Motor
    engine: Optional[str] = Field(default=None, description="Engine description (e.g., '3.0L Twin-Turbo I6')")
    transmission: Optional[str] = Field(default=None, description="Transmission type")
    drivetrain: Optional[str] = Field(default=None, description="Drivetrain (FWD, RWD, AWD)")
    
    # Efficiency
    mpg_city: Optional[int] = Field(default=None, description="City MPG")
    mpg_highway: Optional[int] = Field(default=None, description="Highway MPG")
    range_miles: Optional[int] = Field(default=None, description="Electric range in miles")
    
    # Physical
    weight_lbs: Optional[int] = Field(default=None, description="Curb weight in pounds")
    
    # Pricing
    base_price: Optional[int] = Field(default=None, description="Starting MSRP in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "make": "Toyota",
                "model": "GR Supra",
                "year": 2024,
                "horsepower": 382,
                "torque": "368 lb-ft",
                "zero_to_sixty": "3.9 seconds",
                "top_speed": "155 mph",
                "engine": "3.0L Twin-Scroll Turbo I6",
                "transmission": "8-speed automatic",
                "drivetrain": "RWD",
                "mpg_city": 22,
                "mpg_highway": 30,
                "weight_lbs": 3400,
                "base_price": 56250
            }
        }


class CarComparison(BaseModel):
    """
    Structured model for comparing two or more cars.
    """
    cars: List[CarSpecifications] = Field(description="List of cars being compared")
    recommendation: str = Field(description="Which car is recommended and why")
    winner_performance: Optional[str] = Field(default=None, description="Best performing car")
    winner_value: Optional[str] = Field(default=None, description="Best value car")
    winner_efficiency: Optional[str] = Field(default=None, description="Most fuel efficient car")


class MaintenanceInfo(BaseModel):
    """
    Structured model for car maintenance information.
    """
    make: str = Field(description="Car manufacturer")
    model: str = Field(description="Car model")
    
    # Common intervals
    oil_change_miles: Optional[int] = Field(default=None, description="Oil change interval in miles")
    tire_rotation_miles: Optional[int] = Field(default=None, description="Tire rotation interval")
    brake_inspection_miles: Optional[int] = Field(default=None, description="Brake inspection interval")
    major_service_miles: Optional[int] = Field(default=None, description="Major service interval")
    
    # Common issues
    common_problems: List[str] = Field(default=[], description="Known common issues")
    
    # Costs
    annual_maintenance_cost: Optional[int] = Field(default=None, description="Estimated annual maintenance cost")
    
    reliability_rating: Optional[float] = Field(default=None, description="Reliability rating 1-10")


class CarPricing(BaseModel):
    """
    Structured model for car pricing information.
    """
    make: str = Field(description="Car manufacturer")
    model: str = Field(description="Car model")
    year: int = Field(description="Model year")
    
    msrp: Optional[int] = Field(default=None, description="Manufacturer's suggested retail price")
    invoice_price: Optional[int] = Field(default=None, description="Dealer invoice price")
    average_market_price: Optional[int] = Field(default=None, description="Average transaction price")
    
    # Ownership costs
    insurance_annual: Optional[int] = Field(default=None, description="Estimated annual insurance")
    depreciation_5yr: Optional[str] = Field(default=None, description="5-year depreciation percentage")
    total_cost_of_ownership: Optional[int] = Field(default=None, description="5-year total cost")


class CarQuery(BaseModel):
    """
    Structured model for understanding user queries.
    
    Used for intent classification and query parsing.
    """
    query_type: Literal[
        "specifications",
        "comparison",
        "pricing",
        "maintenance",
        "review",
        "recommendation",
        "history",
        "general"
    ] = Field(description="Type of car query")
    
    cars_mentioned: List[str] = Field(default=[], description="Car names mentioned in query")
    
    specific_specs_requested: List[str] = Field(
        default=[],
        description="Specific specs asked about (e.g., 'horsepower', 'mpg', 'price')"
    )
    
    year_range: Optional[str] = Field(default=None, description="Year or year range mentioned")
    
    budget: Optional[int] = Field(default=None, description="Budget mentioned in query")
    
    category_preference: Optional[str] = Field(
        default=None,
        description="Car category (sports car, SUV, sedan, truck, electric)"
    )


class StructuredCarResponse(BaseModel):
    """
    Wrapper for structured car responses.
    
    Combines specs with natural language explanation.
    """
    specs: Optional[CarSpecifications] = Field(default=None, description="Extracted car specs")
    comparison: Optional[CarComparison] = Field(default=None, description="Comparison data if applicable")
    maintenance: Optional[MaintenanceInfo] = Field(default=None, description="Maintenance info if applicable")
    pricing: Optional[CarPricing] = Field(default=None, description="Pricing info if applicable")
    
    natural_response: str = Field(description="Natural language response for the user")
    sources: List[str] = Field(default=[], description="Sources used for information")
    confidence: float = Field(default=0.8, description="Confidence in the response 0-1")
