"""
AutoMind Models Package
=======================

Pydantic models for structured output parsing.
"""

from models.structured_output import (
    CarSpecifications,
    CarComparison,
    MaintenanceInfo,
    CarPricing,
    CarQuery,
    StructuredCarResponse,
)

__all__ = [
    "CarSpecifications",
    "CarComparison",
    "MaintenanceInfo",
    "CarPricing",
    "CarQuery",
    "StructuredCarResponse",
]
