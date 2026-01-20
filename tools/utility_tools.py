"""
Utility Tools for AutoMind
==========================

This module contains calculator and unit conversion tools for car-related
calculations.

NLP/LangChain Concepts:
- Structured Tools: Tools with typed inputs and outputs
- Unit Conversion: Common automotive unit conversions
- Calculations: Performance metrics, fuel economy, etc.
"""

from langchain_core.tools import tool
from typing import Literal


@tool
def unit_converter(
    value: float,
    conversion_type: str
) -> str:
    """
    Convert between common automotive units.
    
    Use this tool when users need to convert:
    - Speed: mph to km/h or km/h to mph
    - Distance: miles to km or km to miles  
    - Power: hp to kW or kW to hp
    - Torque: lb-ft to Nm or Nm to lb-ft
    - Fuel economy: mpg to L/100km or L/100km to mpg
    - Weight: lbs to kg or kg to lbs
    - Temperature: F to C or C to F
    
    Args:
        value: The numeric value to convert
        conversion_type: Type of conversion. Options:
            - 'mph_to_kmh': Miles per hour to kilometers per hour
            - 'kmh_to_mph': Kilometers per hour to miles per hour
            - 'miles_to_km': Miles to kilometers
            - 'km_to_miles': Kilometers to miles
            - 'hp_to_kw': Horsepower to kilowatts
            - 'kw_to_hp': Kilowatts to horsepower
            - 'lbft_to_nm': Pound-feet to Newton-meters
            - 'nm_to_lbft': Newton-meters to pound-feet
            - 'mpg_to_l100km': MPG to liters per 100km
            - 'l100km_to_mpg': Liters per 100km to MPG
            - 'lbs_to_kg': Pounds to kilograms
            - 'kg_to_lbs': Kilograms to pounds
            - 'f_to_c': Fahrenheit to Celsius
            - 'c_to_f': Celsius to Fahrenheit
    
    Returns:
        Converted value with units
    """
    conversions = {
        'mph_to_kmh': (1.60934, 'km/h'),
        'kmh_to_mph': (0.621371, 'mph'),
        'miles_to_km': (1.60934, 'km'),
        'km_to_miles': (0.621371, 'miles'),
        'hp_to_kw': (0.7457, 'kW'),
        'kw_to_hp': (1.341, 'hp'),
        'lbft_to_nm': (1.3558, 'Nm'),
        'nm_to_lbft': (0.7376, 'lb-ft'),
        'lbs_to_kg': (0.453592, 'kg'),
        'kg_to_lbs': (2.20462, 'lbs'),
    }
    
    conversion_type = conversion_type.lower().strip()
    
    # Special cases for temperature and fuel economy
    if conversion_type == 'f_to_c':
        result = (value - 32) * 5/9
        return f"{value}°F = **{result:.1f}°C**"
    elif conversion_type == 'c_to_f':
        result = (value * 9/5) + 32
        return f"{value}°C = **{result:.1f}°F**"
    elif conversion_type == 'mpg_to_l100km':
        result = 235.215 / value if value > 0 else 0
        return f"{value} MPG = **{result:.1f} L/100km**"
    elif conversion_type == 'l100km_to_mpg':
        result = 235.215 / value if value > 0 else 0
        return f"{value} L/100km = **{result:.1f} MPG**"
    elif conversion_type in conversions:
        factor, unit = conversions[conversion_type]
        result = value * factor
        return f"{value} {conversion_type.split('_')[0]} = **{result:.2f} {unit}**"
    else:
        return f"Unknown conversion type: {conversion_type}. Available: {list(conversions.keys()) + ['f_to_c', 'c_to_f', 'mpg_to_l100km', 'l100km_to_mpg']}"


@tool
def car_calculator(
    calculation_type: str,
    **kwargs
) -> str:
    """
    Perform car-related calculations.
    
    Use this tool for:
    - Estimating 0-60 time from power-to-weight ratio
    - Calculating power-to-weight ratio
    - Fuel cost calculations
    - Lease vs buy comparisons
    - Depreciation estimates
    
    Args:
        calculation_type: Type of calculation:
            - 'power_to_weight': Calculate power-to-weight ratio
              Requires: horsepower (float), weight_lbs (float)
            - 'fuel_cost': Estimate fuel cost for a trip
              Requires: distance_miles (float), mpg (float), fuel_price (float)
            - 'monthly_payment': Estimate car loan payment
              Requires: price (float), down_payment (float), interest_rate (float), months (int)
    
    Returns:
        Calculation result with explanation
    """
    try:
        calc_type = calculation_type.lower().strip()
        
        if calc_type == 'power_to_weight':
            hp = kwargs.get('horsepower', 0)
            weight = kwargs.get('weight_lbs', 0)
            
            if hp <= 0 or weight <= 0:
                return "Please provide valid horsepower and weight_lbs values"
            
            # Calculate lb/hp ratio
            ratio = weight / hp
            
            # Estimate 0-60 time (rough approximation)
            # Based on empirical data: 0-60 ≈ ratio/10 + 2 for modern cars
            est_060 = (ratio / 10) + 2
            
            return f"""**Power-to-Weight Analysis:**
            
🏋️ **Weight:** {weight:,.0f} lbs
⚡ **Power:** {hp:,.0f} hp
📊 **Ratio:** {ratio:.1f} lbs/hp

**Performance Estimate:**
- Under 8 lbs/hp: Sports car territory 🏎️
- 8-12 lbs/hp: Quick and fun 💨
- 12-18 lbs/hp: Average performance
- Over 18 lbs/hp: Economy/comfort focus

Your ratio of {ratio:.1f} lbs/hp suggests ~{est_060:.1f}s 0-60 time (estimate)"""
        
        elif calc_type == 'fuel_cost':
            distance = kwargs.get('distance_miles', 0)
            mpg = kwargs.get('mpg', 0)
            fuel_price = kwargs.get('fuel_price', 3.50)  # Default $3.50/gallon
            
            if distance <= 0 or mpg <= 0:
                return "Please provide valid distance_miles and mpg values"
            
            gallons = distance / mpg
            cost = gallons * fuel_price
            
            return f"""**Fuel Cost Estimate:**
            
📏 **Distance:** {distance:,.0f} miles
⛽ **Fuel Economy:** {mpg:.1f} MPG
💰 **Fuel Price:** ${fuel_price:.2f}/gallon

**Results:**
- Gallons needed: {gallons:.1f} gallons
- **Total cost: ${cost:.2f}**
- Cost per mile: ${cost/distance:.3f}"""
        
        elif calc_type == 'monthly_payment':
            price = kwargs.get('price', 0)
            down = kwargs.get('down_payment', 0)
            rate = kwargs.get('interest_rate', 6.0) / 100 / 12  # Monthly rate
            months = kwargs.get('months', 60)
            
            if price <= 0:
                return "Please provide a valid price"
            
            principal = price - down
            
            if rate > 0:
                payment = principal * (rate * (1 + rate)**months) / ((1 + rate)**months - 1)
            else:
                payment = principal / months
            
            total_paid = payment * months
            total_interest = total_paid - principal
            
            return f"""**Car Loan Calculator:**
            
🚗 **Vehicle Price:** ${price:,.0f}
💵 **Down Payment:** ${down:,.0f}
📊 **Loan Amount:** ${principal:,.0f}
📈 **Interest Rate:** {kwargs.get('interest_rate', 6.0):.1f}% APR
📅 **Loan Term:** {months} months

**Results:**
- **Monthly Payment: ${payment:,.2f}**
- Total Amount Paid: ${total_paid:,.2f}
- Total Interest: ${total_interest:,.2f}"""
        
        else:
            return f"Unknown calculation type: {calc_type}. Available: power_to_weight, fuel_cost, monthly_payment"
    
    except Exception as e:
        return f"Calculation error: {str(e)}"
