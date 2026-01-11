"""
Rule-Based System for Vehicle Price Prediction
Implements expert knowledge rules for validation and adjustment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class VehiclePricingRules:
    """
    Expert rule-based system for vehicle pricing
    Uses domain knowledge to validate inputs and adjust predictions
    """
    
    def __init__(self):
        self.rules_applied = []
        self.warnings = []
        
    def reset(self):
        """Reset rules tracking"""
        self.rules_applied = []
        self.warnings = []
    
    # ========================================================================
    # INPUT VALIDATION RULES (Pre-processing)
    # ========================================================================
    
    def validate_input(self, vehicle_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate vehicle input using domain knowledge rules
        
        Returns:
        --------
        (is_valid, warnings)
        """
        self.reset()
        warnings = []
        
        year = vehicle_data.get('year')
        odometer = vehicle_data.get('odometer')
        condition = vehicle_data.get('condition')
        make = vehicle_data.get('make', '')
        body = vehicle_data.get('body', '')
        
        # Rule 1: Age-Mileage Consistency
        if year and odometer:
            vehicle_age = 2026 - year
            expected_max_mileage = vehicle_age * 15000  # 15k miles/year is typical
            expected_min_mileage = vehicle_age * 5000   # 5k miles/year is low
            
            if odometer > expected_max_mileage * 1.5:
                warnings.append(f"Rule: High mileage for age - {odometer:,} miles for {vehicle_age} year old vehicle (expected max ~{expected_max_mileage:,})")
                self.rules_applied.append("HIGH_MILEAGE_FOR_AGE")
            
            if odometer < expected_min_mileage and vehicle_age > 2:
                warnings.append(f"Rule: Unusually low mileage - {odometer:,} miles for {vehicle_age} year old vehicle (minimum expected ~{expected_min_mileage:,})")
                self.rules_applied.append("LOW_MILEAGE_FOR_AGE")
        
        # Rule 2: Condition-Age Consistency
        if year and condition:
            vehicle_age = 2026 - year
            expected_max_condition = max(49 - (vehicle_age * 2), 15)
            
            if condition > expected_max_condition:
                warnings.append(f"Rule: High condition rating for age - Condition {condition} unusual for {vehicle_age} year old vehicle")
                self.rules_applied.append("HIGH_CONDITION_FOR_AGE")
        
        # Rule 3: Luxury Brand Premium
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'Tesla', 'Land Rover']
        if make in luxury_brands:
            self.rules_applied.append("LUXURY_BRAND_DETECTED")
        
        # Rule 4: Popular Truck Models Hold Value
        popular_trucks = ['F-150', 'Silverado', 'Ram', 'Tundra', 'Tacoma']
        model = vehicle_data.get('model', '')
        if body == 'Truck' and any(truck in model for truck in popular_trucks):
            self.rules_applied.append("POPULAR_TRUCK_MODEL")
        
        # Rule 5: Very Old Vehicle Check
        if year and year < 2000:
            warnings.append(f"Rule: Classic/vintage vehicle ({year}) - Price may be influenced by collectibility")
            self.rules_applied.append("VINTAGE_VEHICLE")
        
        # Rule 6: Extreme Odometer Check
        if odometer and odometer > 200000:
            warnings.append(f"Rule: High mileage vehicle ({odometer:,} miles) - Expect significant depreciation")
            self.rules_applied.append("HIGH_MILEAGE_VEHICLE")
        
        # Rule 7: Poor Condition Check
        if condition and condition < 20:
            warnings.append(f"Rule: Poor condition ({condition}/49) - May require significant repairs")
            self.rules_applied.append("POOR_CONDITION")
        
        self.warnings = warnings
        return True, warnings
    
    # ========================================================================
    # PREDICTION ADJUSTMENT RULES (Post-processing)
    # ========================================================================
    
    def adjust_prediction(self, base_prediction: float, vehicle_data: Dict) -> Dict:
        """
        Apply expert rules to adjust ML model prediction
        
        Returns:
        --------
        {
            'adjusted_price': float,
            'base_price': float,
            'adjustments': list,
            'confidence': str
        }
        """
        adjusted_price = base_prediction
        adjustments = []
        
        year = vehicle_data.get('year')
        odometer = vehicle_data.get('odometer')
        condition = vehicle_data.get('condition')
        make = vehicle_data.get('make', '')
        model = vehicle_data.get('model', '')
        body = vehicle_data.get('body', '')
        transmission = vehicle_data.get('transmission', '')
        
        # Rule 1: Luxury Depreciation Curve
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche', 'Tesla']
        if make in luxury_brands and year:
            vehicle_age = 2026 - year
            if vehicle_age > 5:
                # Luxury cars depreciate faster after 5 years
                depreciation_factor = 1 - (vehicle_age - 5) * 0.03
                adjustment = adjusted_price * (depreciation_factor - 1)
                adjusted_price *= depreciation_factor
                adjustments.append(f"Luxury depreciation: {adjustment:+,.0f} (Age {vehicle_age} years)")
        
        # Rule 2: Popular Truck Premium
        popular_trucks = ['F-150', 'Silverado', 'Ram 1500', 'Tundra', 'Tacoma']
        if body == 'Truck' and any(truck in model for truck in popular_trucks):
            premium = base_prediction * 0.05  # 5% premium
            adjusted_price += premium
            adjustments.append(f"Popular truck premium: +${premium:,.0f}")
        
        # Rule 3: High Mileage Penalty
        if odometer and odometer > 150000:
            penalty_rate = min((odometer - 150000) / 100000 * 0.1, 0.15)
            penalty = base_prediction * penalty_rate
            adjusted_price -= penalty
            adjustments.append(f"High mileage penalty ({odometer:,} mi): -${penalty:,.0f}")
        
        # Rule 4: Excellent Condition Bonus
        if condition and condition >= 45:
            bonus = base_prediction * 0.08  # 8% bonus
            adjusted_price += bonus
            adjustments.append(f"Excellent condition bonus: +${bonus:,.0f}")
        
        # Rule 5: Poor Condition Penalty
        elif condition and condition <= 15:
            penalty = base_prediction * 0.15  # 15% penalty
            adjusted_price -= penalty
            adjustments.append(f"Poor condition penalty: -${penalty:,.0f}")
        
        # Rule 6: Manual Transmission Adjustment
        if transmission == 'manual':
            # Manual transmission less desirable in most cases (except sports cars)
            sports_makes = ['Porsche', 'BMW', 'Mazda', 'Subaru']
            if make in sports_makes:
                bonus = base_prediction * 0.03
                adjusted_price += bonus
                adjustments.append(f"Sports car manual bonus: +${bonus:,.0f}")
            else:
                penalty = base_prediction * 0.05
                adjusted_price -= penalty
                adjustments.append(f"Manual transmission penalty: -${penalty:,.0f}")
        
        # Rule 7: Hybrid/Electric Premium (Tesla, Prius, etc.)
        electric_models = ['Model 3', 'Model S', 'Model X', 'Model Y', 'Prius', 'Leaf', 'Bolt']
        if any(e_model in model for e_model in electric_models):
            premium = base_prediction * 0.10  # 10% premium for electric/hybrid
            adjusted_price += premium
            adjustments.append(f"Electric/Hybrid premium: +${premium:,.0f}")
        
        # Rule 8: SUV Market Demand (2020+)
        if body == 'SUV' and year and year >= 2020:
            premium = base_prediction * 0.03  # 3% premium for recent SUVs
            adjusted_price += premium
            adjustments.append(f"Recent SUV demand premium: +${premium:,.0f}")
        
        # Rule 9: Convertible Seasonal Adjustment
        if body == 'Convertible':
            # Convertibles generally worth less (limited market)
            penalty = base_prediction * 0.05
            adjusted_price -= penalty
            adjustments.append(f"Convertible market penalty: -${penalty:,.0f}")
        
        # Rule 10: Very Low Mileage Premium
        if odometer and year:
            vehicle_age = 2026 - year
            expected_mileage = vehicle_age * 12000
            if odometer < expected_mileage * 0.5 and vehicle_age > 2:
                premium = base_prediction * 0.07  # 7% premium
                adjusted_price += premium
                adjustments.append(f"Low mileage premium ({odometer:,} mi): +${premium:,.0f}")
        
        # Ensure price doesn't go below minimum
        adjusted_price = max(adjusted_price, 500)
        
        # Calculate confidence based on rules applied
        confidence = self._calculate_confidence(vehicle_data, adjustments)
        
        return {
            'adjusted_price': round(adjusted_price, 2),
            'base_price': round(base_prediction, 2),
            'adjustments': adjustments,
            'total_adjustment': round(adjusted_price - base_prediction, 2),
            'confidence': confidence,
            'rules_applied': len(adjustments)
        }
    
    def _calculate_confidence(self, vehicle_data: Dict, adjustments: List) -> str:
        """Calculate confidence level based on data quality and rules"""
        confidence_score = 100
        
        # Reduce confidence for edge cases
        if vehicle_data.get('year', 2020) < 2000:
            confidence_score -= 15
        
        if vehicle_data.get('odometer', 0) > 200000:
            confidence_score -= 10
        
        if vehicle_data.get('condition', 30) < 15:
            confidence_score -= 15
        
        if len(adjustments) > 5:
            confidence_score -= 10
        
        if confidence_score >= 85:
            return "High"
        elif confidence_score >= 70:
            return "Medium"
        else:
            return "Low"
    
    # ========================================================================
    # EXPERT KNOWLEDGE RULES
    # ========================================================================
    
    def get_market_insights(self, vehicle_data: Dict) -> List[str]:
        """
        Provide expert market insights based on rules
        """
        insights = []
        
        make = vehicle_data.get('make', '')
        model = vehicle_data.get('model', '')
        year = vehicle_data.get('year')
        body = vehicle_data.get('body', '')
        
        # Insight 1: Brand Reliability
        reliable_brands = ['Toyota', 'Honda', 'Lexus', 'Mazda', 'Subaru']
        if make in reliable_brands:
            insights.append(f"{make} is known for reliability - typically holds resale value well")
        
        # Insight 2: Popular Models
        popular_models = ['Camry', 'Accord', 'Civic', 'F-150', 'Silverado', 'CR-V', 'RAV4']
        if any(pm in model for pm in popular_models):
            insights.append(f"{model} is a popular model - high market demand")
        
        # Insight 3: Market Trends
        if body == 'Sedan' and year and year > 2015:
            insights.append("Sedan market has declined in recent years - SUVs more popular")
        
        if body == 'SUV' and year and year > 2018:
            insights.append("SUV market is strong - high demand in current market")
        
        # Insight 4: Depreciation Patterns
        if year:
            vehicle_age = 2026 - year
            if vehicle_age <= 3:
                insights.append("Vehicle in early depreciation phase - loses value quickly in first 3 years")
            elif vehicle_age <= 7:
                insights.append("Vehicle in moderate depreciation phase - value stabilizing")
            else:
                insights.append("Older vehicle - depreciation has slowed significantly")
        
        return insights


def apply_rules(vehicle_data: Dict, ml_prediction: float) -> Dict:
    """
    Convenience function to apply all rules
    
    Parameters:
    -----------
    vehicle_data : dict
        Vehicle information
    ml_prediction : float
        Base prediction from ML model
    
    Returns:
    --------
    dict with validation, adjustment, and insights
    """
    rules_engine = VehiclePricingRules()
    
    # Validate input
    is_valid, warnings = rules_engine.validate_input(vehicle_data)
    
    # Adjust prediction
    adjustment_result = rules_engine.adjust_prediction(ml_prediction, vehicle_data)
    
    # Get insights
    insights = rules_engine.get_market_insights(vehicle_data)
    
    return {
        'is_valid': is_valid,
        'warnings': warnings,
        'base_prediction': adjustment_result['base_price'],
        'adjusted_prediction': adjustment_result['adjusted_price'],
        'adjustments': adjustment_result['adjustments'],
        'total_adjustment': adjustment_result['total_adjustment'],
        'confidence': adjustment_result['confidence'],
        'rules_applied': adjustment_result['rules_applied'],
        'market_insights': insights
    }


# Example usage
if __name__ == "__main__":
    # Test the rule-based system
    test_vehicle = {
        'year': 2015,
        'make': 'BMW',
        'model': '3 Series',
        'body': 'Sedan',
        'transmission': 'automatic',
        'condition': 35,
        'odometer': 85000
    }
    
    # Simulate ML prediction
    ml_price = 18500
    
    # Apply rules
    result = apply_rules(test_vehicle, ml_price)
    
    print("Rule-Based System Results:")
    print("=" * 60)
    print(f"Base ML Prediction: ${result['base_prediction']:,.2f}")
    print(f"Adjusted Price: ${result['adjusted_prediction']:,.2f}")
    print(f"Total Adjustment: ${result['total_adjustment']:+,.2f}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nAdjustments Applied ({result['rules_applied']}):")
    for adj in result['adjustments']:
        print(f"  - {adj}")
    print(f"\nWarnings ({len(result['warnings'])}):")
    for warn in result['warnings']:
        print(f"  - {warn}")
    print(f"\nMarket Insights ({len(result['market_insights'])}):")
    for insight in result['market_insights']:
        print(f"  - {insight}")
