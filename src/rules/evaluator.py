"""
Hybrid Decision Engine
Combines ML-based intent classification with regex entity extraction
for robust geometry command understanding.
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class GeometryResult:
    """Result of a geometry calculation."""
    value: float
    property_name: str
    formula: str
    explanation_steps: List[str]
    unit: str = "units"


@dataclass
class DecisionResult:
    """Complete result from the decision engine."""
    success: bool
    intent: str
    shape: Optional[str]
    property_requested: Optional[str]
    parameters: Dict[str, float]
    calculations: List[GeometryResult] = field(default_factory=list)
    confidence: float = 0.0
    ml_prediction: Optional[Any] = None
    regex_extraction: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DecisionStrategy(Enum):
    """Strategy for combining ML and regex results."""
    ML_PRIORITY = "ml_priority"  # Trust ML, use regex for extraction
    REGEX_PRIORITY = "regex_priority"  # Trust regex, use ML for validation
    ENSEMBLE = "ensemble"  # Combine both with voting
    CASCADING = "cascading"  # Try ML first, fallback to regex


class GeometryCalculator:
    """Performs geometry calculations with step-by-step explanations."""
    
    @staticmethod
    def calculate_circle(radius: float) -> List[GeometryResult]:
        """Calculate circle properties."""
        results = []
        
        # Area
        area = math.pi * radius ** 2
        results.append(GeometryResult(
            value=area,
            property_name="area",
            formula="A = π × r²",
            explanation_steps=[
                f"Using the formula: Area = π × radius²",
                f"Radius = {radius}",
                f"Radius² = {radius}² = {radius ** 2}",
                f"Area = π × {radius ** 2} = {area:.4f}",
                f"The area is {area:.4f} square units"
            ],
            unit="square units"
        ))
        
        # Circumference
        circumference = 2 * math.pi * radius
        results.append(GeometryResult(
            value=circumference,
            property_name="circumference",
            formula="C = 2πr",
            explanation_steps=[
                f"Using the formula: Circumference = 2 × π × radius",
                f"Radius = {radius}",
                f"Circumference = 2 × π × {radius} = {circumference:.4f}",
                f"The circumference is {circumference:.4f} units"
            ],
            unit="units"
        ))
        
        # Diameter
        diameter = 2 * radius
        results.append(GeometryResult(
            value=diameter,
            property_name="diameter",
            formula="d = 2r",
            explanation_steps=[
                f"Diameter = 2 × radius",
                f"Diameter = 2 × {radius} = {diameter}",
                f"The diameter is {diameter} units"
            ],
            unit="units"
        ))
        
        return results
    
    @staticmethod
    def calculate_rectangle(length: float, width: float) -> List[GeometryResult]:
        """Calculate rectangle properties."""
        results = []
        
        # Area
        area = length * width
        results.append(GeometryResult(
            value=area,
            property_name="area",
            formula="A = l × w",
            explanation_steps=[
                f"Using the formula: Area = length × width",
                f"Length = {length}, Width = {width}",
                f"Area = {length} × {width} = {area}",
                f"The area is {area} square units"
            ],
            unit="square units"
        ))
        
        # Perimeter
        perimeter = 2 * (length + width)
        results.append(GeometryResult(
            value=perimeter,
            property_name="perimeter",
            formula="P = 2(l + w)",
            explanation_steps=[
                f"Using the formula: Perimeter = 2 × (length + width)",
                f"Length = {length}, Width = {width}",
                f"Sum = {length} + {width} = {length + width}",
                f"Perimeter = 2 × {length + width} = {perimeter}",
                f"The perimeter is {perimeter} units"
            ],
            unit="units"
        ))
        
        # Diagonal
        diagonal = math.sqrt(length ** 2 + width ** 2)
        results.append(GeometryResult(
            value=diagonal,
            property_name="diagonal",
            formula="d = √(l² + w²)",
            explanation_steps=[
                f"Using the Pythagorean theorem: Diagonal = √(length² + width²)",
                f"Length² = {length ** 2}, Width² = {width ** 2}",
                f"Sum = {length ** 2 + width ** 2}",
                f"Diagonal = √{length ** 2 + width ** 2} = {diagonal:.4f}",
                f"The diagonal is {diagonal:.4f} units"
            ],
            unit="units"
        ))
        
        return results
    
    @staticmethod
    def calculate_triangle(side_a: float, side_b: float, side_c: float) -> List[GeometryResult]:
        """Calculate triangle properties using Heron's formula."""
        results = []
        
        # Perimeter
        perimeter = side_a + side_b + side_c
        results.append(GeometryResult(
            value=perimeter,
            property_name="perimeter",
            formula="P = a + b + c",
            explanation_steps=[
                f"Perimeter = side_a + side_b + side_c",
                f"Perimeter = {side_a} + {side_b} + {side_c} = {perimeter}",
                f"The perimeter is {perimeter} units"
            ],
            unit="units"
        ))
        
        # Area using Heron's formula
        s = perimeter / 2
        area_squared = s * (s - side_a) * (s - side_b) * (s - side_c)
        if area_squared > 0:
            area = math.sqrt(area_squared)
            results.append(GeometryResult(
                value=area,
                property_name="area",
                formula="A = √[s(s-a)(s-b)(s-c)]",
                explanation_steps=[
                    f"Using Heron's formula",
                    f"Semi-perimeter s = {perimeter}/2 = {s}",
                    f"s - a = {s - side_a}, s - b = {s - side_b}, s - c = {s - side_c}",
                    f"Product = {area_squared:.4f}",
                    f"Area = √{area_squared:.4f} = {area:.4f}",
                    f"The area is {area:.4f} square units"
                ],
                unit="square units"
            ))
        
        return results
    
    @staticmethod
    def calculate_sphere(radius: float) -> List[GeometryResult]:
        """Calculate sphere properties."""
        results = []
        
        # Volume
        volume = (4/3) * math.pi * radius ** 3
        results.append(GeometryResult(
            value=volume,
            property_name="volume",
            formula="V = (4/3)πr³",
            explanation_steps=[
                f"Using the formula: Volume = (4/3) × π × radius³",
                f"Radius = {radius}",
                f"Radius³ = {radius ** 3}",
                f"Volume = (4/3) × π × {radius ** 3} = {volume:.4f}",
                f"The volume is {volume:.4f} cubic units"
            ],
            unit="cubic units"
        ))
        
        # Surface area
        surface_area = 4 * math.pi * radius ** 2
        results.append(GeometryResult(
            value=surface_area,
            property_name="surface_area",
            formula="A = 4πr²",
            explanation_steps=[
                f"Using the formula: Surface Area = 4 × π × radius²",
                f"Radius² = {radius ** 2}",
                f"Surface Area = 4 × π × {radius ** 2} = {surface_area:.4f}",
                f"The surface area is {surface_area:.4f} square units"
            ],
            unit="square units"
        ))
        
        return results
    
    @staticmethod
    def calculate_cylinder(radius: float, height: float) -> List[GeometryResult]:
        """Calculate cylinder properties."""
        results = []
        
        # Volume
        volume = math.pi * radius ** 2 * height
        results.append(GeometryResult(
            value=volume,
            property_name="volume",
            formula="V = πr²h",
            explanation_steps=[
                f"Using the formula: Volume = π × radius² × height",
                f"Radius = {radius}, Height = {height}",
                f"Base area = π × {radius}² = {math.pi * radius ** 2:.4f}",
                f"Volume = {math.pi * radius ** 2:.4f} × {height} = {volume:.4f}",
                f"The volume is {volume:.4f} cubic units"
            ],
            unit="cubic units"
        ))
        
        # Lateral surface area
        lateral = 2 * math.pi * radius * height
        results.append(GeometryResult(
            value=lateral,
            property_name="lateral_surface_area",
            formula="A_lateral = 2πrh",
            explanation_steps=[
                f"Lateral Surface Area = 2 × π × radius × height",
                f"= 2 × π × {radius} × {height} = {lateral:.4f}",
                f"The lateral surface area is {lateral:.4f} square units"
            ],
            unit="square units"
        ))
        
        # Total surface area
        total = 2 * math.pi * radius * (radius + height)
        results.append(GeometryResult(
            value=total,
            property_name="total_surface_area",
            formula="A_total = 2πr(r + h)",
            explanation_steps=[
                f"Total Surface Area = 2 × π × radius × (radius + height)",
                f"= 2 × π × {radius} × ({radius} + {height})",
                f"= 2 × π × {radius} × {radius + height} = {total:.4f}",
                f"The total surface area is {total:.4f} square units"
            ],
            unit="square units"
        ))
        
        return results
    
    @staticmethod
    def calculate_cone(radius: float, height: float) -> List[GeometryResult]:
        """Calculate cone properties."""
        results = []
        
        # Slant height
        slant = math.sqrt(radius ** 2 + height ** 2)
        results.append(GeometryResult(
            value=slant,
            property_name="slant_height",
            formula="l = √(r² + h²)",
            explanation_steps=[
                f"Slant height = √(radius² + height²)",
                f"= √({radius}² + {height}²) = √{radius ** 2 + height ** 2}",
                f"= {slant:.4f}",
                f"The slant height is {slant:.4f} units"
            ],
            unit="units"
        ))
        
        # Volume
        volume = (1/3) * math.pi * radius ** 2 * height
        results.append(GeometryResult(
            value=volume,
            property_name="volume",
            formula="V = (1/3)πr²h",
            explanation_steps=[
                f"Volume = (1/3) × π × radius² × height",
                f"= (1/3) × π × {radius}² × {height}",
                f"= {volume:.4f}",
                f"The volume is {volume:.4f} cubic units"
            ],
            unit="cubic units"
        ))
        
        # Surface area
        surface = math.pi * radius * (radius + slant)
        results.append(GeometryResult(
            value=surface,
            property_name="surface_area",
            formula="A = πr(r + l)",
            explanation_steps=[
                f"Total Surface Area = π × radius × (radius + slant)",
                f"= π × {radius} × ({radius} + {slant:.4f})",
                f"= {surface:.4f}",
                f"The surface area is {surface:.4f} square units"
            ],
            unit="square units"
        ))
        
        return results
    
    @staticmethod
    def calculate_rectangular_prism(length: float, width: float, height: float) -> List[GeometryResult]:
        """Calculate rectangular prism (box/cuboid) properties."""
        results = []
        
        # Volume
        volume = length * width * height
        results.append(GeometryResult(
            value=volume,
            property_name="volume",
            formula="V = l × w × h",
            explanation_steps=[
                f"Using the formula: Volume = length × width × height",
                f"Length = {length}, Width = {width}, Height = {height}",
                f"Volume = {length} × {width} × {height} = {volume}",
                f"The volume is {volume} cubic units"
            ],
            unit="cubic units"
        ))
        
        # Surface area
        surface = 2 * (length * width + width * height + height * length)
        results.append(GeometryResult(
            value=surface,
            property_name="surface_area",
            formula="A = 2(lw + wh + hl)",
            explanation_steps=[
                f"Surface Area = 2 × (length×width + width×height + height×length)",
                f"= 2 × ({length}×{width} + {width}×{height} + {height}×{length})",
                f"= 2 × ({length*width} + {width*height} + {height*length})",
                f"= 2 × {length*width + width*height + height*length} = {surface}",
                f"The surface area is {surface} square units"
            ],
            unit="square units"
        ))
        
        # Space diagonal
        diagonal = math.sqrt(length ** 2 + width ** 2 + height ** 2)
        results.append(GeometryResult(
            value=diagonal,
            property_name="space_diagonal",
            formula="d = √(l² + w² + h²)",
            explanation_steps=[
                f"Space diagonal = √(length² + width² + height²)",
                f"= √({length}² + {width}² + {height}²)",
                f"= √({length**2} + {width**2} + {height**2})",
                f"= √{length**2 + width**2 + height**2} = {diagonal:.4f}",
                f"The space diagonal is {diagonal:.4f} units"
            ],
            unit="units"
        ))
        
        return results
    
    @staticmethod
    def calculate_cube(side: float) -> List[GeometryResult]:
        """Calculate cube properties."""
        results = []
        
        # Volume
        volume = side ** 3
        results.append(GeometryResult(
            value=volume,
            property_name="volume",
            formula="V = s³",
            explanation_steps=[
                f"Volume = side³ = {side}³ = {volume}",
                f"The volume is {volume} cubic units"
            ],
            unit="cubic units"
        ))
        
        # Surface area
        surface = 6 * side ** 2
        results.append(GeometryResult(
            value=surface,
            property_name="surface_area",
            formula="A = 6s²",
            explanation_steps=[
                f"Surface Area = 6 × side²",
                f"= 6 × {side}² = 6 × {side ** 2} = {surface}",
                f"The surface area is {surface} square units"
            ],
            unit="square units"
        ))
        
        # Diagonal
        diagonal = side * math.sqrt(3)
        results.append(GeometryResult(
            value=diagonal,
            property_name="space_diagonal",
            formula="d = s√3",
            explanation_steps=[
                f"Space diagonal = side × √3",
                f"= {side} × √3 = {diagonal:.4f}",
                f"The space diagonal is {diagonal:.4f} units"
            ],
            unit="units"
        ))
        
        return results
    
    @staticmethod
    def calculate_regular_polygon(n: int, side: float) -> List[GeometryResult]:
        """Calculate regular polygon properties (pentagon, hexagon, etc.)."""
        results = []
        
        polygon_names = {
            5: "pentagon",
            6: "hexagon",
            7: "heptagon",
            8: "octagon",
            9: "nonagon",
            10: "decagon"
        }
        name = polygon_names.get(n, f"{n}-sided polygon")
        
        # Perimeter
        perimeter = n * side
        results.append(GeometryResult(
            value=perimeter,
            property_name="perimeter",
            formula=f"P = n × s = {n} × s",
            explanation_steps=[
                f"A regular {name} has {n} equal sides",
                f"Perimeter = number of sides × side length",
                f"Perimeter = {n} × {side} = {perimeter}",
                f"The perimeter is {perimeter} units"
            ],
            unit="units"
        ))
        
        # Area of regular polygon: A = (n × s² × cot(π/n)) / 4
        # Alternatively: A = (n × s²) / (4 × tan(π/n))
        area = (n * side ** 2) / (4 * math.tan(math.pi / n))
        results.append(GeometryResult(
            value=area,
            property_name="area",
            formula="A = (n × s²) / (4 × tan(π/n))",
            explanation_steps=[
                f"Using the formula for area of a regular {name}:",
                f"Area = (n × s²) / (4 × tan(π/n))",
                f"Where n = {n} (number of sides), s = {side} (side length)",
                f"tan(π/{n}) = tan({180/n:.2f}°) = {math.tan(math.pi / n):.6f}",
                f"Area = ({n} × {side}²) / (4 × {math.tan(math.pi / n):.6f})",
                f"Area = ({n} × {side ** 2}) / {4 * math.tan(math.pi / n):.6f}",
                f"Area = {area:.4f}",
                f"The area is {area:.4f} square units"
            ],
            unit="square units"
        ))
        
        # Interior angle
        interior_angle = (n - 2) * 180 / n
        results.append(GeometryResult(
            value=interior_angle,
            property_name="interior_angle",
            formula="Interior angle = ((n-2) × 180°) / n",
            explanation_steps=[
                f"Interior angle of a regular {name}:",
                f"Interior angle = ((n-2) × 180°) / n",
                f"= (({n}-2) × 180°) / {n}",
                f"= ({n-2} × 180°) / {n}",
                f"= {(n-2) * 180}° / {n}",
                f"= {interior_angle:.2f}°",
                f"Each interior angle is {interior_angle:.2f} degrees"
            ],
            unit="degrees"
        ))
        
        # Apothem (distance from center to middle of a side)
        apothem = side / (2 * math.tan(math.pi / n))
        results.append(GeometryResult(
            value=apothem,
            property_name="apothem",
            formula="a = s / (2 × tan(π/n))",
            explanation_steps=[
                f"Apothem = s / (2 × tan(π/n))",
                f"= {side} / (2 × tan(π/{n}))",
                f"= {side} / (2 × {math.tan(math.pi / n):.6f})",
                f"= {apothem:.4f}",
                f"The apothem is {apothem:.4f} units"
            ],
            unit="units"
        ))
        
        return results
    
    @staticmethod
    def calculate_square(side: float) -> List[GeometryResult]:
        """Calculate square properties."""
        results = []
        
        area = side ** 2
        results.append(GeometryResult(
            value=area,
            property_name="area",
            formula="A = s²",
            explanation_steps=[
                f"Area = side² = {side}² = {area}",
                f"The area is {area} square units"
            ],
            unit="square units"
        ))
        
        perimeter = 4 * side
        results.append(GeometryResult(
            value=perimeter,
            property_name="perimeter",
            formula="P = 4s",
            explanation_steps=[
                f"Perimeter = 4 × side = 4 × {side} = {perimeter}",
                f"The perimeter is {perimeter} units"
            ],
            unit="units"
        ))
        
        diagonal = side * math.sqrt(2)
        results.append(GeometryResult(
            value=diagonal,
            property_name="diagonal",
            formula="d = s√2",
            explanation_steps=[
                f"Diagonal = side × √2 = {side} × √2 = {diagonal:.4f}",
                f"The diagonal is {diagonal:.4f} units"
            ],
            unit="units"
        ))
        
        return results


class HybridDecisionEngine:
    """
    Combines ML intent classification with regex entity extraction
    for robust natural language understanding of geometry commands.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        strategy: DecisionStrategy = DecisionStrategy.ENSEMBLE
    ):
        """
        Initialize the hybrid decision engine.
        
        Args:
            model_path: Path to trained ML models
            strategy: Decision strategy for combining ML and regex
        """
        self.strategy = strategy
        self.ml_classifier = None
        self.entity_extractor = None
        self.calculator = GeometryCalculator()
        
        self._init_components(model_path)
    
    def _init_components(self, model_path: Optional[str]):
        """Initialize ML and regex components."""
        # Initialize regex extractor
        try:
            from src.preprocessing.entity_extractor import RegexEntityExtractor
            self.entity_extractor = RegexEntityExtractor()
            print("✓ Regex entity extractor initialized")
        except ImportError as e:
            print(f"⚠ Could not load entity extractor: {e}")
        
        # Initialize ML classifier if model exists
        if model_path and os.path.exists(model_path):
            try:
                from src.ml.model import IntentClassifier
                self.ml_classifier = IntentClassifier.load(model_path)
                print("✓ ML intent classifier loaded")
            except Exception as e:
                print(f"⚠ Could not load ML model: {e}")
    
    def process(self, text: str) -> DecisionResult:
        """
        Process user input and return a decision result.
        
        Args:
            text: Natural language input
            
        Returns:
            DecisionResult with intent, parameters, and calculations
        """
        result = DecisionResult(
            success=False,
            intent="unknown",
            shape=None,
            property_requested=None,
            parameters={}
        )
        
        # Get ML prediction
        ml_prediction = None
        if self.ml_classifier:
            try:
                ml_prediction = self.ml_classifier.predict(text)
                result.ml_prediction = ml_prediction
            except Exception as e:
                result.warnings.append(f"ML prediction failed: {e}")
        
        # Get regex extraction
        regex_result = None
        if self.entity_extractor:
            try:
                regex_result = self.entity_extractor.extract(text)
                result.regex_extraction = regex_result
            except Exception as e:
                result.warnings.append(f"Regex extraction failed: {e}")
        
        # Combine results based on strategy
        if self.strategy == DecisionStrategy.ENSEMBLE:
            result = self._ensemble_decision(result, ml_prediction, regex_result)
        elif self.strategy == DecisionStrategy.ML_PRIORITY:
            result = self._ml_priority_decision(result, ml_prediction, regex_result)
        elif self.strategy == DecisionStrategy.REGEX_PRIORITY:
            result = self._regex_priority_decision(result, ml_prediction, regex_result)
        else:
            result = self._cascading_decision(result, ml_prediction, regex_result)
        
        # Perform calculations if we have shape and parameters
        if result.shape and result.parameters:
            result = self._calculate(result)
        
        return result
    
    def _ensemble_decision(
        self, 
        result: DecisionResult,
        ml_pred: Any,
        regex_result: Any
    ) -> DecisionResult:
        """Combine ML and regex results with voting/averaging."""
        
        # Determine shape (prefer regex if specific, ML for general)
        if regex_result and regex_result.shape:
            result.shape = regex_result.shape
            result.confidence = 0.9
        elif ml_pred and ml_pred.shape:
            result.shape = ml_pred.shape
            result.confidence = ml_pred.confidence * 0.8
        
        # Determine intent from ML
        if ml_pred:
            result.intent = ml_pred.intent
            result.property_requested = ml_pred.property_type
            result.confidence = (result.confidence + ml_pred.confidence) / 2
        
        # Get parameters from regex
        if regex_result and regex_result.parameters:
            result.parameters = {
                name: val.value 
                for name, val in regex_result.parameters.items()
            }
        
        result.success = bool(result.shape and result.parameters)
        return result
    
    def _ml_priority_decision(
        self,
        result: DecisionResult,
        ml_pred: Any,
        regex_result: Any
    ) -> DecisionResult:
        """Prioritize ML predictions, use regex for extraction."""
        if ml_pred:
            result.intent = ml_pred.intent
            result.shape = ml_pred.shape
            result.property_requested = ml_pred.property_type
            result.confidence = ml_pred.confidence
        
        if regex_result:
            # Override shape from regex if ML confidence is low
            if regex_result.shape and (not ml_pred or ml_pred.confidence < 0.7):
                result.shape = regex_result.shape
            
            result.parameters = {
                name: val.value 
                for name, val in regex_result.parameters.items()
            }
        
        result.success = bool(result.shape and result.parameters)
        return result
    
    def _regex_priority_decision(
        self,
        result: DecisionResult,
        ml_pred: Any,
        regex_result: Any
    ) -> DecisionResult:
        """Prioritize regex extraction, use ML for validation."""
        if regex_result:
            result.shape = regex_result.shape
            result.parameters = {
                name: val.value 
                for name, val in regex_result.parameters.items()
            }
            result.confidence = 0.85
        
        if ml_pred:
            result.intent = ml_pred.intent
            result.property_requested = ml_pred.property_type
            
            # Validate with ML
            if ml_pred.shape and ml_pred.shape != result.shape:
                result.warnings.append(
                    f"ML suggested '{ml_pred.shape}' but regex found '{result.shape}'"
                )
        
        result.success = bool(result.shape and result.parameters)
        return result
    
    def _cascading_decision(
        self,
        result: DecisionResult,
        ml_pred: Any,
        regex_result: Any
    ) -> DecisionResult:
        """Try ML first, fallback to regex if confidence is low."""
        if ml_pred and ml_pred.confidence > 0.8:
            result.intent = ml_pred.intent
            result.shape = ml_pred.shape
            result.property_requested = ml_pred.property_type
            result.confidence = ml_pred.confidence
        elif regex_result and regex_result.shape:
            result.shape = regex_result.shape
            result.confidence = 0.75
        elif ml_pred:
            result.intent = ml_pred.intent
            result.shape = ml_pred.shape
            result.confidence = ml_pred.confidence
        
        # Always use regex for parameters
        if regex_result and regex_result.parameters:
            result.parameters = {
                name: val.value 
                for name, val in regex_result.parameters.items()
            }
        
        result.success = bool(result.shape and result.parameters)
        return result
    
    def _calculate(self, result: DecisionResult) -> DecisionResult:
        """Perform geometry calculations."""
        try:
            shape = result.shape
            params = result.parameters
            
            if shape == 'circle':
                radius = params.get('radius') or params.get('r')
                if radius:
                    result.calculations = self.calculator.calculate_circle(radius)
            
            elif shape == 'rectangle':
                length = params.get('length') or params.get('l')
                width = params.get('width') or params.get('w')
                if length and width:
                    result.calculations = self.calculator.calculate_rectangle(length, width)
            
            elif shape == 'square':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_square(side)
            
            elif shape == 'triangle':
                a = params.get('side_a') or params.get('a')
                b = params.get('side_b') or params.get('b')
                c = params.get('side_c') or params.get('c')
                if a and b and c:
                    result.calculations = self.calculator.calculate_triangle(a, b, c)
            
            elif shape == 'sphere':
                radius = params.get('radius') or params.get('r')
                if radius:
                    result.calculations = self.calculator.calculate_sphere(radius)
            
            elif shape == 'cylinder':
                radius = params.get('radius') or params.get('r')
                height = params.get('height') or params.get('h')
                if radius and height:
                    result.calculations = self.calculator.calculate_cylinder(radius, height)
            
            elif shape == 'cone':
                radius = params.get('radius') or params.get('r')
                height = params.get('height') or params.get('h')
                if radius and height:
                    result.calculations = self.calculator.calculate_cone(radius, height)
            
            elif shape == 'cube':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_cube(side)
            
            elif shape == 'rectangular_prism':
                length = params.get('length') or params.get('l')
                width = params.get('width') or params.get('w')
                height = params.get('height') or params.get('h')
                if length and width and height:
                    result.calculations = self.calculator.calculate_rectangular_prism(length, width, height)
            
            # Regular polygons (pentagon, hexagon, etc.)
            elif shape == 'pentagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(5, side)
            
            elif shape == 'hexagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(6, side)
            
            elif shape == 'heptagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(7, side)
            
            elif shape == 'octagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(8, side)
            
            elif shape == 'nonagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(9, side)
            
            elif shape == 'decagon':
                side = params.get('side') or params.get('s')
                if side:
                    result.calculations = self.calculator.calculate_regular_polygon(10, side)
            
            elif shape == 'polygon':
                # Generic polygon with n sides
                n = int(params.get('sides') or params.get('n') or 5)
                side = params.get('side') or params.get('s')
                if side and n >= 3:
                    result.calculations = self.calculator.calculate_regular_polygon(n, side)
            
            result.success = len(result.calculations) > 0
            
        except Exception as e:
            result.errors.append(f"Calculation error: {e}")
            result.success = False
        
        return result
    
    def explain_result(self, result: DecisionResult) -> str:
        """Generate a human-readable explanation of the result."""
        lines = []
        
        if not result.success:
            lines.append("❌ Could not process your request.")
            if result.errors:
                for error in result.errors:
                    lines.append(f"   Error: {error}")
            return "\n".join(lines)
        
        lines.append(f"✓ Shape: {result.shape.replace('_', ' ').title()}")
        lines.append(f"  Confidence: {result.confidence:.1%}")
        lines.append(f"  Intent: {result.intent}")
        
        if result.parameters:
            lines.append("\n📐 Parameters:")
            for name, value in result.parameters.items():
                lines.append(f"   {name}: {value}")
        
        if result.calculations:
            lines.append("\n📊 Calculations:")
            for calc in result.calculations:
                lines.append(f"\n   {calc.property_name.upper()}: {calc.value:.4f} {calc.unit}")
                lines.append(f"   Formula: {calc.formula}")
                for step in calc.explanation_steps:
                    lines.append(f"     → {step}")
        
        return "\n".join(lines)


def main():
    """Test the hybrid decision engine."""
    print("=" * 60)
    print("HYBRID DECISION ENGINE - Test Suite")
    print("=" * 60)
    
    # Initialize engine (without ML model for testing)
    engine = HybridDecisionEngine(strategy=DecisionStrategy.REGEX_PRIORITY)
    
    test_cases = [
        "calculate the area of a circle with radius 5",
        "rectangle length 10 width 5",
        "sphere radius 3",
        "cylinder radius 2 height 8",
        "cube side 4",
        "cone with radius 3 and height 7",
        "triangle with sides 3, 4, and 5",
        "square with side 6",
    ]
    
    for text in test_cases:
        print(f"\n{'─' * 50}")
        print(f"Input: '{text}'")
        print("─" * 50)
        
        result = engine.process(text)
        explanation = engine.explain_result(result)
        print(explanation)
    
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    main()