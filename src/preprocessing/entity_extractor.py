"""
Advanced Entity Extractor using Regular Expressions
Extracts geometric parameters, measurements, and units from natural language.
"""

import regex as re  # Using 'regex' package for advanced features
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math


class UnitType(Enum):
    """Types of measurement units."""
    LENGTH = "length"
    AREA = "area"
    VOLUME = "volume"
    ANGLE = "angle"
    UNKNOWN = "unknown"


@dataclass
class ExtractedValue:
    """Represents an extracted numerical value with context."""
    value: float
    original_text: str
    parameter_name: Optional[str] = None
    unit: Optional[str] = None
    unit_type: UnitType = UnitType.UNKNOWN
    confidence: float = 1.0
    position: Tuple[int, int] = (0, 0)  # Start, end positions in text


@dataclass
class ExtractionResult:
    """Complete extraction result from text."""
    shape: Optional[str] = None
    parameters: Dict[str, ExtractedValue] = field(default_factory=dict)
    raw_values: List[ExtractedValue] = field(default_factory=list)
    units_detected: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    original_text: str = ""
    normalized_text: str = ""


class RegexEntityExtractor:
    """
    Advanced regex-based entity extractor for geometry calculations.
    Uses sophisticated patterns to extract shapes, measurements, and units.
    """
    
    # ==================== REGEX PATTERNS ====================
    
    # Number patterns (supports integers, decimals, fractions, scientific notation)
    NUMBER_PATTERNS = {
        'decimal': r'(?P<num>-?\d+\.?\d*)',
        'fraction': r'(?P<num_frac>\d+\s*/\s*\d+)',
        'mixed_fraction': r'(?P<num_mixed>\d+\s+\d+\s*/\s*\d+)',
        'scientific': r'(?P<num_sci>-?\d+\.?\d*\s*[eE]\s*[+-]?\d+)',
        'word_number': r'(?P<num_word>zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|quarter|third)',
        'pi_multiple': r'(?P<num_pi>(?:\d*\.?\d*\s*)?[πpi])',
    }
    
    # Combined number pattern
    NUMBER_REGEX = re.compile(
        r'''
        (?P<number>
            # Scientific notation first (most specific)
            -?\d+\.?\d*\s*[eE]\s*[+-]?\d+
            |
            # Mixed fractions (e.g., "2 1/2")
            \d+\s+\d+\s*/\s*\d+
            |
            # Simple fractions (e.g., "1/2")
            \d+\s*/\s*\d+
            |
            # Decimal or integer
            -?\d+\.?\d*
            |
            # Pi notation
            (?:\d*\.?\d*\s*)?[πpi]
        )
        ''',
        re.VERBOSE | re.IGNORECASE
    )
    
    # Unit patterns
    UNIT_PATTERNS = {
        # Length units
        'length': {
            'pattern': r'(?P<unit>(?:milli|centi|deci|kilo)?met(?:er|re)s?|mm|cm|dm|km|m|inch(?:es)?|in|foot|feet|ft|yard|yards?|yd|mile|miles?|mi)',
            'type': UnitType.LENGTH,
            'conversions': {
                'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001, 'millimetre': 0.001,
                'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01, 'centimetre': 0.01,
                'dm': 0.1, 'decimeter': 0.1, 'decimeters': 0.1, 'decimetre': 0.1,
                'm': 1.0, 'meter': 1.0, 'meters': 1.0, 'metre': 1.0, 'metres': 1.0,
                'km': 1000.0, 'kilometer': 1000.0, 'kilometers': 1000.0, 'kilometre': 1000.0,
                'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
                'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
                'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
                'mi': 1609.34, 'mile': 1609.34, 'miles': 1609.34,
            }
        },
        # Area units
        'area': {
            'pattern': r'(?P<unit>(?:square\s+)?(?:(?:milli|centi|kilo)?met(?:er|re)s?|mm|cm|km|m|inch(?:es)?|in|foot|feet|ft|yard|yards?|yd|mile|miles?|mi)(?:\s*(?:squared|\^2|²))?|(?:sq\.?\s*)?(?:mm|cm|m|km|in|ft|yd|mi)|acres?|hectares?|ha)',
            'type': UnitType.AREA,
        },
        # Volume units
        'volume': {
            'pattern': r'(?P<unit>(?:cubic\s+)?(?:(?:milli|centi|kilo)?met(?:er|re)s?|mm|cm|km|m|inch(?:es)?|in|foot|feet|ft|yard|yards?|yd)(?:\s*(?:cubed|\^3|³))?|(?:cu\.?\s*)?(?:mm|cm|m|km|in|ft|yd)|lit(?:er|re)s?|ml|l|gallon|gallons?|gal|pint|pints?|pt|quart|quarts?|qt)',
            'type': UnitType.VOLUME,
        },
        # Angle units
        'angle': {
            'pattern': r'(?P<unit>degrees?|deg|°|radians?|rad|grads?|gradians?)',
            'type': UnitType.ANGLE,
        }
    }
    
    # Shape patterns with variations
    SHAPE_PATTERNS = {
        # 2D Shapes
        'circle': r'\b(?:circle|circular|round)\b',
        'square': r'\b(?:square)\b',
        'rectangle': r'\b(?:rectangle|rectangular|oblong)\b',
        'triangle': r'\b(?:triangle|triangular)(?!\s+(?:right|right-angled))\b',
        'right_triangle': r'\b(?:right(?:\s*-?\s*angled)?\s+triangle|triangle\s+(?:that\s+is\s+)?right(?:\s*-?\s*angled)?)\b',
        'trapezoid': r'\b(?:trapezoid|trapezium|trapezoidal)\b',
        'parallelogram': r'\b(?:parallelogram)\b',
        'rhombus': r'\b(?:rhombus|rhombi|diamond(?:\s+shape)?)\b',
        'ellipse': r'\b(?:ellipse|elliptical|oval)\b',
        # Regular polygons - each with its own pattern for correct shape recognition
        'pentagon': r'\b(?:pentagon|pentagonal)\b',
        'hexagon': r'\b(?:hexagon|hexagonal)\b',
        'heptagon': r'\b(?:heptagon|heptagonal|septagon)\b',
        'octagon': r'\b(?:octagon|octagonal)\b',
        'nonagon': r'\b(?:nonagon|enneagon)\b',
        'decagon': r'\b(?:decagon)\b',
        'polygon': r'\b(?:polygon)\b',
        
        # 3D Shapes
        'sphere': r'\b(?:sphere|spherical|ball|globe)\b',
        'cube': r'\b(?:cube|cubic)\b',
        'cylinder': r'\b(?:cylinder|cylindrical|tube|pipe)\b',
        'cone': r'\b(?:cone|conical)\b',
        'rectangular_prism': r'\b(?:rectangular\s+prism|cuboid|box|block)\b',
        'pyramid': r'\b(?:pyramid|pyramidal)\b',
        'torus': r'\b(?:torus|doughnut|donut)\b',
        'prism': r'\b(?:prism|triangular\s+prism|hexagonal\s+prism)\b',
    }
    
    # Parameter patterns (what the number represents)
    PARAMETER_PATTERNS = {
        'radius': r'\b(?:radius|radii|rad|r)\s*(?:=|is|:|\s)\s*',
        'diameter': r'\b(?:diameter|dia|d)\s*(?:=|is|:|\s)\s*',
        'side': r'\b(?:side(?:\s+length)?|edge|s)\s*(?:=|is|:|\s)\s*',
        'length': r'\b(?:length|len|l)\s*(?:=|is|:|\s)\s*',
        'width': r'\b(?:width|wid|w|breadth)\s*(?:=|is|:|\s)\s*',
        'height': r'\b(?:height|h|tall(?:ness)?|altitude)\s*(?:=|is|:|\s)\s*',
        'base': r'\b(?:base|bottom|b)\s*(?:=|is|:|\s)\s*',
        'base1': r'\b(?:(?:first|top|upper)\s+(?:base|parallel\s+side)|base\s*1|b1)\s*(?:=|is|:|\s)\s*',
        'base2': r'\b(?:(?:second|bottom|lower)\s+(?:base|parallel\s+side)|base\s*2|b2)\s*(?:=|is|:|\s)\s*',
        'semi_major': r'\b(?:semi[-\s]?major(?:\s+axis)?|major\s+radius|a)\s*(?:=|is|:|\s)\s*',
        'semi_minor': r'\b(?:semi[-\s]?minor(?:\s+axis)?|minor\s+radius|b)\s*(?:=|is|:|\s)\s*',
        'slant_height': r'\b(?:slant(?:\s+height)?)\s*(?:=|is|:|\s)\s*',
        'apothem': r'\b(?:apothem)\s*(?:=|is|:|\s)\s*',
        'side_a': r'\b(?:(?:first|side\s*a|a)\s*(?:side)?)\s*(?:=|is|:|\s)\s*',
        'side_b': r'\b(?:(?:second|side\s*b|b)\s*(?:side)?)\s*(?:=|is|:|\s)\s*',
        'side_c': r'\b(?:(?:third|side\s*c|c)\s*(?:side)?)\s*(?:=|is|:|\s)\s*',
    }
    
    # Word to number mapping
    WORD_NUMBERS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    }
    
    def __init__(self):
        """Initialize the entity extractor with compiled patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns for efficiency."""
        # Compile shape patterns
        self.shape_regex = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.SHAPE_PATTERNS.items()
        }
        
        # Compile parameter patterns with number capture
        self.param_regex = {}
        for param, pattern in self.PARAMETER_PATTERNS.items():
            full_pattern = pattern + r'(' + self.NUMBER_REGEX.pattern + r')'
            self.param_regex[param] = re.compile(full_pattern, re.IGNORECASE | re.VERBOSE)
        
        # Compile unit patterns
        self.unit_regex = {
            name: re.compile(info['pattern'], re.IGNORECASE)
            for name, info in self.UNIT_PATTERNS.items()
        }
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all geometric entities from text.
        
        Args:
            text: Natural language input
            
        Returns:
            ExtractionResult with all extracted entities
        """
        result = ExtractionResult(original_text=text)
        result.normalized_text = self._normalize_text(text)
        
        # Extract shape
        result.shape = self._extract_shape(result.normalized_text)
        
        # Extract named parameters
        result.parameters = self._extract_parameters(result.normalized_text)
        
        # Extract all raw numerical values
        result.raw_values = self._extract_all_numbers(result.normalized_text)
        
        # Extract units
        result.units_detected = self._extract_units(result.normalized_text)
        
        # If no named parameters found, try to infer from position
        if not result.parameters and result.raw_values:
            result.parameters = self._infer_parameters(
                result.shape, result.raw_values
            )
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching."""
        normalized = text.lower().strip()
        
        # Replace common variations
        replacements = [
            (r'\s+', ' '),  # Multiple spaces to single
            (r'[–—]', '-'),  # Normalize dashes
            (r'[''`]', "'"),  # Normalize quotes
            (r'[""]', '"'),
            (r'×|✕|✖', 'x'),  # Multiplication signs
            (r'÷', '/'),  # Division
            (r'π', 'pi'),  # Pi symbol
            (r'²', '^2'),  # Superscripts
            (r'³', '^3'),
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
    
    def _extract_shape(self, text: str) -> Optional[str]:
        """Extract the shape name from text."""
        # Check specific patterns first (right_triangle before triangle)
        priority_order = [
            'right_triangle', 'rectangular_prism', 'triangle',
            'circle', 'square', 'rectangle', 'trapezoid',
            'parallelogram', 'rhombus', 'ellipse',
            # Regular polygons (check specific ones before generic 'polygon')
            'pentagon', 'hexagon', 'heptagon', 'octagon', 'nonagon', 'decagon', 'polygon',
            'sphere', 'cube', 'cylinder', 'cone', 'pyramid',
            'torus', 'prism'
        ]
        
        for shape_name in priority_order:
            if shape_name in self.shape_regex:
                if self.shape_regex[shape_name].search(text):
                    return shape_name
        
        return None
    
    def _extract_parameters(self, text: str) -> Dict[str, ExtractedValue]:
        """Extract named parameters with their values."""
        parameters = {}
        
        for param_name, regex in self.param_regex.items():
            match = regex.search(text)
            if match:
                try:
                    value_str = match.group(1)
                    value = self._parse_number(value_str)
                    if value is not None:
                        parameters[param_name] = ExtractedValue(
                            value=value,
                            original_text=match.group(0),
                            parameter_name=param_name,
                            position=(match.start(), match.end()),
                            confidence=0.95
                        )
                except (ValueError, IndexError):
                    pass
        
        return parameters
    
    def _extract_all_numbers(self, text: str) -> List[ExtractedValue]:
        """Extract all numerical values from text."""
        values = []
        
        for match in self.NUMBER_REGEX.finditer(text):
            value_str = match.group('number')
            value = self._parse_number(value_str)
            
            if value is not None:
                # Check for following unit
                unit = None
                unit_match = re.search(
                    r'^\s*(' + '|'.join(
                        info['pattern'] for info in self.UNIT_PATTERNS.values()
                    ) + ')',
                    text[match.end():],
                    re.IGNORECASE
                )
                if unit_match:
                    unit = unit_match.group(1)
                
                values.append(ExtractedValue(
                    value=value,
                    original_text=value_str,
                    unit=unit,
                    position=(match.start(), match.end()),
                    confidence=0.9
                ))
        
        return values
    
    def _extract_units(self, text: str) -> List[str]:
        """Extract all units mentioned in text."""
        units = []
        for name, regex in self.unit_regex.items():
            for match in regex.finditer(text):
                units.append(match.group())
        return list(set(units))
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse a number from various formats."""
        text = text.strip().lower()
        
        # Check for word numbers
        if text in self.WORD_NUMBERS:
            return float(self.WORD_NUMBERS[text])
        
        # Check for pi notation
        if 'pi' in text:
            text = text.replace('pi', '')
            if text.strip() == '':
                return math.pi
            try:
                return float(text) * math.pi
            except ValueError:
                return math.pi
        
        # Check for fractions
        if '/' in text:
            parts = text.split()
            if len(parts) == 2:  # Mixed fraction: "2 1/2"
                try:
                    whole = float(parts[0])
                    frac_parts = parts[1].split('/')
                    fraction = float(frac_parts[0]) / float(frac_parts[1])
                    return whole + fraction
                except (ValueError, IndexError, ZeroDivisionError):
                    pass
            else:  # Simple fraction: "1/2"
                try:
                    frac_parts = text.split('/')
                    return float(frac_parts[0]) / float(frac_parts[1])
                except (ValueError, IndexError, ZeroDivisionError):
                    pass
        
        # Check for scientific notation
        if 'e' in text:
            try:
                return float(text.replace(' ', ''))
            except ValueError:
                pass
        
        # Try direct conversion
        try:
            return float(text)
        except ValueError:
            return None
    
    def _infer_parameters(
        self, 
        shape: Optional[str], 
        raw_values: List[ExtractedValue]
    ) -> Dict[str, ExtractedValue]:
        """Infer parameter names based on shape and value positions."""
        if not shape or not raw_values:
            return {}
        
        # Parameter order for each shape
        param_order = {
            'circle': ['radius'],
            'square': ['side'],
            'rectangle': ['length', 'width'],
            'triangle': ['side_a', 'side_b', 'side_c'],
            'right_triangle': ['base', 'height'],
            'trapezoid': ['base1', 'base2', 'height'],
            'parallelogram': ['base', 'height', 'side'],
            'ellipse': ['semi_major', 'semi_minor'],
            'sphere': ['radius'],
            'cube': ['side'],
            'cylinder': ['radius', 'height'],
            'cone': ['radius', 'height'],
            'rectangular_prism': ['length', 'width', 'height'],
            'pyramid': ['base', 'height'],
        }
        
        expected_params = param_order.get(shape, [])
        inferred = {}
        
        for i, value in enumerate(raw_values):
            if i < len(expected_params):
                param_name = expected_params[i]
                inferred[param_name] = ExtractedValue(
                    value=value.value,
                    original_text=value.original_text,
                    parameter_name=param_name,
                    unit=value.unit,
                    position=value.position,
                    confidence=0.7  # Lower confidence for inferred parameters
                )
        
        return inferred
    
    def validate_extraction(self, result: ExtractionResult) -> List[str]:
        """Validate the extraction result and return any errors."""
        errors = []
        
        # Check for negative values where not allowed
        for name, param in result.parameters.items():
            if param.value <= 0:
                errors.append(f"Parameter '{name}' must be positive, got {param.value}")
        
        # Check for required parameters based on shape
        required = {
            'circle': ['radius'],
            'rectangle': ['length', 'width'],
            'triangle': ['side_a', 'side_b', 'side_c'],
            'sphere': ['radius'],
            'cylinder': ['radius', 'height'],
        }
        
        if result.shape and result.shape in required:
            for param in required[result.shape]:
                if param not in result.parameters:
                    errors.append(f"Missing required parameter '{param}' for {result.shape}")
        
        return errors


class AdvancedPatternMatcher:
    """
    Advanced pattern matching with fuzzy matching and context awareness.
    """
    
    # Fuzzy matching patterns using regex with error tolerance
    FUZZY_PATTERNS = {
        # Allow up to 2 character substitutions/insertions/deletions
        'circle': r'\b(?:c[iy]rcl?e|circel|cicle)\b',
        'rectangle': r'\b(?:recta?ngl?e|rectangel|rectanlge)\b',
        'triangle': r'\b(?:tria?ngl?e|triangel|trianlge)\b',
        'sphere': r'\b(?:sphe?re?|spehere|shpere)\b',
    }
    
    @classmethod
    def fuzzy_shape_match(cls, text: str, max_distance: int = 2) -> Optional[str]:
        """
        Match shape names with tolerance for typos.
        
        Args:
            text: Input text
            max_distance: Maximum edit distance allowed
            
        Returns:
            Matched shape name or None
        """
        text_lower = text.lower()
        
        for shape, pattern in cls.FUZZY_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return shape
        
        # Fallback to basic fuzzy matching
        shapes = ['circle', 'square', 'rectangle', 'triangle', 'sphere', 
                  'cube', 'cylinder', 'cone', 'pyramid']
        
        for word in text_lower.split():
            for shape in shapes:
                if cls._levenshtein_distance(word, shape) <= max_distance:
                    return shape
        
        return None
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate the Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return AdvancedPatternMatcher._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def main():
    """Test the entity extractor."""
    print("=" * 60)
    print("REGEX ENTITY EXTRACTOR - Test Suite")
    print("=" * 60)
    
    extractor = RegexEntityExtractor()
    
    test_cases = [
        "calculate the area of a circle with radius 5",
        "rectangle with length 10.5 cm and width 7.3 cm",
        "what is the volume of a sphere with r = 3.14",
        "find the perimeter of a triangle with sides 3, 4, and 5",
        "cylinder: radius is 2 meters, height is 10 meters",
        "cone with base radius 5 inches and height 12 inches",
        "rectangular prism 4 x 5 x 6",
        "ellipse with semi-major axis 8 and semi-minor 5",
        "trapezoid with bases 10 and 6, height 4",
        "area of a 5cm radius circle",
        "right triangle base 3 height 4",
        "cube with side 7.5",
    ]
    
    for text in test_cases:
        print(f"\n{'─' * 50}")
        print(f"Input: '{text}'")
        result = extractor.extract(text)
        
        print(f"  Shape: {result.shape}")
        print(f"  Parameters:")
        for name, val in result.parameters.items():
            print(f"    {name}: {val.value} (confidence: {val.confidence:.2f})")
        if result.units_detected:
            print(f"  Units: {result.units_detected}")
        if result.errors:
            print(f"  Errors: {result.errors}")
    
    # Test fuzzy matching
    print(f"\n{'=' * 60}")
    print("FUZZY MATCHING - Typo Tolerance Test")
    print("=" * 60)
    
    typo_tests = [
        "circl radius 5",  # Missing 'e'
        "rectagle 10 by 5",  # Missing 'n'
        "shpere with radius 3",  # Transposed letters
        "trianlge sides 3 4 5",  # Typo
    ]
    
    for text in typo_tests:
        shape = AdvancedPatternMatcher.fuzzy_shape_match(text)
        print(f"  '{text}' → Shape: {shape}")
    
    print("\n" + "=" * 60)
    print("Entity extraction tests complete!")


if __name__ == "__main__":
    main()
