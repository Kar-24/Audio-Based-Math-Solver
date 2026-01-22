"""
Math Expression Calculator
Handles general math expressions including trigonometry, arithmetic, and more.
"""

import math
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MathResult:
    """Result of a math calculation."""
    success: bool
    value: float
    expression: str
    steps: List[str]
    error: Optional[str] = None


class MathCalculator:
    """
    Evaluates math expressions with step-by-step explanations.
    Supports: trig functions, arithmetic, powers, roots, etc.
    """
    
    # Trigonometric values for common angles (in degrees)
    TRIG_VALUES = {
        'sin': {0: 0, 30: 0.5, 45: math.sqrt(2)/2, 60: math.sqrt(3)/2, 90: 1, 
                180: 0, 270: -1, 360: 0},
        'cos': {0: 1, 30: math.sqrt(3)/2, 45: math.sqrt(2)/2, 60: 0.5, 90: 0,
                180: -1, 270: 0, 360: 1},
        'tan': {0: 0, 30: math.sqrt(3)/3, 45: 1, 60: math.sqrt(3), 180: 0, 360: 0}
    }
    
    def __init__(self):
        # Patterns for parsing
        self.patterns = {
            # Trig functions: sin 90, cos(45), tan 30
            'trig': re.compile(r'(sin|cos|tan|cot|sec|csc)\s*\(?(\d+\.?\d*)\)?', re.IGNORECASE),
            # Inverse trig
            'inv_trig': re.compile(r'(asin|acos|atan|arcsin|arccos|arctan)\s*\(?(-?\d+\.?\d*)\)?', re.IGNORECASE),
            # Square root: sqrt 16, square root of 25, √16
            'sqrt': re.compile(r'(?:sqrt|square\s*root\s*(?:of)?|√)\s*\(?(\d+\.?\d*)\)?', re.IGNORECASE),
            # Power: 2^3, 2 to the power of 3, 2 raised to 3, 2 squared, 2 cubed
            'power': re.compile(r'(\d+\.?\d*)\s*(?:\^|to\s*the\s*power\s*(?:of)?|raised\s*to)\s*(\d+\.?\d*)|(\d+\.?\d*)\s*(?:squared|cubed)', re.IGNORECASE),
            # Logarithm: log 100, ln 10
            'log': re.compile(r'(log|ln)\s*\(?(\d+\.?\d*)\)?', re.IGNORECASE),
            # Factorial: 5!, factorial of 5, 5 factorial
            'factorial': re.compile(r'(\d+)\s*!|factorial\s*(?:of)?\s*(\d+)|(\d+)\s*factorial', re.IGNORECASE),
            # Percentage: 20% of 150
            'percentage': re.compile(r'(\d+\.?\d*)\s*%\s*(?:of)\s*(\d+\.?\d*)', re.IGNORECASE),
            # Basic arithmetic with numbers
            'arithmetic': re.compile(r'^[\d\s\+\-\*\/\.\(\)]+$'),
        }
    
    def evaluate(self, expression: str) -> MathResult:
        """
        Evaluate a math expression and return step-by-step solution.
        """
        original = expression
        steps = []
        
        # Normalize the expression
        expr = expression.lower().strip()
        original_upper = expression.upper()
        
        # Strip common question prefixes
        prefixes = [
            r'^what\s+is\s+', r'^calculate\s+', r'^compute\s+', r'^find\s+',
            r'^solve\s+', r'^evaluate\s+', r'^the\s+', r'^a\s+',
        ]
        for prefix in prefixes:
            expr = re.sub(prefix, '', expr)
        expr = expr.strip()
        
        # Normalize operators
        expr = expr.replace('×', '*').replace('÷', '/').replace('−', '-')
        expr = expr.replace('plus', '+').replace('minus', '-')
        expr = expr.replace('times', '*').replace('divided by', '/')
        expr = expr.replace(' x ', '*').replace(' by ', '*')

        # Handle geometric segment word problems such as "B is a point on AC"
        segment_result = self._solve_point_on_segment(original_upper)
        if segment_result:
            return segment_result

        # Handle straight-line angle relationships (linear pair)
        linear_angle_result = self._solve_linear_angle(original_upper)
        if linear_angle_result:
            return linear_angle_result

        # Handle well-known trig identities explicitly
        identity_pattern = re.compile(
            r'(sin(?:\s*(?:\^|power\s*of)?\s*2|\s*square)\s*(?:theta|θ))\s*(?:\+|plus)\s*'
            r'(cos(?:\s*(?:\^|power\s*of)?\s*2|\s*square)\s*(?:theta|θ))'
        )
        if identity_pattern.search(expr):
            steps = [
                "Using the fundamental trigonometric identity: sin²θ + cos²θ = 1",
                "Regardless of the angle θ, sin²θ + cos²θ always simplifies to 1",
                "Therefore, the value of the expression is 1"
            ]
            return MathResult(True, 1.0, original, steps)
        
        try:
            # Check for different types of expressions
            
            # 1. Trigonometric expressions
            trig_matches = list(self.patterns['trig'].finditer(expr))
            if trig_matches:
                result, trig_steps = self._evaluate_trig_expression(expr, trig_matches)
                steps.extend(trig_steps)
                return MathResult(True, result, original, steps)
            
            # 2. Square root
            sqrt_match = self.patterns['sqrt'].search(expr)
            if sqrt_match:
                num = float(sqrt_match.group(1))
                result = math.sqrt(num)
                steps = [
                    f"Finding the square root of {num}",
                    f"√{num} = {result:.4f}",
                    f"The answer is {result:.4f}"
                ]
                return MathResult(True, result, original, steps)
            
            # 3. Power/exponent
            power_match = self.patterns['power'].search(expr)
            if power_match:
                # Handle normal power (groups 1,2) or squared/cubed (group 3)
                if power_match.group(1) and power_match.group(2):
                    base = float(power_match.group(1))
                    exp = float(power_match.group(2))
                elif power_match.group(3):
                    base = float(power_match.group(3))
                    exp = 2.0 if 'squared' in expr.lower() else 3.0
                else:
                    base = exp = 0
                result = base ** exp
                steps = [
                    f"Calculating {base} raised to the power of {int(exp) if exp == int(exp) else exp}",
                    f"{base}^{int(exp) if exp == int(exp) else exp} = {base} " + " × ".join([str(base)] * int(exp)) if exp == int(exp) and exp <= 5 else f"{base}^{exp}",
                    f"= {result:.4f}" if result != int(result) else f"= {int(result)}",
                    f"The answer is {result:.4f}" if result != int(result) else f"The answer is {int(result)}"
                ]
                return MathResult(True, result, original, steps)
            
            # 4. Logarithm
            log_match = self.patterns['log'].search(expr)
            if log_match:
                func = log_match.group(1).lower()
                num = float(log_match.group(2))
                if func == 'ln':
                    result = math.log(num)
                    steps = [f"Calculating natural logarithm of {num}", f"ln({num}) = {result:.4f}"]
                else:
                    result = math.log10(num)
                    steps = [f"Calculating base-10 logarithm of {num}", f"log({num}) = {result:.4f}"]
                steps.append(f"The answer is {result:.4f}")
                return MathResult(True, result, original, steps)
            
            # 5. Factorial
            fact_match = self.patterns['factorial'].search(expr)
            if fact_match:
                # Handle groups: group(1) is "5!", group(2) is "factorial of 5", group(3) is "5 factorial"
                num = int(fact_match.group(1) or fact_match.group(2) or fact_match.group(3))
                result = math.factorial(num)
                if num <= 6:
                    expansion = " × ".join(str(i) for i in range(num, 0, -1))
                    steps = [f"Calculating {num}!", f"{num}! = {expansion}", f"= {result}"]
                else:
                    steps = [f"Calculating {num}!", f"{num}! = {result}"]
                steps.append(f"The answer is {result}")
                return MathResult(True, float(result), original, steps)
            
            # 6. Percentage
            pct_match = self.patterns['percentage'].search(expr)
            if pct_match:
                pct = float(pct_match.group(1))
                num = float(pct_match.group(2))
                result = (pct / 100) * num
                steps = [
                    f"Calculating {pct}% of {num}",
                    f"Convert percentage: {pct}% = {pct}/100 = {pct/100}",
                    f"Multiply: {pct/100} × {num} = {result:.4f}",
                    f"The answer is {result:.4f}"
                ]
                return MathResult(True, result, original, steps)
            
            # 7. Try basic arithmetic (safe eval)
            result, arith_steps = self._safe_arithmetic(expr)
            if result is not None:
                steps.extend(arith_steps)
                return MathResult(True, result, original, steps)
            
            return MathResult(False, 0, original, [], f"Could not parse expression: {expression}")
            
        except Exception as e:
            return MathResult(False, 0, original, [], str(e))
    
    def _evaluate_trig_expression(self, expr: str, matches: List) -> Tuple[float, List[str]]:
        """Evaluate expression with trig functions."""
        steps = []
        result_expr = expr
        
        for match in matches:
            func = match.group(1).lower()
            angle = float(match.group(2))
            
            # Convert to radians for calculation
            rad = math.radians(angle)
            
            if func == 'sin':
                value = math.sin(rad)
                # Round to avoid floating point errors (sin(180) should be 0)
                if abs(value) < 1e-10:
                    value = 0.0
                value = round(value, 10)
                steps.append(f"sin({angle}°) = {value:.4f}")
            elif func == 'cos':
                value = math.cos(rad)
                # Round to avoid floating point errors (cos(90) should be 0)
                if abs(value) < 1e-10:
                    value = 0.0
                value = round(value, 10)
                steps.append(f"cos({angle}°) = {value:.4f}")
            elif func == 'tan':
                if angle % 180 == 90:
                    raise ValueError(f"tan({angle}°) is undefined")
                value = math.tan(rad)
                value = round(value, 10)
                steps.append(f"tan({angle}°) = {value:.4f}")
            elif func == 'cot':
                value = 1 / math.tan(rad)
                value = round(value, 10)
                steps.append(f"cot({angle}°) = {value:.4f}")
            elif func == 'sec':
                value = 1 / math.cos(rad)
                value = round(value, 10)
                steps.append(f"sec({angle}°) = {value:.4f}")
            elif func == 'csc':
                value = 1 / math.sin(rad)
                value = round(value, 10)
                steps.append(f"csc({angle}°) = {value:.4f}")
            
            # Replace in expression
            result_expr = result_expr.replace(match.group(0), str(value), 1)
        
        # Evaluate the resulting arithmetic expression
        result_expr = result_expr.replace(' ', '')
        # Safe eval for arithmetic
        result = self._eval_arithmetic(result_expr)
        
        steps.append(f"Final calculation: {result:.4f}")
        steps.append(f"The answer is {result:.4f}")
        
        return result, steps
    
    def _safe_arithmetic(self, expr: str) -> Tuple[Optional[float], List[str]]:
        """Safely evaluate arithmetic expression."""
        # Remove spaces
        expr = expr.replace(' ', '')
        
        # Only allow safe characters (including e for scientific notation)
        if not re.match(r'^[\d\.\+\-\*\/\(\)eE]+$', expr):
            return None, []
        
        try:
            result = self._eval_arithmetic(expr)
            steps = [f"Evaluating: {expr}", f"= {result:.4f}" if result != int(result) else f"= {int(result)}"]
            return result, steps
        except:
            return None, []
    
    def _eval_arithmetic(self, expr: str) -> float:
        """Evaluate simple arithmetic safely."""
        # Only allow digits, operators, parentheses, decimal points, and scientific notation
        if not re.match(r'^[\d\.\+\-\*\/\(\)\seE]+$', expr):
            raise ValueError("Invalid expression")
        
        # Use eval with restricted namespace
        return float(eval(expr, {"__builtins__": {}}, {}))

    def _solve_point_on_segment(self, text: str) -> Optional[MathResult]:
        """Solve problems where a point lies on a segment (e.g., B on AC)."""
        point_pattern = re.search(r'([A-Z])\s+IS\s+(?:A\s+)?POINT\s+ON\s+([A-Z])([A-Z])', text)
        if not point_pattern:
            return None

        point = point_pattern.group(1)
        start = point_pattern.group(2)
        end = point_pattern.group(3)

        def seg_key(seg: str) -> str:
            return ''.join(sorted(seg))

        length_pattern = re.compile(
            r'([A-Z]{2})\s*(?:IS|=)?\s*(?:THE\s+)?(?:LENGTH|MEASURE)?\s*(?:OF)?\s*(?:IS|=)?\s*(\d+\.?\d*)'
        )
        lengths: Dict[str, float] = {}
        for match in length_pattern.finditer(text):
            seg = match.group(1)
            value = float(match.group(2))
            lengths[seg_key(seg)] = value

        target_match = re.search(r'LENGTH\s+OF\s+([A-Z]{2})', text)
        if not target_match:
            return None
        target_seg = target_match.group(1)
        target_key = seg_key(target_seg)

        total_key = seg_key(start + end)
        left_key = seg_key(start + point)
        right_key = seg_key(point + end)

        total = lengths.get(total_key)
        left = lengths.get(left_key)
        right = lengths.get(right_key)

        if target_key == left_key and total is not None and right is not None:
            value = total - right
            steps = [
                f"Point {point} lies on segment {start}{end}, so {start}{point} + {point}{end} = {start}{end}.",
                f"Given {start}{end} = {total} and {point}{end} = {right}.",
                f"{start}{point} = {start}{end} - {point}{end} = {total} - {right} = {value}.",
                f"Therefore, {start}{point} = {value} units."
            ]
            return MathResult(True, value, text, steps)

        if target_key == right_key and total is not None and left is not None:
            value = total - left
            steps = [
                f"Point {point} lies on segment {start}{end}, so {start}{point} + {point}{end} = {start}{end}.",
                f"Given {start}{end} = {total} and {start}{point} = {left}.",
                f"{point}{end} = {start}{end} - {start}{point} = {total} - {left} = {value}.",
                f"Therefore, {point}{end} = {value} units."
            ]
            return MathResult(True, value, text, steps)

        if target_key == total_key and left is not None and right is not None:
            value = left + right
            steps = [
                f"Point {point} lies on segment {start}{end}, so {start}{point} + {point}{end} = {start}{end}.",
                f"Given {start}{point} = {left} and {point}{end} = {right}.",
                f"{start}{end} = {start}{point} + {point}{end} = {left} + {right} = {value}.",
                f"Therefore, {start}{end} = {value} units."
            ]
            return MathResult(True, value, text, steps)

        return None

    def _solve_linear_angle(self, text: str) -> Optional[MathResult]:
        """Solve problems using linear pairs on straight lines."""
        straight_match = re.search(r'([A-Z]{2})\s+IS\s+(?:A\s+)?STRAIGHT\s+LINE', text)
        if not straight_match:
            return None
        line_letters = set(straight_match.group(1))
        if len(line_letters) != 2:
            return None

        point_match = re.search(r'([A-Z])\s+(?:IS|AS)\s+(?:A\s+)?POINT\s+ON\s+([A-Z]{2})', text)
        if not point_match:
            return None
        vertex = point_match.group(1)
        if set(point_match.group(2)) != line_letters:
            return None

        angle_given = re.search(r'ANGLE\s+([A-Z]{3})\s*(?:IS|=)\s*(\d+\.?\d*)', text)
        if not angle_given:
            return None
        known_angle = angle_given.group(1)
        known_value = float(angle_given.group(2))

        query_match = re.search(r'(?:WHAT\s+IS|FIND)\s+(?:THE\s+MEASURE\s+OF\s+)?(?:ANGLE\s+)?([A-Z]{3})', text)
        if not query_match:
            return None
        target_angle = query_match.group(1)

        if len(known_angle) != 3 or len(target_angle) != 3:
            return None

        if known_angle[1] != vertex or target_angle[1] != vertex:
            return None

        if known_angle[0] not in line_letters and known_angle[2] not in line_letters:
            return None
        if target_angle[0] not in line_letters and target_angle[2] not in line_letters:
            return None

        endpoints = sorted(line_letters)
        ray_a, ray_b = endpoints[0], endpoints[1]

        value = 180.0 - known_value
        steps = [
            f"Since {''.join(endpoints)} is a straight line and {vertex} lies on it, the angles along rays {vertex}{ray_a} and {vertex}{ray_b} form a linear pair.",
            f"Angle {known_angle} + angle {target_angle} = 180 degrees.",
            f"Angle {target_angle} = 180 - angle {known_angle} = 180 - {known_value} = {value} degrees.",
            f"Therefore, angle {target_angle} measures {value} degrees."
        ]
        return MathResult(True, value, text, steps)


# Shape aliases for more natural language
SHAPE_ALIASES = {
    'box': 'cuboid',
    'rectangular prism': 'cuboid',
    'rectangular box': 'cuboid',
    'cuboid': 'cuboid',
}


def calculate_cuboid(length: float, width: float, height: float) -> dict:
    """Calculate cuboid/box properties."""
    volume = length * width * height
    surface_area = 2 * (length*width + width*height + height*length)
    diagonal = math.sqrt(length**2 + width**2 + height**2)
    
    return {
        'shape': 'Box (Cuboid)',
        'confidence': 0.95,
        'intent': 'calculate_all',
        'parameters': {'length': length, 'width': width, 'height': height},
        'calculations': [
            {
                'property': 'volume',
                'value': volume,
                'unit': 'cubic units',
                'formula': 'V = l × w × h',
                'steps': [
                    f'Using the formula: Volume = length × width × height',
                    f'Length = {length}, Width = {width}, Height = {height}',
                    f'Volume = {length} × {width} × {height}',
                    f'Volume = {volume:.4f}',
                    f'The volume is {volume:.4f} cubic units'
                ]
            },
            {
                'property': 'surface_area',
                'value': surface_area,
                'unit': 'square units',
                'formula': 'A = 2(lw + wh + hl)',
                'steps': [
                    f'Using the formula: Surface Area = 2(lw + wh + hl)',
                    f'lw = {length} × {width} = {length*width}',
                    f'wh = {width} × {height} = {width*height}',
                    f'hl = {height} × {length} = {height*length}',
                    f'Sum = {length*width + width*height + height*length}',
                    f'Surface Area = 2 × {length*width + width*height + height*length} = {surface_area:.4f}',
                    f'The surface area is {surface_area:.4f} square units'
                ]
            },
            {
                'property': 'space_diagonal',
                'value': diagonal,
                'unit': 'units',
                'formula': 'd = √(l² + w² + h²)',
                'steps': [
                    f'Using the formula: Diagonal = √(l² + w² + h²)',
                    f'l² = {length}² = {length**2}',
                    f'w² = {width}² = {width**2}',
                    f'h² = {height}² = {height**2}',
                    f'Sum = {length**2 + width**2 + height**2}',
                    f'Diagonal = √{length**2 + width**2 + height**2} = {diagonal:.4f}',
                    f'The space diagonal is {diagonal:.4f} units'
                ]
            }
        ]
    }


SPECIAL_TAN_VALUES = {
    30: '1/√3',
    45: '1',
    60: '√3',
}


def solve_height_distance_problem(query: str) -> Optional[dict]:
    """Handle height-and-distance word problems that rely on tangent."""
    lowered = query.lower()
    if 'angle of elevation' not in lowered and 'angle of depression' not in lowered:
        return None

    distance_match = re.search(r'(\d+\.?\d*)\s*(?:m|meter|metre)s?', query, re.IGNORECASE)
    angle_match = re.search(r'(\d+\.?\d*)\s*(?:°|degrees?)', query, re.IGNORECASE)
    if not distance_match or not angle_match:
        return None

    distance = float(distance_match.group(1))
    angle = float(angle_match.group(1))
    tan_value = math.tan(math.radians(angle))
    height_val = distance * tan_value

    steps = [
        "Model a right triangle where the tower forms the vertical (opposite) side and the given ground distance is the horizontal (adjacent) side.",
        f"Use the tangent ratio: tan({angle}°) = opposite / adjacent = height / {distance}.",
        f"Rearrange for the unknown height: height = {distance} × tan({angle}°).",
    ]

    symbolic = None
    angle_int = int(round(angle))
    if abs(angle - angle_int) < 1e-6 and angle_int in SPECIAL_TAN_VALUES:
        special = SPECIAL_TAN_VALUES[angle_int]
        if special == '1':
            symbolic = f"{distance}"
        elif special == '1/√3':
            symbolic = f"{distance} ÷ √3"
        else:
            symbolic = f"{distance} × √3"
        steps.append(f"For {angle_int}°, tan({angle_int}°) = {special}, giving height = {symbolic} meters.")

    steps.append(f"Numerically, tan({angle}°) ≈ {tan_value:.4f}, so height ≈ {height_val:.2f} meters.")

    formula = f"height = {distance} × tan({angle}°)"
    return {
        'shape': 'Height-Distance Word Problem',
        'confidence': 0.92,
        'intent': 'step-by-step procedure',
        'parameters': {
            'ground_distance_m': distance,
            'angle_degrees': angle,
        },
        'calculations': [{
            'property': 'tower height',
            'value': height_val,
            'unit': 'meters',
            'formula': formula,
            'steps': steps,
        }],
    }


def _split_linear_terms(expr: str) -> List[str]:
    expr = expr.replace(' ', '')
    terms = []
    i = 0
    while i < len(expr):
        sign = ''
        if expr[i] in '+-':
            sign = expr[i]
            i += 1
        start = i
        while i < len(expr) and expr[i] not in '+-':
            i += 1
        term = sign + expr[start:i]
        if term:
            terms.append(term)
    return terms


def _parse_coeff_part(part: str) -> Tuple[float, Optional[str]]:
    if not part:
        return 1.0, None
    match = re.fullmatch(r'([+-]?\d*\.?\d*)([a-zA-Z]?)', part)
    if not match:
        raise ValueError(f'Unsupported coefficient format: {part}')
    num_str, symbol = match.groups()
    if not num_str or num_str in ('+', '-'):
        value = 1.0 if num_str != '-' else -1.0
    else:
        value = float(num_str)
    symbol = symbol.lower() if symbol else None
    return value, symbol


def _add_coeff(existing: Optional[Tuple[float, Optional[str]]], new: Tuple[float, Optional[str]]) -> Tuple[float, Optional[str]]:
    if existing is None:
        return new
    val1, sym1 = existing
    val2, sym2 = new
    if sym1 != sym2:
        raise ValueError('Cannot combine different symbolic coefficients')
    return (val1 + val2, sym1)


def _parse_linear_equation(eq: str) -> Tuple[Tuple[float, Optional[str]], Tuple[float, Optional[str]]]:
    cleaned = eq.replace(' ', '')
    if '=' not in cleaned:
        raise ValueError('Equation must contain = sign')
    left, right = cleaned.split('=', 1)
    coeffs: Dict[str, Optional[Tuple[float, Optional[str]]]] = {'x': None, 'y': None}

    for part, multiplier in ((left, 1), (right, -1)):
        for term in _split_linear_terms(part):
            if not term:
                continue
            term_sign = multiplier
            if term[0] == '+':
                term = term[1:]
            elif term[0] == '-':
                term = term[1:]
                term_sign *= -1
            if not term:
                continue
            if term[-1].lower() in ('x', 'y'):
                var = term[-1].lower()
                coeff_part = term[:-1]
                value, symbol = _parse_coeff_part(coeff_part)
                coeffs[var] = _add_coeff(coeffs[var], (value * term_sign, symbol))
            else:
                # Constant term; no impact on unique-solution criterion
                continue

    if coeffs['x'] is None or coeffs['y'] is None:
        raise ValueError('Could not extract both x and y coefficients')
    return coeffs['x'], coeffs['y']


def _multiply_coeffs(a: Tuple[float, Optional[str]], b: Tuple[float, Optional[str]]) -> Tuple[float, Optional[str]]:
    val = a[0] * b[0]
    sym_a = a[1]
    sym_b = b[1]
    if sym_a and sym_b:
        if sym_a != sym_b:
            raise ValueError('Multiple symbolic parameters are not supported')
        raise ValueError('Quadratic symbolic terms are not supported')
    symbol = sym_a or sym_b
    return val, symbol


def _format_coeff(coeff: Tuple[float, Optional[str]]) -> str:
    value, symbol = coeff
    if symbol:
        if abs(value - 1.0) < 1e-9:
            return symbol
        if abs(value + 1.0) < 1e-9:
            return f'-{symbol}'
        return f'{value:g}{symbol}'
    return f'{value:g}'


def _format_linear_expression(symbol: Optional[str], symbol_coeff: float, constant: float) -> str:
    parts = []
    if abs(symbol_coeff) > 1e-9 and symbol:
        if abs(symbol_coeff - 1) < 1e-9:
            parts.append(symbol)
        elif abs(symbol_coeff + 1) < 1e-9:
            parts.append(f'-{symbol}')
        else:
            parts.append(f'{symbol_coeff:g}{symbol}')
    if abs(constant) > 1e-9:
        const_str = f'{constant:g}'
        if parts and constant > 0:
            const_str = f'+ {const_str}'
        elif parts:
            const_str = f'- {abs(constant):g}'
        parts.append(const_str)
    if not parts:
        return '0'
    return ' '.join(parts)


def solve_linear_unique_condition(query: str) -> Optional[dict]:
    if 'unique solution' not in query.lower():
        return None

    eq_lines = [line.strip() for line in query.splitlines() if '=' in line]
    if len(eq_lines) < 2:
        return None

    try:
        a1, b1 = _parse_linear_equation(eq_lines[0])
        a2, b2 = _parse_linear_equation(eq_lines[1])
    except ValueError:
        return None

    try:
        prod1 = _multiply_coeffs(a1, b2)
        prod2 = _multiply_coeffs(a2, b1)
    except ValueError:
        return None

    symbol = prod1[1] or prod2[1]
    symbol_coeff = 0.0
    constant = 0.0
    for value, sym, sign in ((prod1[0], prod1[1], 1), (prod2[0], prod2[1], -1)):
        if sym:
            if symbol and sym != symbol:
                return None
            symbol = sym
            symbol_coeff += sign * value
        else:
            constant += sign * value

    determinant_expr = _format_linear_expression(symbol, symbol_coeff, constant)
    steps = [
        'For a pair of linear equations a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0, a unique solution exists when a₁/a₂ ≠ b₁/b₂ (equivalently, a₁b₂ - a₂b₁ ≠ 0).',
        f'Here a₁ = { _format_coeff(a1) }, b₁ = { _format_coeff(b1) }, a₂ = { _format_coeff(a2) }, b₂ = { _format_coeff(b2) }.',
        f'Determinant: (a₁ × b₂) - (a₂ × b₁) = {_format_coeff(a1)}×{_format_coeff(b2)} - {_format_coeff(a2)}×{_format_coeff(b1)} = {determinant_expr}.',
    ]

    result_text = ''
    if symbol is None:
        if abs(constant) < 1e-9:
            steps.append('The determinant is always 0, so the system never has a unique solution.')
            result_text = 'No unique solution for any value.'
        else:
            steps.append('The determinant is a non-zero constant, so the system always has a unique solution.')
            result_text = 'Unique solution for all values.'
    else:
        if abs(symbol_coeff) < 1e-9:
            if abs(constant) < 1e-9:
                steps.append('The determinant is identically zero, so there is never a unique solution.')
                result_text = 'No unique solution for any value.'
            else:
                steps.append('The determinant is a non-zero constant, so every value of the parameter gives a unique solution.')
                result_text = 'Unique solution for all parameter values.'
        else:
            threshold = -constant / symbol_coeff
            steps.append(f'Set determinant ≠ 0: {determinant_expr} ≠ 0 ⇒ {symbol} ≠ {threshold:g}.')
            result_text = f'{symbol} ≠ {threshold:g}'

    return {
        'shape': 'Linear Equation Pair',
        'confidence': 0.9,
        'intent': 'unique_solution_condition',
        'parameters': {
            'equation_1': eq_lines[0],
            'equation_2': eq_lines[1],
        },
        'calculations': [{
            'property': 'Unique solution condition',
            'value': result_text,
            'unit': '',
            'formula': 'a₁b₂ - a₂b₁ ≠ 0',
            'steps': steps,
        }],
    }


def parse_math_query(query: str) -> Optional[dict]:
    """
    Parse a query that might be a math expression or a box/cuboid calculation.
    Returns a result dict if it's a math expression, None otherwise.
    """
    original_query = query
    query_lower = query.lower().strip()
    
    # Strip common question prefixes
    prefixes_to_strip = [
        r'^what\s+is\s+',
        r'^calculate\s+',
        r'^compute\s+',
        r'^find\s+',
        r'^solve\s+',
        r'^evaluate\s+',
        r'^tell\s+me\s+',
        r'^give\s+me\s+',
        r'^what\'?s\s+',
        r'^can\s+you\s+(?:calculate|find|solve|compute)\s+',
        r'^please\s+(?:calculate|find|solve|compute)\s+',
        r'^the\s+(?:value\s+of\s+)?',
    ]
    
    cleaned_query = query_lower
    for prefix in prefixes_to_strip:
        cleaned_query = re.sub(prefix, '', cleaned_query, flags=re.IGNORECASE)
    cleaned_query = cleaned_query.strip()
    
    # Check for box/cuboid with dimensions
    box_patterns = [
        r'(?:volume\s+(?:of\s+)?)?(?:a\s+)?(?:box|cuboid|rectangular\s+(?:prism|box))\s+(?:with\s+)?(?:dimensions?\s+)?(\d+\.?\d*)\s*[,\s]+(\d+\.?\d*)\s*(?:and\s+)?[,\s]+(\d+\.?\d*)',
        r'(?:box|cuboid)\s+(?:with\s+)?(?:length|l)\s*[=:]?\s*(\d+\.?\d*)\s*(?:width|w)\s*[=:]?\s*(\d+\.?\d*)\s*(?:height|h)\s*[=:]?\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:by|x|×)\s*(\d+\.?\d*)\s*(?:by|x|×)\s*(\d+\.?\d*)\s*(?:box|cuboid)?',
    ]
    
    for pattern in box_patterns:
        match = re.search(pattern, cleaned_query)
        if match:
            dims = [float(match.group(i)) for i in range(1, 4)]
            return calculate_cuboid(dims[0], dims[1], dims[2])

    height_distance = solve_height_distance_problem(original_query)
    if height_distance:
        return height_distance

    unique_condition = solve_linear_unique_condition(original_query)
    if unique_condition:
        return unique_condition
    
    calc = MathCalculator()
    result = calc.evaluate(cleaned_query)
    
    if result.success:
        return {
            'shape': 'Math Expression',
            'confidence': 0.95,
            'intent': 'calculate',
            'parameters': {'expression': original_query},
            'calculations': [{
                'property': 'result',
                'value': result.value,
                'unit': '',
                'formula': cleaned_query,
                'steps': result.steps
            }]
        }
    
    return None
