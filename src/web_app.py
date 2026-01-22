"""
Minimal Web Interface for ML-Powered Geometry Solver
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Load environment variables from a local .env file if present (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.isfile(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Do not override explicitly-set environment variables
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as env_err:  # pragma: no cover - best-effort helper
        print(f"Warning: could not read .env file: {env_err}")

# Make sure the geometry_solver package is importable (supports sibling repo layout)
GEOMETRY_SOLVER_DIR = None
for candidate in (
    os.path.abspath(os.path.join(CURRENT_DIR, '..', 'geometry_solver')),
    os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', 'geometry_solver')),
):
    if os.path.isdir(candidate):
        GEOMETRY_SOLVER_DIR = candidate
        break

if GEOMETRY_SOLVER_DIR:
    geometry_solver_parent = os.path.dirname(GEOMETRY_SOLVER_DIR)
    if geometry_solver_parent not in sys.path:
        sys.path.insert(0, geometry_solver_parent)

from inference import GeometryInferenceEngine
from math_calculator import parse_math_query

try:
    from textbook_knowledge import TextbookKnowledgeBase
    TEXTBOOK_SUPPORT = True
except Exception:
    TextbookKnowledgeBase = None  # type: ignore
    TEXTBOOK_SUPPORT = False

try:
    from geometry_solver.input_parser import InputParser
    from geometry_solver.theory import find_theory_topic, format_theory_response
    from geometry_solver.diagram_generator import generate_diagram
    from geometry_solver.shapes import (
        Circle,
        Rectangle,
        Square,
        Triangle,
        RightTriangle,
        Trapezoid,
        Parallelogram,
        Ellipse,
        Sphere,
        Cube,
        Cylinder,
        Cone,
        RectangularPrism,
        Pyramid,
    )
    GEOMETRY_SOLVER_AVAILABLE = True
except ImportError:
    GEOMETRY_SOLVER_AVAILABLE = False

app = Flask(__name__, template_folder='templates')

# Initialize shared singletons
engine = None
textbook_kb = None
gemini_model = None

# Configure Gemini API (key is read from environment for safety)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_engine():
    global engine
    if engine is None:
        engine = GeometryInferenceEngine(model_path='../models/baseline')
    return engine


def get_textbook_kb():
    if not TEXTBOOK_SUPPORT:
        return None
    global textbook_kb
    if textbook_kb is None:
        try:
            textbook_kb = TextbookKnowledgeBase()
        except Exception:
            textbook_kb = None
    return textbook_kb


def get_gemini_model():
    """Initialize and return Gemini model."""
    if not GEMINI_AVAILABLE:
        return None
    if not GEMINI_API_KEY:
        print("Gemini API key not configured. Set GEMINI_API_KEY env var to enable Gemini features.")
        return None
    global gemini_model
    if gemini_model is None:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Use the correct model names from the API - models/gemini-2.5-flash is the stable one
            model_names = [
                'models/gemini-2.5-flash',
                'models/gemini-flash-latest',
                'models/gemini-2.0-flash',
                'models/gemini-pro-latest'
            ]
            for model_name in model_names:
                try:
                    gemini_model = genai.GenerativeModel(model_name)
                    print(f"Initialized Gemini model: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to init {model_name}: {str(e)[:100]}")
                    continue
            
            if gemini_model is None:
                print("All Gemini models failed to initialize")
                return None
        except Exception as e:
            print(f"Gemini configuration error: {e}")
            gemini_model = None
            return None
    return gemini_model


if GEOMETRY_SOLVER_AVAILABLE:

    def _build_circle(values: List[float]) -> Circle:
        return Circle(values[0])


    def _build_rectangle(values: List[float]) -> Rectangle:
        return Rectangle(values[0], values[1])


    def _build_square(values: List[float]) -> Square:
        return Square(values[0])


    def _build_triangle(values: List[float]) -> Triangle:
        return Triangle(values[0], values[1], values[2])


    def _build_right_triangle(values: List[float]) -> RightTriangle:
        return RightTriangle(values[0], values[1])


    def _build_trapezoid(values: List[float]) -> Trapezoid:
        return Trapezoid(values[0], values[1], values[2])


    def _build_parallelogram(values: List[float]) -> Parallelogram:
        side = values[2] if len(values) > 2 else values[0]
        return Parallelogram(values[0], values[1], side)


    def _build_ellipse(values: List[float]) -> Ellipse:
        return Ellipse(values[0], values[1])


    def _build_sphere(values: List[float]) -> Sphere:
        return Sphere(values[0])


    def _build_cube(values: List[float]) -> Cube:
        return Cube(values[0])


    def _build_cylinder(values: List[float]) -> Cylinder:
        return Cylinder(values[0], values[1])


    def _build_cone(values: List[float]) -> Cone:
        return Cone(values[0], values[1])


    def _build_rectangular_prism(values: List[float]) -> RectangularPrism:
        return RectangularPrism(values[0], values[1], values[2])


    def _build_pyramid(values: List[float]) -> Pyramid:
        return Pyramid(values[0], values[1])


    GEOMETRY_SHAPE_SPECS: Dict[str, Dict[str, object]] = {
        'circle': {
            'builder': _build_circle,
            'min_inputs': 1,
            'params': ['radius'],
            'example': 'area of a circle with radius 5',
        },
        'rectangle': {
            'builder': _build_rectangle,
            'min_inputs': 2,
            'params': ['length', 'width'],
            'example': 'rectangle length 10 width 4',
        },
        'square': {
            'builder': _build_square,
            'min_inputs': 1,
            'params': ['side'],
            'example': 'square side 5',
        },
        'triangle': {
            'builder': _build_triangle,
            'min_inputs': 3,
            'params': ['side_a', 'side_b', 'side_c'],
            'example': 'triangle 3 4 5',
        },
        'right triangle': {
            'builder': _build_right_triangle,
            'min_inputs': 2,
            'params': ['base', 'height'],
            'example': 'right triangle base 6 height 8',
        },
        'trapezoid': {
            'builder': _build_trapezoid,
            'min_inputs': 3,
            'params': ['base1', 'base2', 'height'],
            'example': 'trapezoid bases 8 and 5 height 4',
        },
        'parallelogram': {
            'builder': _build_parallelogram,
            'min_inputs': 2,
            'params': ['base', 'height', 'side'],
            'example': 'parallelogram base 6 height 3 side 5',
        },
        'ellipse': {
            'builder': _build_ellipse,
            'min_inputs': 2,
            'params': ['semi_major', 'semi_minor'],
            'example': 'ellipse semi major 9 semi minor 4',
        },
        'sphere': {
            'builder': _build_sphere,
            'min_inputs': 1,
            'params': ['radius'],
            'example': 'volume of sphere radius 3',
        },
        'cube': {
            'builder': _build_cube,
            'min_inputs': 1,
            'params': ['side'],
            'example': 'cube side 4',
        },
        'cylinder': {
            'builder': _build_cylinder,
            'min_inputs': 2,
            'params': ['radius', 'height'],
            'example': 'cylinder radius 4 height 10',
        },
        'cone': {
            'builder': _build_cone,
            'min_inputs': 2,
            'params': ['radius', 'height'],
            'example': 'cone radius 5 height 12',
        },
        'rectangular prism': {
            'builder': _build_rectangular_prism,
            'min_inputs': 3,
            'params': ['length', 'width', 'height'],
            'example': 'rectangular prism 5 6 7',
        },
        'pyramid': {
            'builder': _build_pyramid,
            'min_inputs': 2,
            'params': ['base_side', 'height'],
            'example': 'pyramid base 6 height 9',
        },
    }

    SHAPE_ALIASES = {
        'box': 'rectangular prism',
    }
else:
    GEOMETRY_SHAPE_SPECS = {}
    SHAPE_ALIASES = {}

EXAMPLE_TRIGGER_PATTERN = re.compile(r"\b(?:example|ex\.?)\s*\d+(?:\.\d+)?[a-z]?", re.IGNORECASE)
SOLUTION_SECTION_PATTERN = re.compile(r"\bsolution\s*:\s*", re.IGNORECASE)
SECONDARY_SOLUTION_MARKERS: Tuple[str, ...] = (
    'A NOTE TO THE READER',
    'Hint :',
    'Hint:',
    'Using the section formula',
    'Using section formula',
    'Using the distance formula',
    'Using distance formula',
    'Using the midpoint formula',
    'Using midpoint formula',
    'Using Pythagoras',
    'Let P(',
    'Let Q(',
    'therefore',
    'hence',
)


def _normalize_shape_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lowered = name.lower()
    return SHAPE_ALIASES.get(lowered, lowered)


def _looks_like_textbook_example(query: str) -> bool:
    if not query:
        return False
    if EXAMPLE_TRIGGER_PATTERN.search(query):
        return True
    lowered = query.lower()
    if 'example question' in lowered or 'textbook example' in lowered:
        return True
    if 'example' in lowered and 'chapter' in lowered:
        return True
    return False


def _split_textbook_solution(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return '', None
    normalized = text.strip()
    parts = SOLUTION_SECTION_PATTERN.split(normalized, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    # Heuristic: split just before coordinate formulas like "x = ..., y = ..."
    coord_match = re.search(r"\b[xy]\s*=", normalized, re.IGNORECASE)
    if coord_match and coord_match.start() > 20:
        summary = normalized[:coord_match.start()].strip(' -:\n')
        solution = normalized[coord_match.start():].strip()
        if summary and solution:
            return summary, solution

    lowered = normalized.lower()
    for marker in SECONDARY_SOLUTION_MARKERS:
        idx = lowered.find(marker.lower())
        if idx != -1 and idx > 20:
            summary = normalized[:idx].strip(' -:\n')
            solution = normalized[idx:].strip()
            if summary and solution:
                return summary, solution

    return normalized, None


def _is_word_problem(query: str) -> bool:
    """Detect if query is a word problem."""
    if not query:
        return False
    lowered = query.lower()
    word_problem_indicators = (
        'how many',
        'find the number',
        'a person',
        'students',
        'teacher',
        'garden',
        'field',
        'park',
        'road',
        'building',
        'tower',
        'tree',
        'ladder',
        'rope',
        'wire',
        'cost',
        'price',
        'rupees',
        'dollars',
        'age',
        'years old',
        'if',
        'when',
        'ratio',
        'percentage',
        'speed',
        'distance',
        'time',
    )
    return any(indicator in lowered for indicator in word_problem_indicators)


def solve_with_gemini(query: str) -> Optional[Dict[str, object]]:
    """Use Gemini API to solve complex problems with step-by-step solutions."""
    print(f"Calling Gemini for: {query[:60]}...")
    
    model = get_gemini_model()
    if not model:
        print("Gemini model not available")
        return None
    
    try:
        # Detect the type of question
        query_lower = query.lower()
        
        # Check if it's a theory/explanation question
        is_theory = any(word in query_lower for word in [
            'explain', 'what is', 'define', 'describe', 'why', 'how does', 'meaning of'
        ])
        
        # Check if it's a numeric calculation problem (has numbers)
        import re
        has_numbers = bool(re.search(r'\d+', query))
        
        if is_theory:
            # Simple explanation prompt for theory questions
            prompt = f"""Explain this concept simply:

{query}

INSTRUCTIONS:
- Give a clear, simple explanation in 2-3 sentences
- Use everyday language that anyone can understand
- If helpful, give one simple example
- No special formatting, asterisks, or symbols
- Keep it short and easy to read aloud

Format:
SUMMARY: [2-3 sentence simple explanation]
STEPS:
1. [key point one]
2. [key point two]
3. [simple example if helpful]
ANSWER: [one sentence conclusion]"""

        elif has_numbers:
            # Numeric problem with equations
            prompt = f"""Solve this math problem showing equations:

{query}

INSTRUCTIONS:
- Show the equation or formula first
- Substitute the numbers step by step
- Show each calculation clearly
- Use 3-5 steps maximum
- Write equations in a readable way (e.g., "3 squared = 9" or "3^2 = 9")
- No asterisks or special formatting
- Keep it simple

Format:
SUMMARY: [what we need to find]
STEPS:
1. [write the formula or equation]
2. [substitute the values: show the equation with numbers]
3. [calculate: show the math]
4. [simplify to get the answer]
ANSWER: [final numeric answer with units]

Example for "find hypotenuse with base 3 and height 4":
SUMMARY: Find the hypotenuse using Pythagorean theorem
STEPS:
1. Formula: a^2 + b^2 = c^2
2. Substitute: 3^2 + 4^2 = c^2
3. Calculate: 9 + 16 = 25
4. Take square root: c = 5
ANSWER: The hypotenuse is 5 units"""

        else:
            # Word problem - step by step explanation
            prompt = f"""Solve this word problem step by step:

{query}

INSTRUCTIONS:
- First identify what we need to find
- Break down the problem into simple steps
- Explain each step in plain English
- Show any calculations clearly
- Use 3-6 steps
- No asterisks or special formatting
- Make it easy to read aloud

Format:
SUMMARY: [what the problem is asking]
STEPS:
1. [identify what we know]
2. [identify what we need to find]
3. [explain the method to solve]
4. [do the calculation if needed]
5. [state the result]
ANSWER: [clear final answer]"""
        
        print("📤 Sending request to Gemini...")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            print("❌ Gemini returned empty response")
            return None
        
        print("✓ Received Gemini response")
        text = response.text.strip()
        
        # Parse the response
        summary = ''
        steps = []
        answer = ''
        formulas = ''
        example = ''
        
        lines = text.split('\n')
        current_section = None
        current_step = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.upper().startswith('SUMMARY:'):
                summary = line.split(':', 1)[1].strip()
                current_section = 'summary'
            elif line.upper().startswith('FORMULAS:'):
                formulas = line.split(':', 1)[1].strip()
                current_section = 'formulas'
            elif line.upper().startswith('STEPS:'):
                current_section = 'steps'
            elif line.upper().startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()
                current_section = 'answer'
            elif line.upper().startswith('EXAMPLE:'):
                example = line.split(':', 1)[1].strip()
                current_section = 'example'
            elif current_section == 'steps':
                # Check if this is a numbered step
                if line[0].isdigit() or line.startswith('-') or line.startswith('•'):
                    # Save previous step if exists
                    if current_step:
                        steps.append(current_step)
                    # Extract step text, removing numbering
                    step_text = re.sub(r'^[\d\.\-•]\s*', '', line)
                    current_step = step_text if step_text else None
                elif current_step and line:
                    # This is a continuation of the current step
                    current_step += ' ' + line
            elif current_section == 'summary' and not summary:
                summary = line
            elif current_section == 'answer' and not answer:
                answer = line
            elif current_section == 'example' and not example:
                example = line
            elif current_section == 'formulas' and not formulas:
                formulas = line
        
        # Don't forget to add the last step
        if current_step:
            steps.append(current_step)
        
        # If parsing failed, use raw text
        if not steps:
            steps = [s.strip() for s in text.split('\n') if s.strip()]
        
        if not summary:
            summary = query
        
        # Build final answer text
        if not answer and example:
            answer = f"See example: {example}"
        elif not answer:
            answer = 'See steps above'
        
        return {
            'source': 'gemini_ai',
            'type': 'ai_solution',
            'shape': 'Solution',
            'intent': 'solve with AI',
            'confidence': 0.85,
            'summary': summary,
            'solution': answer,
            'formula': formulas if formulas else None,
            'example': example if example else None,
            'steps': steps,
            'raw_response': text,
        }
    
    except Exception as e:
        print(f"❌ Gemini API error: {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return None


def accessible_solver_response(query: str) -> Optional[Dict[str, object]]:
    """Use the CLI geometry solver logic to answer queries for the web UI."""
    if not (GEOMETRY_SOLVER_AVAILABLE and query):
        return None

    theory_topic = find_theory_topic(query)
    if theory_topic:
        payload = format_theory_response(theory_topic)
        title = payload.get('title')
        diagram_data = None
        diagram_id = payload.get('diagram_id')
        if diagram_id:
            try:
                diagram_data = generate_diagram(diagram_id)
            except Exception:
                diagram_data = None
        
        # Check if theory explanation is substantial
        steps = payload.get('steps', [])
        summary = payload.get('summary', '')
        has_good_content = (len(steps) >= 3 and len(summary) > 50)
        
        theory_result = {
            'source': 'geometry_solver',
            'type': 'theory',
            'shape': title,
            'topic': title,
            'summary': payload.get('summary'),
            'formula': payload.get('formula'),
            'steps': payload.get('steps'),
            'example': payload.get('example'),
            'intent': 'theory explanation',
            'confidence': 1.0,
            'diagram': diagram_data,
            'diagram_caption': payload.get('diagram_caption'),
            'chapter': payload.get('chapter'),
            'parameters': {},
            'calculations': [],
        }
        
        # If theory content is weak, enhance with Gemini
        if not has_good_content and GEMINI_AVAILABLE:
            return None  # Let caller try Gemini
        
        return theory_result

    parsed = InputParser.parse_full_query(query)
    normalized_shape = _normalize_shape_name(parsed.get('shape'))
    if not normalized_shape:
        return None

    spec = GEOMETRY_SHAPE_SPECS.get(normalized_shape)
    if not spec:
        return None

    numbers = parsed.get('numbers', [])
    if len(numbers) < spec['min_inputs']:
        example = spec.get('example', 'circle radius 5')
        needed = spec['min_inputs']
        raise ValueError(
            f"I need {needed} measurement{'s' if needed > 1 else ''} to solve a {normalized_shape}. Try: '{example}'."
        )

    shape_builder: Callable[[List[float]], object] = spec['builder']  # type: ignore[assignment]
    shape = shape_builder(numbers)

    properties = shape.get_properties()
    prop_filter = parsed.get('property')
    if prop_filter:
        prop_filter = prop_filter.replace(' ', '_')

    calculations = []
    for key, (value, steps) in properties.items():
        if prop_filter and key != prop_filter:
            continue
        calculations.append({
            'property': key.replace('_', ' ').title(),
            'value': value,
            'unit': 'units',
            'formula': '',
            'steps': steps,
        })

    if not calculations:
        requested = prop_filter.replace('_', ' ') if prop_filter else 'that property'
        raise ValueError(f"I couldn't find {requested} for your {normalized_shape}. Try a different measurement.")

    parameters = {}
    for idx, label in enumerate(spec.get('params', [])):
        if idx < len(numbers):
            parameters[label] = numbers[idx]

    intent = prop_filter.replace('_', ' ') if prop_filter else 'calculate'

    return {
        'source': 'geometry_solver',
        'shape': normalized_shape.title(),
        'confidence': 1.0,
        'intent': intent,
        'parameters': parameters,
        'calculations': calculations,
    }


def _should_prioritize_textbook(query: str) -> bool:
    if _looks_like_textbook_example(query):
        return True
    lowered = query.lower()
    trigger_phrases = (
        'prove that',
        'show that',
        'hence prove',
        'explain theorem',
        'exercise',
        'question number',
    )
    if any(phrase in lowered for phrase in trigger_phrases):
        return True

    contains_digit = bool(re.search(r'\d', query))
    conceptual_keywords = (
        'tangent',
        'secant',
        'chord',
        'bisect',
        'construction',
        'similar triangle',
        'similar triangles',
        'cyclic quadrilateral',
        'locus',
        'theorem',
    )
    if not contains_digit and any(keyword in lowered for keyword in conceptual_keywords):
        return True

    return False


def textbook_lookup_response(query: str) -> Optional[Dict[str, object]]:
    kb = get_textbook_kb()
    if not kb:
        return None
    lookup = kb.search(query)
    if not lookup or not lookup.get('match'):
        return None

    primary = lookup['match'][0]
    related = lookup.get('related', [])
    score = float(primary.get('score', 0.0))
    confidence = min(0.95, 0.35 + 0.1 * score)
    question_raw = primary.get('question_text', '')
    summary_text, solution_text = _split_textbook_solution(question_raw)

    return {
        'source': 'textbook',
        'type': 'textbook_reference',
        'shape': primary.get('chapter_title', 'Textbook context'),
        'intent': 'textbook lookup',
        'confidence': confidence,
        'chapter': primary.get('chapter_title'),
        'chapter_number': primary.get('chapter_number'),
        'exercise': primary.get('exercise'),
        'question_number': primary.get('question_number'),
        'question_type': primary.get('question_type'),
        'summary': summary_text or question_raw,
        'solution': solution_text,
        'raw_text': question_raw,
        'topics': primary.get('topics', []),
        'focus': primary.get('focus'),
        'steps': primary.get('insights', []),
        'related_questions': [item.get('question_text') for item in related],
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    """Process a geometry query and return results."""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a query'}), 400
    
    try:
        # Check if it's a word problem FIRST - route directly to Gemini
        if _is_word_problem(query):
            if GEMINI_AVAILABLE:
                print(f"🔍 Detected word problem, routing to Gemini: {query[:50]}...")
                gemini_result = solve_with_gemini(query)
                if gemini_result:
                    print("✓ Gemini solved word problem")
                    return jsonify(gemini_result)
                else:
                    print("❌ Gemini failed for word problem")
            else:
                print("⚠️ Word problem detected but Gemini not available")
        
        # First, try math expressions (trig, box, sqrt, etc.)
        math_result = parse_math_query(query)
        if math_result:
            return jsonify(math_result)
        
        inference_error = None
        accessible_attempted = False
        initial_textbook = None
        prioritize_textbook = _should_prioritize_textbook(query)

        if prioritize_textbook:
            initial_textbook = textbook_lookup_response(query)
            if initial_textbook:
                # Check if textbook solution is incomplete or missing
                solution_text = initial_textbook.get('solution', '')
                question_type = initial_textbook.get('question_type', '')
                
                # More aggressive quality check for proof questions
                is_proof = question_type == 'Proof' or any(word in query.lower() for word in ['prove', 'show that', 'demonstrate'])
                solution_quality_ok = solution_text and len(solution_text) > 100
                
                # For proof questions, require even better quality or use Gemini
                if is_proof and GEMINI_AVAILABLE:
                    if not solution_quality_ok or len(solution_text) < 200:
                        gemini_result = solve_with_gemini(query)
                        if gemini_result:
                            gemini_result['textbook_reference'] = {
                                'chapter': initial_textbook.get('chapter'),
                                'exercise': initial_textbook.get('exercise'),
                                'question_number': initial_textbook.get('question_number'),
                            }
                            return jsonify(gemini_result)
                
                # If textbook solution is poor for any question, try Gemini
                if not solution_quality_ok and GEMINI_AVAILABLE:
                    gemini_result = solve_with_gemini(query)
                    if gemini_result:
                        gemini_result['textbook_reference'] = {
                            'chapter': initial_textbook.get('chapter'),
                            'exercise': initial_textbook.get('exercise'),
                            'question_number': initial_textbook.get('question_number'),
                        }
                        return jsonify(gemini_result)
                
                return jsonify(initial_textbook)
            if GEOMETRY_SOLVER_AVAILABLE:
                try:
                    fallback = accessible_solver_response(query)
                    accessible_attempted = True
                    if fallback:
                        return jsonify(fallback)
                except ValueError as exc:
                    inference_error = str(exc)
        try:
            eng = get_engine()
            result = eng.process(query)
            if result.success:
                response = {
                    'shape': result.shape or 'Unknown',
                    'confidence': result.confidence,
                    'intent': result.intent,
                    'parameters': result.parameters,
                    'calculations': []
                }

                for calc in result.results:
                    response['calculations'].append({
                        'property': calc.get('property', ''),
                        'value': calc.get('value', 0),
                        'unit': calc.get('unit', 'units'),
                        'formula': calc.get('formula', ''),
                        'steps': calc.get('steps', []),
                    })

                # Check if we should use Gemini for better solution
                is_word_prob = _is_word_problem(query)
                low_confidence = result.confidence < 0.75
                
                if (is_word_prob or low_confidence) and GEMINI_AVAILABLE:
                    gemini_result = solve_with_gemini(query)
                    if gemini_result:
                        return jsonify(gemini_result)
                
                return jsonify(response)

            inference_error = result.errors[0] if result.errors else 'Could not understand the query. Try "area of circle with radius 5".'
        except Exception as exc:
            inference_error = str(exc)

        if GEOMETRY_SOLVER_AVAILABLE and not accessible_attempted:
            try:
                fallback = accessible_solver_response(query)
                if fallback:
                    # Don't add inference error as warning for successful geometry calculations
                    return jsonify(fallback)
                # If accessible_solver returned None (theory was weak), try Gemini
                elif GEMINI_AVAILABLE:
                    gemini_result = solve_with_gemini(query)
                    if gemini_result:
                        return jsonify(gemini_result)
            except ValueError as exc:
                inference_error = str(exc)

        textbook_payload = initial_textbook or textbook_lookup_response(query)
        if textbook_payload:
            # Check if we should enhance with Gemini
            solution_text = textbook_payload.get('solution', '')
            question_type = textbook_payload.get('question_type', '')
            
            is_proof = question_type == 'Proof' or any(word in query.lower() for word in ['prove', 'show that', 'demonstrate'])
            solution_quality_ok = solution_text and len(solution_text) > 100
            
            # For proof questions or poor solutions, use Gemini
            if (is_proof or not solution_quality_ok) and GEMINI_AVAILABLE:
                gemini_result = solve_with_gemini(query)
                if gemini_result:
                    gemini_result['textbook_reference'] = {
                        'chapter': textbook_payload.get('chapter'),
                        'exercise': textbook_payload.get('exercise'),
                        'question_number': textbook_payload.get('question_number'),
                    }
                    return jsonify(gemini_result)
            
            if inference_error:
                textbook_payload['warning'] = inference_error
            return jsonify(textbook_payload)

        # Last resort: try Gemini for any remaining queries
        if GEMINI_AVAILABLE:
            gemini_result = solve_with_gemini(query)
            if gemini_result:
                return jsonify(gemini_result)

        error_message = inference_error or 'Could not understand that query. Please provide more details.'
        return jsonify({'error': error_message}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/examples')
def examples():
    """Return example queries."""
    examples = [
        "area of a circle with radius 5",
        "perimeter of a rectangle length 10 width 5",
        "volume of a sphere with radius 3",
        "hypotenuse of triangle with height 4 base 3",
        "explain pythagorean theorem",
        "sin 90 + cos 0"
    ]
    return jsonify(examples)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🎯 ML Geometry Solver - Web Interface")
    print("=" * 50)
    print("\n🌐 Open http://localhost:5000 in your browser\n")
    # Disable Flask's reloader so refreshes don't briefly drop the server when code updates
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
