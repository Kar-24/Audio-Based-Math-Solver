"""
Inference API for Geometry Solver
Provides a clean interface for ML-powered geometry understanding.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.entity_extractor import RegexEntityExtractor
from src.rules.evaluator import HybridDecisionEngine, DecisionStrategy, DecisionResult


@dataclass
class InferenceResult:
    """Result from the inference API."""
    success: bool
    query: str
    shape: Optional[str]
    intent: str
    confidence: float
    parameters: Dict[str, float]
    results: List[Dict[str, Any]]
    explanation: str
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class GeometryInferenceEngine:
    """
    High-level inference API for geometry calculations.
    Combines ML intent classification, regex entity extraction,
    and rule-based calculation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        strategy: str = "ensemble",
        use_tts: bool = False
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained ML models
            strategy: Decision strategy ('ensemble', 'ml_priority', 'regex_priority', 'cascading')
            use_tts: Whether to enable text-to-speech
        """
        self.model_path = model_path or "models/baseline"
        self.use_tts = use_tts
        self.audio = None
        
        # Map strategy string to enum
        strategy_map = {
            'ensemble': DecisionStrategy.ENSEMBLE,
            'ml_priority': DecisionStrategy.ML_PRIORITY,
            'regex_priority': DecisionStrategy.REGEX_PRIORITY,
            'cascading': DecisionStrategy.CASCADING
        }
        self.strategy = strategy_map.get(strategy, DecisionStrategy.ENSEMBLE)
        
        # Initialize components
        self.preprocessor = PreprocessingPipeline()
        self.decision_engine = HybridDecisionEngine(
            model_path=self.model_path,
            strategy=self.strategy
        )
        
        if use_tts:
            self._init_audio()
        
        print("✓ Geometry Inference Engine initialized")
    
    def _init_audio(self):
        """Initialize text-to-speech if available."""
        try:
            import pyttsx3
            self.audio = pyttsx3.init()
            self.audio.setProperty('rate', 150)
            print("✓ Text-to-speech enabled")
        except ImportError:
            print("⚠ pyttsx3 not available, TTS disabled")
        except Exception as e:
            print(f"⚠ TTS initialization failed: {e}")
    
    def _extract_requested_property(self, query: str, decision: DecisionResult) -> Optional[str]:
        """Extract which specific property the user is asking for."""
        query_lower = query.lower()
        
        # Check for explicit property mentions in the query
        property_keywords = {
            'volume': ['volume', 'cubic', 'capacity'],
            'area': ['area', 'square footage', 'surface area'],
            'perimeter': ['perimeter', 'circumference', 'around', 'edge length'],
            'diagonal': ['diagonal'],
            'radius': ['radius'],
            'diameter': ['diameter'],
            'height': ['height', 'tall', 'altitude'],
            'slant_height': ['slant height', 'slant'],
            'interior_angle': ['angle', 'interior angle'],
            'apothem': ['apothem'],
        }
        
        for prop, keywords in property_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return prop
        
        # Also check the decision intent
        if decision.property_requested:
            return decision.property_requested
        
        # If intent contains property hints
        intent = decision.intent.lower() if decision.intent else ''
        if 'volume' in intent:
            return 'volume'
        if 'area' in intent:
            return 'area'
        if 'perimeter' in intent or 'circumference' in intent:
            return 'perimeter'
        
        return None
    
    def _matches_property(self, calc_property: str, requested: str) -> bool:
        """Check if a calculation property matches the requested property."""
        calc_lower = calc_property.lower()
        requested_lower = requested.lower()
        
        # Direct match
        if requested_lower in calc_lower or calc_lower in requested_lower:
            return True
        
        # Handle synonyms
        synonyms = {
            'circumference': ['perimeter'],
            'perimeter': ['circumference'],
            'surface_area': ['area'],
            'total_surface_area': ['area', 'surface_area'],
            'lateral_surface_area': ['area', 'surface_area'],
        }
        
        if calc_lower in synonyms:
            if requested_lower in synonyms[calc_lower]:
                return True
        
        return False
    
    def speak(self, text: str):
        """Speak text if TTS is enabled."""
        if self.audio:
            try:
                self.audio.say(text)
                self.audio.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def process(self, query: str, verbose: bool = False) -> InferenceResult:
        """
        Process a geometry query.
        
        Args:
            query: Natural language geometry query
            verbose: Whether to print detailed output
            
        Returns:
            InferenceResult with calculations and explanations
        """
        # Preprocess the query
        preprocessed = self.preprocessor.process(query)
        
        if verbose:
            print(f"\n📝 Query: '{query}'")
            print(f"   Normalized: '{preprocessed.normalized}'")
            print(f"   Detected: shape={preprocessed.features.get('detected_shape')}, "
                  f"property={preprocessed.features.get('detected_property')}")
        
        # Run through decision engine
        decision = self.decision_engine.process(preprocessed.normalized)
        
        # Determine which property the user is asking for
        requested_property = self._extract_requested_property(query, decision)
        
        # Build results list - filter to only requested property if specified
        results = []
        for calc in decision.calculations:
            # If user asked for a specific property, only include that one
            if requested_property:
                if self._matches_property(calc.property_name, requested_property):
                    results.append({
                        'property': calc.property_name,
                        'value': calc.value,
                        'unit': calc.unit,
                        'formula': calc.formula,
                        'steps': calc.explanation_steps
                    })
            else:
                # No specific property requested, include all
                results.append({
                    'property': calc.property_name,
                    'value': calc.value,
                    'unit': calc.unit,
                    'formula': calc.formula,
                    'steps': calc.explanation_steps
                })
        
        # If we filtered but got no results, include all (fallback)
        if requested_property and not results:
            for calc in decision.calculations:
                results.append({
                    'property': calc.property_name,
                    'value': calc.value,
                    'unit': calc.unit,
                    'formula': calc.formula,
                    'steps': calc.explanation_steps
                })
        
        # Generate explanation
        explanation = self.decision_engine.explain_result(decision)
        
        # Create result
        result = InferenceResult(
            success=decision.success,
            query=query,
            shape=decision.shape,
            intent=decision.intent,
            confidence=decision.confidence,
            parameters=decision.parameters,
            results=results,
            explanation=explanation,
            errors=decision.errors,
            warnings=decision.warnings
        )
        
        if verbose:
            print(explanation)
        
        # Speak the results if TTS is enabled
        if self.use_tts and decision.success:
            self._speak_results(decision)
        
        return result
    
    def _speak_results(self, decision: DecisionResult):
        """Speak the calculation results."""
        if not decision.calculations:
            return
        
        intro = f"Calculating properties of {decision.shape.replace('_', ' ')}"
        self.speak(intro)
        
        for calc in decision.calculations:
            text = f"The {calc.property_name.replace('_', ' ')} is {calc.value:.2f} {calc.unit}"
            self.speak(text)
    
    def interactive_session(self):
        """Run an interactive query session."""
        print("\n" + "=" * 60)
        print("   GEOMETRY SOLVER - Interactive Mode")
        print("   ML-Powered Natural Language Understanding")
        print("=" * 60)
        print("\nCommands: 'help', 'shapes', 'quit'")
        print("Example: 'calculate the area of a circle with radius 5'\n")
        
        if self.use_tts:
            self.speak("Geometry solver ready. How can I help you?")
        
        while True:
            try:
                query = input("\n🔢 Enter query: ").strip()
                
                if not query:
                    continue
                
                query_lower = query.lower()
                
                if query_lower in ['quit', 'exit', 'q', 'bye']:
                    print("\n👋 Goodbye!")
                    if self.use_tts:
                        self.speak("Goodbye!")
                    break
                
                if query_lower in ['help', 'h', '?']:
                    self._show_help()
                    continue
                
                if query_lower in ['shapes', 'list', 'list shapes']:
                    self._list_shapes()
                    continue
                
                # Process the query
                result = self.process(query, verbose=True)
                
                if not result.success:
                    print("\n❌ Could not process query.")
                    if result.errors:
                        for error in result.errors:
                            print(f"   Error: {error}")
                    print("   Try 'help' for usage examples.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║                    GEOMETRY SOLVER HELP                    ║
╚════════════════════════════════════════════════════════════╝

EXAMPLE QUERIES:
  • "calculate the area of a circle with radius 5"
  • "sphere radius 3"
  • "rectangle length 10 width 5"
  • "what is the volume of a cylinder with radius 2 and height 8"
  • "cone base 4 height 6"
  • "triangle sides 3, 4, 5"

SUPPORTED SHAPES:
  2D: circle, square, rectangle, triangle, trapezoid, ellipse
  3D: sphere, cube, cylinder, cone, pyramid, rectangular prism

COMMANDS:
  help     - Show this help message
  shapes   - List all available shapes
  quit     - Exit the program

TIPS:
  • You can use natural language or simple parameter format
  • Units are optional but supported (cm, m, inches, etc.)
  • Parameters can be in any order
"""
        print(help_text)
    
    def _list_shapes(self):
        """List available shapes."""
        shapes_2d = ['circle', 'square', 'rectangle', 'triangle', 
                     'trapezoid', 'parallelogram', 'ellipse']
        shapes_3d = ['sphere', 'cube', 'cylinder', 'cone', 
                     'pyramid', 'rectangular prism']
        
        print("\n📐 2D SHAPES:")
        for shape in shapes_2d:
            print(f"   • {shape}")
        
        print("\n📦 3D SHAPES:")
        for shape in shapes_3d:
            print(f"   • {shape}")


class BatchProcessor:
    """Process multiple queries in batch mode."""
    
    def __init__(self, engine: GeometryInferenceEngine):
        self.engine = engine
    
    def process_batch(self, queries: List[str]) -> List[InferenceResult]:
        """Process a list of queries."""
        return [self.engine.process(q) for q in queries]
    
    def process_file(self, input_file: str, output_file: str):
        """Process queries from a file and save results."""
        with open(input_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = self.process_batch(queries)
        
        output = {
            'total_queries': len(queries),
            'successful': sum(1 for r in results if r.success),
            'failed': sum(1 for r in results if not r.success),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Processed {len(queries)} queries")
        print(f"  Successful: {output['successful']}")
        print(f"  Failed: {output['failed']}")
        print(f"  Output saved to: {output_file}")


def main():
    """Main entry point for the inference API."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Geometry Solver - ML-Powered Natural Language Understanding'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='ensemble',
        choices=['ensemble', 'ml_priority', 'regex_priority', 'cascading'],
        help='Decision strategy for combining ML and regex'
    )
    
    parser.add_argument(
        '--tts',
        action='store_true',
        help='Enable text-to-speech'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/baseline',
        help='Path to trained ML models'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = GeometryInferenceEngine(
        model_path=args.model_path,
        strategy=args.strategy,
        use_tts=args.tts
    )
    
    if args.interactive:
        engine.interactive_session()
    elif args.query:
        result = engine.process(args.query, verbose=not args.json)
        if args.json:
            print(result.to_json())
    else:
        # Default to interactive mode
        engine.interactive_session()


if __name__ == "__main__":
    main()