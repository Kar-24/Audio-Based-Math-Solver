"""
ML-Powered Geometry Solver
Main entry point for training and running the geometry understanding system.
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def train_models(samples_per_intent: int = 800, model_type: str = 'mlp'):
    """Train the ML intent classifier."""
    print("\n" + "=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)
    
    from src.ml.model import TrainingDataGenerator, IntentClassifier
    
    # Generate training data
    print("\n📊 Generating synthetic training data...")
    df = TrainingDataGenerator.generate_dataset(samples_per_intent=samples_per_intent)
    print(f"   Generated {len(df)} training samples")
    
    # Save training data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/training_data.csv', index=False)
    print("   Saved training data")
    
    # Train models
    print("\n🧠 Training intent classifier...")
    classifier = IntentClassifier(model_type=model_type)
    metrics = classifier.train(df)
    
    # Save models
    os.makedirs('models/baseline', exist_ok=True)
    classifier.save('models/baseline')
    
    print("\n✓ Training complete!")
    print(f"  Intent Accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"  Shape Accuracy: {metrics['shape_accuracy']:.4f}")
    
    return metrics


def test_regex_extractor():
    """Test the regex entity extractor."""
    print("\n" + "=" * 60)
    print("TESTING REGEX ENTITY EXTRACTOR")
    print("=" * 60)
    
    from src.preprocessing.entity_extractor import RegexEntityExtractor
    
    extractor = RegexEntityExtractor()
    
    test_cases = [
        "circle radius 5",
        "rectangle length 10 width 5",
        "sphere with radius 3.14 cm",
        "cone base 4 height 6",
        "triangle sides 3, 4, 5",
    ]
    
    for text in test_cases:
        result = extractor.extract(text)
        print(f"\n  '{text}'")
        print(f"    Shape: {result.shape}")
        print(f"    Params: {[(n, v.value) for n, v in result.parameters.items()]}")


def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    from src.preprocessing.pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    
    test_texts = [
        "Calculate the AREA of a circl with radius 5",
        "what is the volume of a shpere?",
        "RECTANGLE   length = 10,  width = 5",
    ]
    
    for text in test_texts:
        result = pipeline.process(text)
        print(f"\n  Original: '{text}'")
        print(f"  Normalized: '{result.normalized}'")
        print(f"  Shape: {result.features.get('detected_shape')}")


def test_decision_engine():
    """Test the hybrid decision engine."""
    print("\n" + "=" * 60)
    print("TESTING HYBRID DECISION ENGINE")
    print("=" * 60)
    
    from src.rules.evaluator import HybridDecisionEngine, DecisionStrategy
    
    engine = HybridDecisionEngine(strategy=DecisionStrategy.REGEX_PRIORITY)
    
    test_queries = [
        "calculate the area of a circle with radius 5",
        "sphere radius 3",
        "rectangle length 10 width 5",
        "cylinder radius 2 height 8",
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"Query: '{query}'")
        result = engine.process(query)
        print(engine.explain_result(result))


def run_interactive(use_tts: bool = False, strategy: str = 'ensemble'):
    """Run the interactive geometry solver."""
    from src.inference import GeometryInferenceEngine
    
    engine = GeometryInferenceEngine(
        model_path='models/baseline',
        strategy=strategy,
        use_tts=use_tts
    )
    
    engine.interactive_session()


def demo():
    """Run a quick demo of the system."""
    print("\n" + "=" * 60)
    print("🎯 ML-POWERED GEOMETRY SOLVER DEMO")
    print("=" * 60)
    
    from src.inference import GeometryInferenceEngine
    
    engine = GeometryInferenceEngine(
        model_path='models/baseline',
        strategy='regex_priority'
    )
    
    demo_queries = [
        "calculate the area of a circle with radius 5",
        "what is the volume of a sphere with radius 3",
        "rectangle length 10 width 5",
        "cone with radius 4 and height 6",
        "cube side 7",
    ]
    
    for query in demo_queries:
        print(f"\n{'=' * 50}")
        result = engine.process(query, verbose=True)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("Run with --interactive for full interactive mode")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ML-Powered Geometry Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train              Train ML models
  python main.py --interactive        Run interactive solver
  python main.py --demo               Quick demonstration
  python main.py --test               Run all tests
  python main.py -i --tts             Interactive with voice
        """
    )
    
    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train the ML models'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run a quick demo'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run all component tests'
    )
    
    parser.add_argument(
        '--tts',
        action='store_true',
        help='Enable text-to-speech'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='ensemble',
        choices=['ensemble', 'ml_priority', 'regex_priority', 'cascading'],
        help='Decision strategy'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=800,
        help='Samples per intent for training'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='mlp',
        choices=['mlp', 'rf', 'gb'],
        help='ML model type (mlp=neural network, rf=random forest, gb=gradient boosting)'
    )
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    if args.train:
        train_models(samples_per_intent=args.samples, model_type=args.model_type)
    elif args.test:
        test_regex_extractor()
        test_preprocessing()
        test_decision_engine()
    elif args.demo:
        demo()
    elif args.interactive:
        run_interactive(use_tts=args.tts, strategy=args.strategy)
    else:
        # Default: show help and run demo
        parser.print_help()
        print("\n")
        demo()


if __name__ == "__main__":
    main()