"""
ML-Powered Geometry Solver Package
Combines machine learning intent classification with regex entity extraction
for natural language understanding of geometry commands.
"""

__version__ = "2.0.0"
__author__ = "Geometry Solver Team"

from src.inference import GeometryInferenceEngine, InferenceResult
from src.rules.evaluator import HybridDecisionEngine, DecisionStrategy
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.entity_extractor import RegexEntityExtractor

__all__ = [
    'GeometryInferenceEngine',
    'InferenceResult',
    'HybridDecisionEngine',
    'DecisionStrategy',
    'PreprocessingPipeline',
    'RegexEntityExtractor',
]