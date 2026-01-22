"""
ML Intent Classifier for Geometry Solver
Uses a neural network to classify user intents and identify shapes from natural language.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


@dataclass
class IntentPrediction:
    """Represents a predicted intent with confidence scores."""
    intent: str
    shape: Optional[str]
    property_type: Optional[str]
    confidence: float
    all_probabilities: Dict[str, float]


class IntentLabels:
    """Intent and shape label definitions."""
    
    # Intent types
    INTENTS = [
        'calculate_area',
        'calculate_perimeter', 
        'calculate_volume',
        'calculate_surface_area',
        'calculate_all',
        'get_help',
        'list_shapes',
        'quit',
        'unknown'
    ]
    
    # Shape types
    SHAPES_2D = ['circle', 'square', 'rectangle', 'triangle', 'right_triangle',
                 'trapezoid', 'parallelogram', 'ellipse', 'polygon']
    
    SHAPES_3D = ['sphere', 'cube', 'cylinder', 'cone', 'rectangular_prism', 
                 'pyramid', 'torus', 'prism']
    
    ALL_SHAPES = SHAPES_2D + SHAPES_3D
    
    # Property types
    PROPERTIES = ['area', 'perimeter', 'circumference', 'volume', 
                  'surface_area', 'diagonal', 'radius', 'diameter', 'all']


class TrainingDataGenerator:
    """Generates synthetic training data for intent classification."""
    
    # Templates for generating training sentences
    TEMPLATES = {
        'calculate_area': [
            "calculate the area of a {shape}",
            "what is the area of a {shape}",
            "find area of {shape}",
            "area of a {shape} with {param}",
            "how big is a {shape}",
            "compute area for {shape}",
            "get the area of {shape}",
            "i need the area of a {shape}",
            "tell me the area of {shape}",
            "{shape} area please",
            "area {shape}",
            "what's the area of this {shape}",
            "calculate {shape} area",
            "find the area",
            "how much space does a {shape} cover"
        ],
        'calculate_perimeter': [
            "calculate the perimeter of a {shape}",
            "what is the perimeter of a {shape}",
            "find perimeter of {shape}",
            "circumference of a {shape}",
            "how long is the border of {shape}",
            "perimeter of {shape}",
            "get the circumference",
            "what is the circumference of a {shape}",
            "length around {shape}",
            "{shape} perimeter",
            "find the perimeter",
            "calculate circumference of {shape}"
        ],
        'calculate_volume': [
            "calculate the volume of a {shape}",
            "what is the volume of a {shape}",
            "find volume of {shape}",
            "how much can a {shape} hold",
            "volume of a {shape}",
            "capacity of {shape}",
            "compute volume for {shape}",
            "{shape} volume",
            "what's the volume",
            "how much space inside {shape}",
            "calculate {shape} volume",
            "find the volume of the {shape}"
        ],
        'calculate_surface_area': [
            "calculate the surface area of a {shape}",
            "what is the surface area of a {shape}",
            "find surface area of {shape}",
            "outer area of {shape}",
            "surface of a {shape}",
            "{shape} surface area",
            "total surface area of {shape}",
            "outside area of {shape}",
            "skin area of {shape}"
        ],
        'calculate_all': [
            "calculate all properties of a {shape}",
            "tell me everything about a {shape}",
            "all measurements for {shape}",
            "{shape}",
            "analyze a {shape}",
            "complete calculation for {shape}",
            "full analysis of {shape}",
            "everything about {shape}",
            "all {shape} properties"
        ],
        'get_help': [
            "help",
            "help me",
            "how do i use this",
            "what can you do",
            "instructions",
            "guide",
            "tutorial",
            "how does this work",
            "explain how to use",
            "what are the commands",
            "show me how",
            "i need help",
            "assist me",
            "what should i do"
        ],
        'list_shapes': [
            "list shapes",
            "what shapes are available",
            "show all shapes",
            "available shapes",
            "which shapes can i use",
            "shapes list",
            "what shapes do you know",
            "show me the shapes",
            "list all shapes",
            "what can i calculate"
        ],
        'quit': [
            "quit",
            "exit",
            "bye",
            "goodbye",
            "stop",
            "end",
            "close",
            "finish",
            "done",
            "leave",
            "terminate",
            "shut down"
        ]
    }
    
    PARAMETERS = [
        "radius {num}", "side {num}", "length {num}", "width {num}",
        "height {num}", "base {num}", "r = {num}", "s = {num}",
        "radius of {num}", "side length {num}", "{num} units",
        "{num} cm", "{num} meters", "{num} inches"
    ]
    
    @classmethod
    def generate_dataset(cls, samples_per_intent: int = 500) -> pd.DataFrame:
        """Generate a synthetic training dataset."""
        data = []
        
        for intent, templates in cls.TEMPLATES.items():
            for _ in range(samples_per_intent):
                template = np.random.choice(templates)
                
                # Fill in shape placeholder
                if '{shape}' in template:
                    if intent in ['calculate_volume', 'calculate_surface_area']:
                        shape = np.random.choice(IntentLabels.SHAPES_3D)
                    elif intent in ['calculate_perimeter']:
                        shape = np.random.choice(IntentLabels.SHAPES_2D)
                    else:
                        shape = np.random.choice(IntentLabels.ALL_SHAPES)
                    template = template.replace('{shape}', shape.replace('_', ' '))
                else:
                    shape = None
                
                # Fill in parameter placeholder
                if '{param}' in template:
                    param = np.random.choice(cls.PARAMETERS)
                    num = np.random.uniform(0.5, 100)
                    param = param.replace('{num}', f"{num:.1f}")
                    template = template.replace('{param}', param)
                
                # Add some noise/variations
                template = cls._add_variations(template)
                
                data.append({
                    'text': template,
                    'intent': intent,
                    'shape': shape
                })
        
        return pd.DataFrame(data)
    
    @classmethod
    def _add_variations(cls, text: str) -> str:
        """Add random variations to training text."""
        variations = [
            lambda t: t,  # Keep original
            lambda t: t.upper(),  # All caps
            lambda t: t.capitalize(),  # Capitalize first
            lambda t: "please " + t,  # Add please
            lambda t: t + " please",
            lambda t: "can you " + t,
            lambda t: "i want to " + t,
            lambda t: t + "?",
            lambda t: t.replace(" ", "  "),  # Double spaces
        ]
        return np.random.choice(variations)(text)


class IntentClassifier:
    """Neural network-based intent classifier for geometry commands."""
    
    def __init__(self, model_type: str = 'mlp'):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'mlp' for neural network, 'rf' for random forest, 'gb' for gradient boosting
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            lowercase=True,
            stop_words=None  # Keep all words for geometry terms
        )
        self.intent_encoder = LabelEncoder()
        self.shape_encoder = LabelEncoder()
        self.intent_model = None
        self.shape_model = None
        self._build_models()
    
    def _build_models(self):
        """Build the ML models."""
        if self.model_type == 'mlp':
            self.intent_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            self.shape_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        elif self.model_type == 'rf':
            self.intent_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.shape_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            self.intent_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.shape_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the intent and shape classifiers.
        
        Args:
            df: DataFrame with 'text', 'intent', and 'shape' columns
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare text features
        X_text = self.vectorizer.fit_transform(df['text'].str.lower())
        
        # Encode intents
        y_intent = self.intent_encoder.fit_transform(df['intent'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_intent, test_size=0.2, random_state=42, stratify=y_intent
        )
        
        # Train intent classifier
        print("Training intent classifier...")
        self.intent_model.fit(X_train, y_train)
        intent_pred = self.intent_model.predict(X_test)
        intent_accuracy = accuracy_score(y_test, intent_pred)
        
        # Train shape classifier (only on samples with shapes)
        shape_mask = df['shape'].notna()
        if shape_mask.sum() > 0:
            X_shape = self.vectorizer.transform(df.loc[shape_mask, 'text'].str.lower())
            y_shape = self.shape_encoder.fit_transform(df.loc[shape_mask, 'shape'])
            
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                X_shape, y_shape, test_size=0.2, random_state=42
            )
            
            print("Training shape classifier...")
            self.shape_model.fit(X_train_s, y_train_s)
            shape_pred = self.shape_model.predict(X_test_s)
            shape_accuracy = accuracy_score(y_test_s, shape_pred)
        else:
            shape_accuracy = 0.0
        
        metrics = {
            'intent_accuracy': intent_accuracy,
            'shape_accuracy': shape_accuracy,
            'intent_report': classification_report(
                y_test, intent_pred,
                target_names=self.intent_encoder.classes_
            ),
            'num_samples': len(df),
            'model_type': self.model_type
        }
        
        print(f"\n✓ Intent Accuracy: {intent_accuracy:.4f}")
        print(f"✓ Shape Accuracy: {shape_accuracy:.4f}")
        
        return metrics
    
    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent and shape from text.
        
        Args:
            text: User input text
            
        Returns:
            IntentPrediction with intent, shape, and confidence
        """
        # Vectorize input
        X = self.vectorizer.transform([text.lower()])
        
        # Predict intent
        intent_proba = self.intent_model.predict_proba(X)[0]
        intent_idx = np.argmax(intent_proba)
        intent = self.intent_encoder.inverse_transform([intent_idx])[0]
        confidence = intent_proba[intent_idx]
        
        # Create probability dictionary
        all_probs = {
            self.intent_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(intent_proba)
        }
        
        # Predict shape if applicable
        shape = None
        if intent not in ['get_help', 'list_shapes', 'quit', 'unknown']:
            try:
                shape_proba = self.shape_model.predict_proba(X)[0]
                shape_idx = np.argmax(shape_proba)
                if shape_proba[shape_idx] > 0.3:  # Confidence threshold
                    shape = self.shape_encoder.inverse_transform([shape_idx])[0]
            except:
                pass
        
        # Determine property type from intent
        property_map = {
            'calculate_area': 'area',
            'calculate_perimeter': 'perimeter',
            'calculate_volume': 'volume',
            'calculate_surface_area': 'surface_area',
            'calculate_all': 'all'
        }
        property_type = property_map.get(intent)
        
        return IntentPrediction(
            intent=intent,
            shape=shape,
            property_type=property_type,
            confidence=confidence,
            all_probabilities=all_probs
        )
    
    def save(self, directory: str):
        """Save the trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        joblib.dump(self.vectorizer, os.path.join(directory, 'vectorizer.joblib'))
        joblib.dump(self.intent_model, os.path.join(directory, 'intent_model.joblib'))
        joblib.dump(self.shape_model, os.path.join(directory, 'shape_model.joblib'))
        joblib.dump(self.intent_encoder, os.path.join(directory, 'intent_encoder.joblib'))
        joblib.dump(self.shape_encoder, os.path.join(directory, 'shape_encoder.joblib'))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'intents': list(self.intent_encoder.classes_),
            'shapes': list(self.shape_encoder.classes_)
        }
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Models saved to {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'IntentClassifier':
        """Load trained models from disk."""
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        classifier = cls(model_type=metadata['model_type'])
        classifier.vectorizer = joblib.load(os.path.join(directory, 'vectorizer.joblib'))
        classifier.intent_model = joblib.load(os.path.join(directory, 'intent_model.joblib'))
        classifier.shape_model = joblib.load(os.path.join(directory, 'shape_model.joblib'))
        classifier.intent_encoder = joblib.load(os.path.join(directory, 'intent_encoder.joblib'))
        classifier.shape_encoder = joblib.load(os.path.join(directory, 'shape_encoder.joblib'))
        
        print(f"✓ Models loaded from {directory}")
        return classifier


def main():
    """Train and evaluate the intent classifier."""
    print("=" * 60)
    print("GEOMETRY INTENT CLASSIFIER - Training Pipeline")
    print("=" * 60)
    
    # Generate training data
    print("\n📊 Generating synthetic training data...")
    df = TrainingDataGenerator.generate_dataset(samples_per_intent=800)
    print(f"   Generated {len(df)} training samples")
    print(f"   Intents: {df['intent'].nunique()}")
    print(f"   Shapes: {df['shape'].nunique()}")
    
    # Save training data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/training_data.csv', index=False)
    print("   Saved to data/processed/training_data.csv")
    
    # Train models
    print("\n🧠 Training ML models...")
    classifier = IntentClassifier(model_type='mlp')
    metrics = classifier.train(df)
    
    # Save models
    print("\n💾 Saving trained models...")
    classifier.save('models/baseline')
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_inputs = [
        "calculate the area of a circle with radius 5",
        "what is the volume of a sphere",
        "help me please",
        "list all shapes",
        "perimeter of rectangle",
        "quit",
        "find the surface area of a cylinder"
    ]
    
    for text in test_inputs:
        pred = classifier.predict(text)
        print(f"\n  Input: '{text}'")
        print(f"  → Intent: {pred.intent} (confidence: {pred.confidence:.2f})")
        if pred.shape:
            print(f"  → Shape: {pred.shape}")
        if pred.property_type:
            print(f"  → Property: {pred.property_type}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()