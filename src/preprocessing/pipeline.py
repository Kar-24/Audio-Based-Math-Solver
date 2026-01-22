"""
Text Preprocessing Pipeline
Advanced NLP preprocessing for geometry command understanding.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import string


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""
    original: str
    normalized: str
    tokens: List[str]
    cleaned_tokens: List[str]
    numeric_tokens: List[Tuple[str, float]]
    features: Dict[str, Any]


class TextNormalizer:
    """Normalizes text for consistent processing."""
    
    # Unicode character mappings
    UNICODE_MAP = {
        '–': '-', '—': '-', '−': '-',  # Dashes
        ''': "'", ''': "'", '`': "'",   # Quotes
        '"': '"', '"': '"',
        '×': 'x', '✕': 'x', '✖': 'x', '*': 'x',  # Multiplication
        '÷': '/', '∕': '/',  # Division
        'π': 'pi',  # Pi
        '²': '^2', '³': '^3',  # Superscripts
        '½': '1/2', '¼': '1/4', '¾': '3/4',  # Fractions
        '°': ' degrees',  # Degree symbol
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        r'\br\b': 'radius',
        r'\bd\b': 'diameter',
        r'\bl\b': 'length',
        r'\bw\b': 'width',
        r'\bh\b': 'height',
        r'\bs\b': 'side',
        r'\bcm\b': 'centimeters',
        r'\bmm\b': 'millimeters',
        r'\bm\b': 'meters',
        r'\bkm\b': 'kilometers',
        r'\bin\b': 'inches',
        r'\bft\b': 'feet',
    }
    
    # Spelling corrections for common geometry terms
    SPELLING_CORRECTIONS = {
        'circl': 'circle',
        'cicle': 'circle',
        'rectagle': 'rectangle',
        'rectangel': 'rectangle',
        'trianlge': 'triangle',
        'triangel': 'triangle',
        'shpere': 'sphere',
        'spehere': 'sphere',
        'sylinder': 'cylinder',
        'cilinder': 'cylinder',
        'piramid': 'pyramid',
        'pyramide': 'pyramid',
        'diamter': 'diameter',
        'raduis': 'radius',
        'hieght': 'height',
        'lenght': 'length',
        'widht': 'width',
    }
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """Apply all normalization steps to text."""
        text = cls._normalize_unicode(text)
        text = cls._normalize_whitespace(text)
        text = cls._correct_spelling(text)
        text = text.lower()
        return text
    
    @classmethod
    def _normalize_unicode(cls, text: str) -> str:
        """Replace unicode characters with ASCII equivalents."""
        for unicode_char, replacement in cls.UNICODE_MAP.items():
            text = text.replace(unicode_char, replacement)
        return text
    
    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """Normalize whitespace."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @classmethod
    def _correct_spelling(cls, text: str) -> str:
        """Correct common misspellings."""
        words = text.lower().split()
        corrected = []
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip(string.punctuation)
            if clean_word in cls.SPELLING_CORRECTIONS:
                word = word.replace(clean_word, cls.SPELLING_CORRECTIONS[clean_word])
            corrected.append(word)
        return ' '.join(corrected)


class Tokenizer:
    """Advanced tokenizer for geometry text."""
    
    # Token patterns in priority order
    TOKEN_PATTERNS = [
        # Measurements with units (e.g., "5.5cm", "10 meters")
        (r'\d+\.?\d*\s*(?:centi|milli|kilo)?meters?|\d+\.?\d*\s*(?:cm|mm|km|m|in|ft|yd)', 'MEASUREMENT'),
        # Scientific notation
        (r'-?\d+\.?\d*[eE][+-]?\d+', 'SCIENTIFIC'),
        # Fractions
        (r'\d+\s*/\s*\d+', 'FRACTION'),
        # Decimal numbers
        (r'-?\d+\.\d+', 'DECIMAL'),
        # Integers
        (r'-?\d+', 'INTEGER'),
        # Words
        (r'[a-zA-Z]+', 'WORD'),
        # Operators
        (r'[+\-*/=^]', 'OPERATOR'),
        # Punctuation
        (r'[.,;:!?]', 'PUNCTUATION'),
    ]
    
    def __init__(self):
        """Compile token patterns."""
        self.patterns = [
            (re.compile(pattern, re.IGNORECASE), token_type)
            for pattern, token_type in self.TOKEN_PATTERNS
        ]
    
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize text into (token, type) pairs.
        
        Args:
            text: Input text
            
        Returns:
            List of (token_text, token_type) tuples
        """
        tokens = []
        pos = 0
        
        while pos < len(text):
            # Skip whitespace
            if text[pos].isspace():
                pos += 1
                continue
            
            # Try each pattern
            matched = False
            for pattern, token_type in self.patterns:
                match = pattern.match(text, pos)
                if match:
                    tokens.append((match.group(), token_type))
                    pos = match.end()
                    matched = True
                    break
            
            # Skip unrecognized characters
            if not matched:
                pos += 1
        
        return tokens
    
    def get_words(self, text: str) -> List[str]:
        """Extract only word tokens."""
        tokens = self.tokenize(text)
        return [token for token, ttype in tokens if ttype == 'WORD']
    
    def get_numbers(self, text: str) -> List[float]:
        """Extract numerical values."""
        tokens = self.tokenize(text)
        numbers = []
        
        for token, ttype in tokens:
            if ttype in ('DECIMAL', 'INTEGER', 'SCIENTIFIC'):
                try:
                    numbers.append(float(token))
                except ValueError:
                    pass
            elif ttype == 'FRACTION':
                try:
                    parts = token.replace(' ', '').split('/')
                    numbers.append(float(parts[0]) / float(parts[1]))
                except (ValueError, ZeroDivisionError):
                    pass
        
        return numbers


class FeatureExtractor:
    """Extract features from preprocessed text for ML models."""
    
    # Keyword categories
    SHAPE_KEYWORDS = {
        '2d': ['circle', 'square', 'rectangle', 'triangle', 'trapezoid', 
               'parallelogram', 'ellipse', 'polygon', 'rhombus'],
        '3d': ['sphere', 'cube', 'cylinder', 'cone', 'pyramid', 
               'prism', 'torus', 'box']
    }
    
    PROPERTY_KEYWORDS = {
        'area': ['area', 'surface', 'space', 'size'],
        'perimeter': ['perimeter', 'circumference', 'around', 'border', 'edge'],
        'volume': ['volume', 'capacity', 'space', 'hold', 'contain'],
        'diagonal': ['diagonal', 'corner'],
    }
    
    ACTION_KEYWORDS = {
        'calculate': ['calculate', 'compute', 'find', 'get', 'determine', 'what'],
        'help': ['help', 'assist', 'how', 'guide', 'explain'],
        'list': ['list', 'show', 'display', 'available'],
        'quit': ['quit', 'exit', 'bye', 'stop', 'end']
    }
    
    @classmethod
    def extract(cls, text: str, tokens: List[str]) -> Dict[str, Any]:
        """
        Extract features from text and tokens.
        
        Args:
            text: Normalized text
            tokens: Word tokens
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'text_length': len(text),
            'word_count': len(tokens),
            'has_numbers': bool(re.search(r'\d', text)),
            'number_count': len(re.findall(r'-?\d+\.?\d*', text)),
            'has_question': '?' in text,
            'has_units': cls._has_units(text),
            'shape_dimension': cls._get_shape_dimension(tokens),
            'detected_shape': cls._detect_shape(tokens),
            'detected_property': cls._detect_property(tokens),
            'detected_action': cls._detect_action(tokens),
            'keyword_vector': cls._create_keyword_vector(tokens),
        }
        
        return features
    
    @classmethod
    def _has_units(cls, text: str) -> bool:
        """Check if text contains measurement units."""
        unit_pattern = r'\b(cm|mm|m|km|in|ft|yd|meters?|inches?|feet|yards?)\b'
        return bool(re.search(unit_pattern, text, re.IGNORECASE))
    
    @classmethod
    def _get_shape_dimension(cls, tokens: List[str]) -> Optional[str]:
        """Determine if shape is 2D or 3D."""
        tokens_lower = [t.lower() for t in tokens]
        
        for token in tokens_lower:
            if token in cls.SHAPE_KEYWORDS['3d']:
                return '3d'
            if token in cls.SHAPE_KEYWORDS['2d']:
                return '2d'
        
        return None
    
    @classmethod
    def _detect_shape(cls, tokens: List[str]) -> Optional[str]:
        """Detect shape name from tokens."""
        tokens_lower = [t.lower() for t in tokens]
        all_shapes = cls.SHAPE_KEYWORDS['2d'] + cls.SHAPE_KEYWORDS['3d']
        
        for token in tokens_lower:
            if token in all_shapes:
                return token
        
        return None
    
    @classmethod
    def _detect_property(cls, tokens: List[str]) -> Optional[str]:
        """Detect requested property."""
        tokens_lower = [t.lower() for t in tokens]
        
        for prop, keywords in cls.PROPERTY_KEYWORDS.items():
            for token in tokens_lower:
                if token in keywords:
                    return prop
        
        return None
    
    @classmethod
    def _detect_action(cls, tokens: List[str]) -> Optional[str]:
        """Detect the action/intent."""
        tokens_lower = [t.lower() for t in tokens]
        
        for action, keywords in cls.ACTION_KEYWORDS.items():
            for token in tokens_lower:
                if token in keywords:
                    return action
        
        return 'calculate'  # Default action
    
    @classmethod
    def _create_keyword_vector(cls, tokens: List[str]) -> Dict[str, bool]:
        """Create a boolean vector of keyword presence."""
        tokens_set = set(t.lower() for t in tokens)
        
        all_keywords = (
            cls.SHAPE_KEYWORDS['2d'] + 
            cls.SHAPE_KEYWORDS['3d'] +
            [kw for kwlist in cls.PROPERTY_KEYWORDS.values() for kw in kwlist] +
            [kw for kwlist in cls.ACTION_KEYWORDS.values() for kw in kwlist]
        )
        
        return {kw: kw in tokens_set for kw in all_keywords}


class PreprocessingPipeline:
    """Complete preprocessing pipeline for geometry text."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.normalizer = TextNormalizer
        self.tokenizer = Tokenizer()
        self.feature_extractor = FeatureExtractor
    
    def process(self, text: str) -> PreprocessedText:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            PreprocessedText with all preprocessing results
        """
        # Normalize
        normalized = self.normalizer.normalize(text)
        
        # Tokenize
        all_tokens = self.tokenizer.tokenize(normalized)
        word_tokens = [t for t, ttype in all_tokens if ttype == 'WORD']
        
        # Extract numeric tokens
        numeric_tokens = []
        for token, ttype in all_tokens:
            if ttype in ('DECIMAL', 'INTEGER', 'SCIENTIFIC', 'FRACTION'):
                try:
                    if ttype == 'FRACTION':
                        parts = token.replace(' ', '').split('/')
                        value = float(parts[0]) / float(parts[1])
                    else:
                        value = float(token)
                    numeric_tokens.append((token, value))
                except (ValueError, ZeroDivisionError):
                    pass
        
        # Extract features
        features = self.feature_extractor.extract(normalized, word_tokens)
        
        return PreprocessedText(
            original=text,
            normalized=normalized,
            tokens=[t for t, _ in all_tokens],
            cleaned_tokens=word_tokens,
            numeric_tokens=numeric_tokens,
            features=features
        )
    
    def process_batch(self, texts: List[str]) -> List[PreprocessedText]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]


def main():
    """Test the preprocessing pipeline."""
    print("=" * 60)
    print("TEXT PREPROCESSING PIPELINE - Test Suite")
    print("=" * 60)
    
    pipeline = PreprocessingPipeline()
    
    test_texts = [
        "Calculate the area of a circle with radius 5 cm",
        "what is the volume of a shpere with r=3.14?",
        "RECTANGLE: length 10, width 5",
        "find the perimeter of triangle with sides 3, 4, and 5",
        "cylinder   height=10   radius=2.5",
        "Help me please!",
        "list all available shapes",
        "cone with base 4½ inches and height 6 inches",
    ]
    
    for text in test_texts:
        print(f"\n{'─' * 50}")
        print(f"Original: '{text}'")
        
        result = pipeline.process(text)
        
        print(f"Normalized: '{result.normalized}'")
        print(f"Word tokens: {result.cleaned_tokens}")
        print(f"Numeric tokens: {result.numeric_tokens}")
        print(f"Features:")
        print(f"  - Shape: {result.features.get('detected_shape')}")
        print(f"  - Property: {result.features.get('detected_property')}")
        print(f"  - Action: {result.features.get('detected_action')}")
        print(f"  - Dimension: {result.features.get('shape_dimension')}")
    
    print("\n" + "=" * 60)
    print("Preprocessing tests complete!")


if __name__ == "__main__":
    main()