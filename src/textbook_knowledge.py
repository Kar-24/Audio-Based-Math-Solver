"""Utility to index and search the CBSE Class 10 mathematics textbook."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - dependency is optional at import time
    PdfReader = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, 'textbook_index.json')
PDF_CANDIDATES = [
    os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'math_10th_tb.pdf')),
    os.path.abspath(os.path.join(BASE_DIR, '..', 'math_10th_tb.pdf')),
    os.path.abspath(os.path.join(BASE_DIR, 'math_10th_tb.pdf')),
]

CHAPTER_TOPICS: Dict[str, List[str]] = {
    'REAL NUMBERS': ['euclid division', 'irrational numbers', 'hcf and lcm'],
    'POLYNOMIALS': ['zeroes of polynomials', 'graph behaviour'],
    'PAIR OF LINEAR EQUATIONS IN TWO VARIABLES': ['simultaneous equations', 'elimination and substitution'],
    'QUADRATIC EQUATIONS': ['factorisation method', 'completing square', 'discriminant analysis'],
    'ARITHMETIC PROGRESSIONS': ['nth term', 'sum of n terms'],
    'TRIANGLES': ['similar triangles', 'pythagoras variants'],
    'COORDINATE GEOMETRY': ['distance formula', 'section formula', 'area of triangle'],
    'INTRODUCTION TO TRIGONOMETRY': ['basic trig ratios', 'identities'],
    'SOME APPLICATIONS OF TRIGONOMETRY': ['angle of elevation', 'angle of depression'],
    'CIRCLES': ['tangent properties', 'secant lengths'],
    'CONSTRUCTIONS': ['bisectors', 'tangents from external points'],
    'AREAS RELATED TO CIRCLES': ['sector area', 'segment area'],
    'SURFACE AREAS AND VOLUMES': ['frustum', 'composite solids'],
    'STATISTICS': ['mean median mode', 'ogive interpretation'],
    'PROBABILITY': ['classical probability', 'independent events'],
}

CHAPTER_INFO: Dict[str, str] = {
    '1': 'Real Numbers',
    '2': 'Polynomials',
    '3': 'Pair of Linear Equations in Two Variables',
    '4': 'Quadratic Equations',
    '5': 'Arithmetic Progressions',
    '6': 'Triangles',
    '7': 'Coordinate Geometry',
    '8': 'Introduction to Trigonometry',
    '9': 'Some Applications of Trigonometry',
    '10': 'Circles',
    '11': 'Constructions',
    '12': 'Areas Related to Circles',
    '13': 'Surface Areas and Volumes',
    '14': 'Statistics',
    '15': 'Probability',
}

CHAPTER_FOCUS: Dict[str, str] = {
    'REAL NUMBERS': 'divisibility arguments and Euclid algorithms',
    'POLYNOMIALS': 'links between zeroes and coefficients',
    'PAIR OF LINEAR EQUATIONS IN TWO VARIABLES': 'graphical and algebraic solving',
    'QUADRATIC EQUATIONS': 'solution methods and nature of roots',
    'ARITHMETIC PROGRESSIONS': 'series manipulation',
    'TRIANGLES': 'similarity proofs and ratio deductions',
    'COORDINATE GEOMETRY': 'distance, section, and area formulae',
    'INTRODUCTION TO TRIGONOMETRY': 'computing trig ratios and identities',
    'SOME APPLICATIONS OF TRIGONOMETRY': 'height and distance problems',
    'CIRCLES': 'tangent and chord relationships',
    'CONSTRUCTIONS': 'ruler-compass builds',
    'AREAS RELATED TO CIRCLES': 'sector and segment computations',
    'SURFACE AREAS AND VOLUMES': '3D mensuration',
    'STATISTICS': 'central tendency and graphs',
    'PROBABILITY': 'outcome counting',
}

FOCUS_KEYWORDS: Dict[str, Sequence[str]] = {
    'euclid division': ['euclid', 'division algorithm'],
    'irrational number proof': ['irrational', 'root'],
    'quadratic roots': ['quadratic', 'root'],
    'ap word problem': ['arithmetic progression', 'ap', 'common difference'],
    'similar triangle proof': ['similar', 'ratio', 'triangles'],
    'trigonometric ratio': ['sine', 'cosine', 'tangent', 'trigonometric'],
    'height and distance': ['elevation', 'depression', 'flagstaff', 'tower'],
    'circle tangents': ['tangent', 'secant', 'circle'],
    'surface area': ['surface area', 'curved surface', 'total surface'],
    'volume composition': ['volume', 'solid', 'cone', 'cylinder'],
    'statistics table': ['mean', 'median', 'mode', 'cumulative'],
    'probability experiment': ['probability', 'coin', 'die', 'cards'],
}

QUESTION_TYPE_RULES: Sequence[Tuple[str, Sequence[str]]] = (
    ('Proof', ('prove', 'show that', 'hence prove', 'demonstrate')), 
    ('Construction', ('construct', 'draw', 'bisect', 'locate point')),
    ('Computation', ('find', 'calculate', 'determine', 'evaluate', 'work out')),
    ('Application', ('word problem', 'height', 'distance', 'real life', 'contextual')),
)

TOKEN_PATTERN = re.compile(r'[a-z0-9]+')
CHAPTER_PATTERN = re.compile(r'^CHAPTER\s+(\d+)\s+(.+)$', re.IGNORECASE)
EXERCISE_PATTERN = re.compile(r'^EXERCISE\s+(\d+\.\d+)', re.IGNORECASE)
QUESTION_PATTERN = re.compile(r'^(?:Q\s*)?(\d+)\.(?!\d)\s*(.+)$')


def _sanitize(text: str) -> str:
    clean = text.encode('ascii', 'ignore').decode('ascii', 'ignore')
    clean = clean.replace('\xa0', ' ')
    clean = ' '.join(clean.split())
    return clean.strip()


def _tokenize(text: str) -> Set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _classify_question(text: str) -> str:
    lowered = text.lower()
    for label, phrases in QUESTION_TYPE_RULES:
        for phrase in phrases:
            if phrase in lowered:
                return label
    return 'Conceptual'


def _derive_focus(text: str, chapter_key: str) -> str:
    lowered = text.lower()
    for focus, keywords in FOCUS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                return focus
    return CHAPTER_FOCUS.get(chapter_key, '')


class TextbookKnowledgeBase:
    """Parses the PDF once and exposes simple search helpers."""

    def __init__(self, pdf_path: Optional[str] = None, cache_path: Optional[str] = None):
        self.pdf_path = pdf_path or self._resolve_pdf_path()
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self._ensure_data_dir()
        self.index = self._load_index()
        self.questions = self._attach_runtime_fields(self.index.get('questions', []))

    @staticmethod
    def _resolve_pdf_path() -> str:
        for cand in PDF_CANDIDATES:
            if os.path.exists(cand):
                return cand
        return ''

    @staticmethod
    def _ensure_data_dir() -> None:
        os.makedirs(DATA_DIR, exist_ok=True)

    @staticmethod
    def _chapter_from_exercise(exercise_label: str, fallback_number: str, fallback_title: str) -> Tuple[str, str]:
        match = re.search(r'(\d+)', exercise_label or '')
        if match:
            chapter_id = match.group(1)
            chapter_name = CHAPTER_INFO.get(chapter_id)
            if chapter_name:
                return chapter_id, chapter_name
            if fallback_title:
                return chapter_id, fallback_title.title()
            return chapter_id, ''
        if fallback_number or fallback_title:
            return fallback_number, fallback_title.title() if fallback_title else fallback_title
        return '', ''

    def _load_index(self) -> Dict[str, object]:
        if not self.pdf_path or not os.path.exists(self.pdf_path) or PdfReader is None:
            return {'questions': []}
        source_mtime = os.path.getmtime(self.pdf_path)
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as handle:
                    cached = json.load(handle)
                if cached.get('source_mtime') == source_mtime:
                    return cached
            except Exception:
                pass
        built = self._build_index(source_mtime)
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as handle:
                json.dump(built, handle, indent=2)
        except Exception:
            pass
        return built

    def _build_index(self, source_mtime: float) -> Dict[str, object]:
        reader = PdfReader(self.pdf_path)
        lines: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ''
            for raw_line in text.splitlines():
                cleaned = _sanitize(raw_line)
                if cleaned:
                    lines.append(cleaned)

        chapter_number = ''
        chapter_title = ''
        exercise_label = ''
        current_question: Optional[Dict[str, str]] = None
        questions: List[Dict[str, str]] = []

        def commit_question() -> None:
            nonlocal current_question
            if not current_question:
                return
            body = ' '.join(current_question['text'].split())
            # Don't truncate - we need full text for solution splitting
            chapter_id, chapter_name = self._chapter_from_exercise(exercise_label, chapter_number, chapter_title)
            chapter_key = (chapter_name or chapter_title).upper()
            base_topics = CHAPTER_TOPICS.get(chapter_key, [])
            focus = _derive_focus(body, chapter_key)
            topics = list(dict.fromkeys(list(base_topics) + ([focus] if focus else [])))
            entry = {
                'chapter_number': chapter_id,
                'chapter_title': chapter_name or chapter_title.title(),
                'exercise': exercise_label,
                'question_number': current_question['number'],
                'question_text': body,
                'question_type': _classify_question(body),
                'focus': focus,
                'topics': topics,
            }
            questions.append(entry)
            current_question = None

        for line in lines:
            chap_match = CHAPTER_PATTERN.match(line)
            if chap_match:
                commit_question()
                chapter_number = chap_match.group(1)
                chapter_title = chap_match.group(2).strip()
                exercise_label = ''
                continue

            ex_match = EXERCISE_PATTERN.match(line)
            if ex_match:
                commit_question()
                exercise_label = f"Exercise {ex_match.group(1)}"
                continue

            q_match = QUESTION_PATTERN.match(line)
            if q_match and exercise_label:
                commit_question()
                current_question = {
                    'number': q_match.group(1),
                    'text': q_match.group(2),
                }
                continue

            if current_question:
                current_question['text'] += f" {line}"

        commit_question()
        return {
            'source': self.pdf_path,
            'source_mtime': source_mtime,
            'questions': questions,
        }

    def _attach_runtime_fields(self, entries: List[Dict[str, str]]) -> List[Dict[str, object]]:
        enriched: List[Dict[str, object]] = []
        for entry in entries:
            tokens = _tokenize(entry['question_text'] + ' ' + ' '.join(entry.get('topics', [])))
            enriched_entry = dict(entry)
            enriched_entry['tokens'] = tokens
            enriched_entry['insights'] = self._build_insights(enriched_entry)
            enriched.append(enriched_entry)
        return enriched

    def _build_insights(self, entry: Dict[str, object]) -> List[str]:
        focus = entry.get('focus') or CHAPTER_FOCUS.get(entry['chapter_title'].upper(), '')
        focus_text = focus if isinstance(focus, str) else ''
        info = [
            f"Located in {entry.get('exercise', 'Exercise')} from Chapter {entry.get('chapter_number', '')} {entry.get('chapter_title', '')}.",
            f"Question type: {entry.get('question_type')} focused on {focus_text or 'core concepts'}.",
        ]
        if entry.get('topics'):
            info.append(f"Core ideas: {', '.join(entry['topics'])}.")
        return info

    def search(self, query: str, max_results: int = 3) -> Optional[Dict[str, List[Dict[str, object]]]]:
        if not query or not self.questions:
            return None
        query_tokens = _tokenize(query)
        if not query_tokens:
            return None

        scored: List[Tuple[float, Dict[str, object]]] = []
        for entry in self.questions:
            overlap = len(query_tokens & entry['tokens'])
            if overlap == 0:
                continue
            if 'prove' in query_tokens and entry['question_type'] == 'Proof':
                overlap += 2
            if 'diagram' in query_tokens and 'diagram' in entry['question_text'].lower():
                overlap += 1
            scored.append((float(overlap), entry))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        primary_score, primary_entry = scored[0]
        return {
            'match': [self._format_entry(primary_entry, primary_score)],
            'related': [self._format_entry(entry, float(score)) for score, entry in scored[1:max_results]],
        }

    @staticmethod
    def _format_entry(entry: Dict[str, object], score: float) -> Dict[str, object]:
        return {
            'chapter_number': entry.get('chapter_number', ''),
            'chapter_title': entry.get('chapter_title', ''),
            'exercise': entry.get('exercise', ''),
            'question_number': entry.get('question_number', ''),
            'question_text': entry.get('question_text', ''),
            'question_type': entry.get('question_type', ''),
            'focus': entry.get('focus', ''),
            'topics': entry.get('topics', []),
            'insights': entry.get('insights', []),
            'score': score,
        }


def textbook_lookup(query: str) -> Optional[Dict[str, object]]:
    """Helper to perform a one-off lookup without instantiating the class manually."""
    kb = TextbookKnowledgeBase()
    search_result = kb.search(query)
    if not search_result:
        return None
    primary = search_result['match'][0]
    related = search_result['related']
    return {
        'match': primary,
        'related': related,
    }
