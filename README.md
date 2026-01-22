# 🔷 Chris - ML-Powered Geometry & Math Solver

## 🎯 Overview

An intelligent math assistant named **Chris** that uses **Machine Learning** for intent classification, **Advanced Regex** for entity extraction, and features a **web interface with voice navigation**. Chris can solve geometry problems, evaluate math expressions, and explain solutions step-by-step.

### ✨ Key Features

| Feature                         | Description                                                |
| ------------------------------- | ---------------------------------------------------------- |
| 🧠 **ML Intent Classification** | Neural Network understands what you want to calculate      |
| 🔍 **Entity Extraction**        | Advanced regex extracts shapes, measurements, and units    |
| 🌐 **Web Interface**            | Beautiful dark-themed web UI at `localhost:5000`           |
| 🎤 **Voice Input**              | Speak your questions using the microphone                  |
| 🔊 **Voice Output**             | Hear step-by-step explanations read aloud                  |
| ⌨️ **Keyboard Shortcuts**       | Hold spacebar to speak, double-tap to skip                 |
| 📐 **Geometry**                 | Circles, spheres, cubes, cylinders, cones, boxes, and more |
| 🔢 **Math Expressions**         | Trigonometry, powers, roots, logarithms, factorials        |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ml-decisioning-project
pip install -r requirements.txt
```

### 2. Train the ML Models (first time only)

```bash
python src/main.py --train
```

### 3. Run the Web Interface

```bash
python src/web_app.py
```

Then open **http://localhost:5000** in your browser (Chrome/Edge recommended for voice).

### 3.1 (Optional) Enable Gemini Word-Problem Support

Some advanced word-problem explanations in the web UI can use Google Gemini if configured.

1. Install the Gemini client library:

```bash
pip install google-generativeai
```

2. Provide your Gemini API key (the project never hard-codes it):

**Option A – .env file (recommended)**

Create a `.env` file in the `ml-decisioning-project` folder (next to this README) based on `.env.example`:

```bash
cp .env.example .env   # or copy manually on Windows
```

Edit `.env` and set your key:

```bash
GEMINI_API_KEY=your-api-key-here
```

**Option B – Environment variable**

Windows (PowerShell, current session):

```powershell
$env:GEMINI_API_KEY = "your-api-key-here"
```

macOS/Linux (bash/zsh):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

If `GEMINI_API_KEY` is not set or `google-generativeai` is not installed, the app will simply run without Gemini features.

### 4. Alternative: Terminal Mode

```bash
# Interactive terminal mode
python src/main.py --interactive

# Quick demo
python src/main.py --demo
```

## 🎤 Voice Navigation

Chris features full voice support in the web interface:

| Action                 | How To                                      |
| ---------------------- | ------------------------------------------- |
| **Start voice input**  | Hold spacebar for 2 seconds, or click 🎤    |
| **Skip to answer**     | Double-tap spacebar while Chris is speaking |
| **Skip intro**         | Double-tap spacebar when page loads         |
| **Replay explanation** | Click the 🔊 button on any result           |

> **Note**: Voice features work best in Google Chrome or Microsoft Edge.

## 📐 What Chris Can Solve

### Geometry Shapes

| Shape      | Example Query                  | Properties Calculated              |
| ---------- | ------------------------------ | ---------------------------------- |
| Circle     | "area of circle with radius 5" | area, circumference, diameter      |
| Sphere     | "volume of sphere radius 3"    | volume, surface area               |
| Rectangle  | "rectangle length 10 width 5"  | area, perimeter, diagonal          |
| Square     | "square side 7"                | area, perimeter, diagonal          |
| Cube       | "cube side 4"                  | volume, surface area, diagonal     |
| Cylinder   | "cylinder radius 3 height 10"  | volume, surface areas              |
| Cone       | "cone radius 4 height 6"       | volume, slant height, surface area |
| Box/Cuboid | "box with 5, 6 and 7"          | volume, surface area, diagonal     |

### Math Expressions

| Type         | Example               | Result |
| ------------ | --------------------- | ------ |
| Trigonometry | "sin 90 + cos 0"      | 2.0    |
| Square Root  | "square root of 144"  | 12     |
| Powers       | "2 to the power of 8" | 256    |
| Logarithms   | "log 100"             | 2.0    |
| Factorials   | "5!"                  | 120    |
| Percentages  | "20% of 150"          | 30     |

## 📁 Project Structure

```
ml-decisioning-project/
├── src/
│   ├── main.py              # CLI entry point
│   ├── web_app.py           # Flask web server
│   ├── inference.py         # Inference API
│   ├── math_calculator.py   # Math expression evaluator
│   ├── ml/
│   │   └── model.py         # ML intent classifier
│   ├── preprocessing/
│   │   ├── pipeline.py      # Text preprocessing
│   │   └── entity_extractor.py  # Regex extraction
│   ├── rules/
│   │   └── evaluator.py     # Geometry calculations
│   └── templates/
│       └── index.html       # Web UI with voice
├── models/
│   └── baseline/            # Trained ML models
├── data/
│   └── processed/           # Training data
├── requirements.txt
└── README.md

Additional helper:

- `test_gemini.py` – optional developer script to list available Gemini models. It uses the `GEMINI_API_KEY` environment variable and **does not** contain any API keys.
```

## 🔧 Command Line Options

```bash
python src/main.py [options]

Options:
  --train, -t         Train the ML models
  --interactive, -i   Run interactive terminal mode
  --demo, -d          Run a quick demonstration
  --test              Run component tests
  --tts               Enable text-to-speech in terminal
  --strategy          Decision strategy: ensemble, ml_priority, regex_priority, cascading
  --samples           Training samples per intent (default: 800)
  --model-type        Model type: mlp (neural network), rf (random forest), gb (gradient boosting)
```

## 💻 Programmatic Usage

```python
from src.inference import GeometryInferenceEngine

# Initialize
engine = GeometryInferenceEngine(model_path='models/baseline')

# Solve a geometry problem
result = engine.process("sphere radius 3")

print(f"Shape: {result.shape}")
print(f"Confidence: {result.confidence:.1%}")
for calc in result.results:
    print(f"{calc['property']}: {calc['value']:.4f} {calc['unit']}")
```

## 🧪 Testing

```bash
# Run all component tests
python src/main.py --test

# Run pytest tests
python -m pytest tests/
```

## 📊 Model Performance

After training with 800 samples per intent:

| Metric          | Score |
| --------------- | ----- |
| Intent Accuracy | ~95%  |
| Shape Accuracy  | ~92%  |

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn (MLP Neural Network)
- **NLP**: TF-IDF Vectorization, Advanced Regex
- **Frontend**: HTML/CSS/JavaScript
- **Voice**: Web Speech API (browser-based)

## 📝 Requirements

- Python 3.8+
- Chrome or Edge browser (for voice features)
- See `requirements.txt` for Python packages

## 📄 License

MIT License - Feel free to use and modify.

---

**Made with ❤️ by Chris the Math Assistant**
