"""Test Gemini API to find usable model names.

This helper script does **not** contain any API keys.

You can provide a key either via:
- a `.env` file in the project root (containing GEMINI_API_KEY=...), or
- an environment variable `GEMINI_API_KEY`.
"""

import os
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("google-generativeai is not installed. Install with 'pip install google-generativeai'.")
    raise SystemExit(1)


def _load_dotenv():
    """Best-effort loader for a local .env file in the project root."""
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:  # pragma: no cover - helper is best-effort
        print(f"Warning: could not read .env file: {exc}")


_load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("GEMINI_API_KEY environment variable is not set. Skipping Gemini test.")
    raise SystemExit(1)

try:
    genai.configure(api_key=API_KEY)

    print("Listing available Gemini models:")
    models = genai.list_models()

    for m in models:
        if "generateContent" in m.supported_generation_methods:
            print(f"\nModel: {m.name}")
            print(f"  Display name: {m.display_name}")

    print("\n" + "=" * 50)
    print("Testing first available model...")

    # Try to use the first available model
    for m in models:
        if "generateContent" in m.supported_generation_methods:
            model_name = m.name
            print(f"\nTrying model: {model_name}")
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("What is 2+2? Answer in one word.")
                print(f"Response: {response.text}")
                print(f"SUCCESS! Use model name: {model_name}")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
