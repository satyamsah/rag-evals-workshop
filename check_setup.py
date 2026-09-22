"""
check_setup.py — run this before the workshop to verify your environment.

  python check_setup.py
"""

import os
import sys

errors = []

# ── Python version ────────────────────────────────────────────────────────
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 10):
    errors.append(f"Python 3.10+ required (you have {major}.{minor})")
else:
    print(f"✓ Python {major}.{minor}")

# ── .env / API key ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ python-dotenv installed")
except ImportError:
    errors.append("python-dotenv not installed — run: pip install -r requirements.txt")

key = os.getenv("ANTHROPIC_API_KEY", "")
if not key or key.startswith("sk-ant-your"):
    errors.append("ANTHROPIC_API_KEY not set — copy .env.example to .env and add your key")
else:
    print("✓ ANTHROPIC_API_KEY found")

# ── Anthropic SDK ─────────────────────────────────────────────────────────
try:
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "ping"}],
    )
    print("✓ Anthropic API reachable")
except Exception as e:
    errors.append(f"Anthropic API error: {e}")

# ── RAGAS ─────────────────────────────────────────────────────────────────
try:
    import ragas  # noqa: F401
    print("✓ ragas installed")
except ImportError:
    errors.append("ragas not installed — run: pip install -r requirements.txt")

# ── datasets ─────────────────────────────────────────────────────────────
try:
    import datasets  # noqa: F401
    print("✓ datasets installed")
except ImportError:
    errors.append("datasets not installed — run: pip install -r requirements.txt")

# ── FAISS ────────────────────────────────────────────────────────────────
try:
    import faiss  # noqa: F401
    print("✓ faiss-cpu installed")
except ImportError:
    errors.append("faiss-cpu not installed — run: pip install -r requirements.txt")

# ── Result ────────────────────────────────────────────────────────────────
print()
if errors:
    print("✗ Setup incomplete:")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("✓ Ready for the workshop!")
