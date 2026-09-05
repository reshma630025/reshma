"""
TrustGuard AI - Root Entry Point
Allows starting the server via:
  python main.py
  python -m uvicorn main:app --host 127.0.0.1 --port 8000
  python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
import sys
from pathlib import Path
import uvicorn

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

if __name__ == "__main__":
    print("=" * 60)
    print("  TrustGuard AI SOC Server Starting on http://127.0.0.1:8000")
    print("  Open your browser at: http://127.0.0.1:8000/")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
