"""Ensure project root is importable in CI (where it isn't installed as a package)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
