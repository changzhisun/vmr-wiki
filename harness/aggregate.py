"""Compatibility entry point for vmr.evaluation.aggregate."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vmr.evaluation.aggregate as _implementation
if __name__ == "__main__":
    _implementation.cli(_implementation.main)
else:
    sys.modules[__name__] = _implementation
