"""Deprecated compatibility module; use vmr compile for sealed artifacts."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmr.compat import ingest as _implementation
if __name__ == "__main__":
    import warnings
    warnings.warn("Use vmr compile; ingest CLI is deprecated", FutureWarning)
    _implementation.cli(_implementation.main)
else:
    sys.modules[__name__] = _implementation
