"""Deprecated compatibility entry point. New experiments use vmr query."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmr.compat import workspace as _implementation
if __name__ == "__main__":
    import warnings
    warnings.warn("Legacy harness CLI is deprecated; use python -m vmr", FutureWarning)
    raise SystemExit("Use vmr query")
else:
    sys.modules[__name__] = _implementation
