"""Deprecated compatibility entry point. New experiments use vmr query."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vmr.compat import run_query as _implementation
if __name__ == "__main__":
    import warnings
    warnings.warn("Legacy harness CLI is deprecated; use python -m vmr", FutureWarning)
    if "--wiki-set" in sys.argv:
        from vmr.cli import entrypoint
        sys.argv.insert(1, "query")
        entrypoint()
    else:
        _implementation.cli(_implementation.main)
else:
    sys.modules[__name__] = _implementation
