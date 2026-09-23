"""Deprecated import compatibility; implementation lives under vmr.compiler."""
import sys
from vmr.compiler.methods.hierarchical import build as _implementation
sys.modules[__name__] = _implementation
