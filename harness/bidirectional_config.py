"""Deprecated import compatibility; implementation lives under vmr.compiler."""
import sys
from vmr.compiler.methods.bidirectional import config as _implementation
sys.modules[__name__] = _implementation
