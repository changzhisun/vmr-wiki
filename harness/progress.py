"""Compatibility alias; implementation lives in vmr.core.progress."""
import sys
import vmr.core.progress as _implementation

sys.modules[__name__] = _implementation
