"""Compatibility alias; implementation lives in vmr.vlm.transport."""
import sys
import vmr.vlm.transport as _implementation

sys.modules[__name__] = _implementation
